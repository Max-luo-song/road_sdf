#!/usr/bin/env python3
"""
Compute metrics from existing renders (no 3DGS re-rendering).

1) Novel left_shift_3m (pseudo-GT): read difix_distill PNG triplets
   (before / fixed / after_distill) and report before_vs_fixed & after_vs_fixed.

2) Original trajectory (real GT): decode per-cam MP4 under novel_traj/ and compare
   to dataset GT images from config.yaml.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def _load_rgb(path: str) -> np.ndarray:
    arr = imageio.imread(path)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    return np.clip(arr.astype(np.float32) / 255.0, 0.0, 1.0)


def _psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = np.mean((pred - gt) ** 2, dtype=np.float64)
    if mse <= 1e-12:
        return 100.0
    return float(10.0 * math.log10(1.0 / mse))


class _LpipsComputer:
    def __init__(self, device: str):
        self.device = torch.device(
            device if device == "cpu" or torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self.available = False
        self.error = None
        try:
            import lpips  # type: ignore

            self.model = lpips.LPIPS(net="alex").to(self.device)
            self.model.eval()
            self.available = True
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)

    def compute(self, pred: np.ndarray, gt: np.ndarray) -> Optional[float]:
        if not self.available or self.model is None:
            return None
        pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        gt_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
        pred_t = pred_t * 2.0 - 1.0
        gt_t = gt_t * 2.0 - 1.0
        with torch.no_grad():
            return float(self.model(pred_t, gt_t).item())


def _mean_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(values))


def _aggregate_metrics(
    psnrs: List[float], ssims: List[float], lpips_vals: List[float]
) -> Dict[str, Optional[float]]:
    return {
        "psnr": _mean_or_none(psnrs),
        "ssim": _mean_or_none(ssims),
        "lpips": _mean_or_none(lpips_vals),
        "num_samples": len(psnrs),
    }


def _parse_stage_samples(stage_dir: str) -> Dict[str, Dict[str, str]]:
    samples: Dict[str, Dict[str, str]] = {}
    for name in os.listdir(stage_dir):
        if not name.endswith(".png"):
            continue
        match = re.match(r"^(cam\d+_frame\d{6})_(before|fixed|after_distill|ref)\.png$", name)
        if match is None:
            continue
        sample_id, kind = match.group(1), match.group(2)
        samples.setdefault(sample_id, {})[kind] = os.path.join(stage_dir, name)
    return samples


def _compute_pair_metrics(
    sample_dict: Dict[str, Dict[str, str]],
    pred_key: str,
    gt_key: str,
    lpips_computer: _LpipsComputer,
) -> Tuple[List[float], List[float], List[float]]:
    psnrs: List[float] = []
    ssims: List[float] = []
    lpips_vals: List[float] = []
    for files in sample_dict.values():
        if pred_key not in files or gt_key not in files:
            continue
        pred = _load_rgb(files[pred_key])
        gt = _load_rgb(files[gt_key])
        if pred.shape != gt.shape:
            continue
        psnrs.append(_psnr(pred, gt))
        ssims.append(float(ssim(gt, pred, data_range=1.0, channel_axis=2)))
        lp = lpips_computer.compute(pred, gt)
        if lp is not None:
            lpips_vals.append(lp)
    return psnrs, ssims, lpips_vals


def eval_novel_from_distill_pngs(
    stage_dir: str, lpips_computer: _LpipsComputer
) -> Dict[str, object]:
    if not os.path.isdir(stage_dir):
        raise FileNotFoundError(f"distill stage dir not found: {stage_dir}")

    samples = _parse_stage_samples(stage_dir)
    before_psnr, before_ssim, before_lpips = _compute_pair_metrics(
        samples, pred_key="before", gt_key="fixed", lpips_computer=lpips_computer
    )
    after_psnr, after_ssim, after_lpips = _compute_pair_metrics(
        samples, pred_key="after_distill", gt_key="fixed", lpips_computer=lpips_computer
    )

    return {
        "source": stage_dir,
        "metric_type": "pseudo_gt (render vs DiFix fixed)",
        "note": "Uses stride-3 PNGs saved during distillation, not layout MP4.",
        "before_vs_fixed": _aggregate_metrics(before_psnr, before_ssim, before_lpips),
        "after_vs_fixed": _aggregate_metrics(after_psnr, after_ssim, after_lpips),
    }


def _resolve_data_path(cfg) -> str:
    data_root = cfg.data.data_root
    scene_idx = cfg.data.scene_idx
    try:
        return os.path.join(data_root, f"{int(scene_idx):03d}")
    except (TypeError, ValueError):
        return os.path.join(data_root, str(scene_idx))


def _resolve_timestep_range(cfg, data_path: str) -> Tuple[int, int]:
    start = int(cfg.data.start_timestep)
    end_cfg = int(cfg.data.end_timestep)
    if end_cfg == -1:
        for sub in ("lidar_pose", "ego_pose"):
            pose_dir = os.path.join(data_path, sub)
            if os.path.isdir(pose_dir):
                end_cfg = len(os.listdir(pose_dir)) - 1
                break
        if end_cfg == -1:
            raise ValueError(f"Cannot infer end_timestep from {data_path}")
    return start, end_cfg + 1


def _find_traj_videos_dir(ckpt_dir: str, traj_type: str) -> str:
    novel_root = os.path.join(ckpt_dir, "novel_traj")
    if not os.path.isdir(novel_root):
        raise FileNotFoundError(f"novel_traj not found under: {ckpt_dir}")

    candidates: List[str] = []
    for name in sorted(os.listdir(novel_root)):
        if traj_type not in name:
            continue
        videos_dir = os.path.join(novel_root, name, "videos")
        if os.path.isdir(videos_dir):
            candidates.append(videos_dir)

    if not candidates:
        raise FileNotFoundError(
            f"No videos dir for traj '{traj_type}' under {novel_root}"
        )

    no_difix = [p for p in candidates if "_difix_" not in os.path.basename(os.path.dirname(p))]
    return (no_difix or candidates)[0]


def _load_gt_rgb(
    data_path: str,
    timestep: int,
    cam_id: int,
    target_hw: Tuple[int, int],
) -> np.ndarray:
    img_path = os.path.join(data_path, "images", f"{timestep:06d}_{cam_id}.png")
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"GT image not found: {img_path}")
    rgb = Image.open(img_path).convert("RGB")
    if rgb.size[1] != target_hw[0] or rgb.size[0] != target_hw[1]:
        rgb = rgb.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def _read_video_frames(video_path: str) -> List[np.ndarray]:
    reader = imageio.get_reader(video_path)
    frames = []
    for frame in reader:
        frames.append(_load_rgb(frame))
    reader.close()
    return frames


def eval_original_traj_from_videos(
    videos_dir: str,
    config_path: str,
    cam_ids: List[int],
    lpips_computer: _LpipsComputer,
    label: str,
) -> Dict[str, object]:
    cfg = OmegaConf.load(config_path)
    data_path = _resolve_data_path(cfg)
    start_ts, end_ts = _resolve_timestep_range(cfg, data_path)
    num_frames = end_ts - start_ts

    psnrs: List[float] = []
    ssims: List[float] = []
    lpips_vals: List[float] = []

    for cam_id in cam_ids:
        video_path = os.path.join(videos_dir, f"original_traj_cam{cam_id}_rgbs.mp4")
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"render video not found: {video_path}")

        frames = _read_video_frames(video_path)
        if len(frames) != num_frames:
            print(
                f"[WARN] {label} cam{cam_id}: video has {len(frames)} frames, "
                f"expected {num_frames}; using min length"
            )
        usable = min(len(frames), num_frames)
        for local_i in range(usable):
            timestep = start_ts + local_i
            pred = frames[local_i]
            gt = _load_gt_rgb(
                data_path,
                timestep,
                cam_id,
                target_hw=(pred.shape[0], pred.shape[1]),
            )
            psnrs.append(_psnr(pred, gt))
            ssims.append(float(ssim(gt, pred, data_range=1.0, channel_axis=2)))
            lp = lpips_computer.compute(pred, gt)
            if lp is not None:
                lpips_vals.append(lp)

    return {
        "source": videos_dir,
        "metric_type": "real_gt (render vs dataset GT on original trajectory)",
        "config": config_path,
        "data_path": data_path,
        "timesteps": [start_ts, end_ts - 1],
        "cam_ids": cam_ids,
        **_aggregate_metrics(psnrs, ssims, lpips_vals),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Metrics from existing renders (distill PNGs + novel_traj MP4s)."
    )
    parser.add_argument(
        "--experiment_dir",
        required=True,
        help="Repair output dir (after checkpoint / distill artifacts).",
    )
    parser.add_argument(
        "--ckpt_dir_before",
        required=True,
        help="Pre-repair checkpoint dir (scene_data .../checkpoint_final.pth parent).",
    )
    parser.add_argument(
        "--config_path",
        default="",
        help="config.yaml for GT loading; default: ckpt_dir_before/config.yaml",
    )
    parser.add_argument(
        "--distill_stage_dir",
        default="",
        help="left_shift_3m distill PNG dir; default: experiment_dir/difix_distill_all_frames_3_0",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="JSON output dir; default: experiment_dir/metrics_from_renders",
    )
    parser.add_argument(
        "--cam_ids",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12],
    )
    parser.add_argument("--lpips_device", default="cuda")
    parser.add_argument("--skip_novel", action="store_true")
    parser.add_argument("--skip_original", action="store_true")
    args = parser.parse_args()

    exp_dir = os.path.abspath(args.experiment_dir)
    ckpt_dir_before = os.path.abspath(args.ckpt_dir_before)
    config_path = os.path.abspath(args.config_path or os.path.join(ckpt_dir_before, "config.yaml"))
    output_dir = os.path.abspath(args.output_dir or os.path.join(exp_dir, "metrics_from_renders"))
    distill_stage_dir = os.path.abspath(
        args.distill_stage_dir or os.path.join(exp_dir, "difix_distill_all_frames_3_0")
    )

    os.makedirs(output_dir, exist_ok=True)
    lpips_computer = _LpipsComputer(args.lpips_device)

    summary: Dict[str, object] = {
        "experiment_dir": exp_dir,
        "ckpt_dir_before": ckpt_dir_before,
        "config_path": config_path,
        "lpips_available": lpips_computer.available,
        "lpips_error": lpips_computer.error,
    }

    if not args.skip_novel:
        print(f"[1/2] Novel left_shift_3m pseudo-GT metrics from: {distill_stage_dir}")
        summary["novel_left_shift_3m"] = eval_novel_from_distill_pngs(
            distill_stage_dir, lpips_computer
        )

    if not args.skip_original:
        before_videos = _find_traj_videos_dir(ckpt_dir_before, "original_traj")
        after_videos = _find_traj_videos_dir(exp_dir, "original_traj")
        print(f"[2/2] Original traj vs GT (before): {before_videos}")
        print(f"[2/2] Original traj vs GT (after):  {after_videos}")
        summary["original_traj"] = {
            "before_repair": eval_original_traj_from_videos(
                before_videos, config_path, args.cam_ids, lpips_computer, "before"
            ),
            "after_repair": eval_original_traj_from_videos(
                after_videos, config_path, args.cam_ids, lpips_computer, "after"
            ),
        }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"metrics_from_renders_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Saved: {out_path}")
    if "novel_left_shift_3m" in summary:
        novel = summary["novel_left_shift_3m"]
        for key in ("before_vs_fixed", "after_vs_fixed"):
            m = novel[key]
            print(
                f"[novel {key}] PSNR={m['psnr']}, SSIM={m['ssim']}, "
                f"LPIPS={m['lpips']}, n={m['num_samples']}"
            )
    if "original_traj" in summary:
        for key in ("before_repair", "after_repair"):
            m = summary["original_traj"][key]
            print(
                f"[original {key}] PSNR={m['psnr']}, SSIM={m['ssim']}, "
                f"LPIPS={m['lpips']}, n={m['num_samples']}"
            )


if __name__ == "__main__":
    main()
