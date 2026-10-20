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
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model = None
        self.available = False
        self.error = None
        try:
            import lpips  # type: ignore
            self.model = lpips.LPIPS(net="alex").to(self.device)
            self.model.eval()
            self.available = True
        except Exception as e:  # noqa: BLE001
            self.error = str(e)

    def compute(self, pred: np.ndarray, gt: np.ndarray) -> Optional[float]:
        if not self.available or self.model is None:
            return None
        pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        gt_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
        pred_t = pred_t * 2.0 - 1.0
        gt_t = gt_t * 2.0 - 1.0
        with torch.no_grad():
            v = self.model(pred_t, gt_t).item()
        return float(v)


def _parse_stage_samples(stage_dir: str) -> Dict[str, Dict[str, str]]:
    samples: Dict[str, Dict[str, str]] = {}
    for name in os.listdir(stage_dir):
        if not name.endswith(".png"):
            continue
        m = re.match(r"^(cam\d+_frame\d{6})_(before|fixed|after_distill|ref)\.png$", name)
        if m is None:
            continue
        sample_id, kind = m.group(1), m.group(2)
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
    for _, files in sample_dict.items():
        if pred_key not in files or gt_key not in files:
            continue
        pred = _load_rgb(files[pred_key])
        gt = _load_rgb(files[gt_key])
        psnrs.append(_psnr(pred, gt))
        ssims.append(float(ssim(gt, pred, data_range=1.0, channel_axis=2)))
        lp = lpips_computer.compute(pred, gt)
        if lp is not None:
            lpips_vals.append(lp)
    return psnrs, ssims, lpips_vals


def _mean_or_none(values: List[float]) -> Optional[float]:
    if len(values) == 0:
        return None
    return float(np.mean(values))


def _collect_stage_dirs(exp_dir: str, stage_glob_prefix: str) -> List[str]:
    stage_dirs = []
    for name in sorted(os.listdir(exp_dir)):
        if name.startswith(stage_glob_prefix):
            p = os.path.join(exp_dir, name)
            if os.path.isdir(p):
                stage_dirs.append(p)
    return stage_dirs


def _render_all_frames_for_eval(args: argparse.Namespace) -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from datasets.driving_dataset_novel_view import DrivingDatasetNovelView
    from sim_render.cam.render_novel_trajectory import (
        _infer_novel_traj_types_from_experiment,
        run_difix_eval_all_frames,
    )
    from utils.misc import import_str

    exp_dir = os.path.abspath(args.experiment_dir)
    config_path = os.path.join(exp_dir, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.yaml not found under experiment_dir: {config_path}")

    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts or []))
    cfg.log_dir = exp_dir

    traj_types = list(args.traj_types or [])
    if len(traj_types) == 0:
        traj_types = _infer_novel_traj_types_from_experiment(
            exp_dir,
            stage_glob_prefix="difix_distill_all_frames_",
            ref_traj_type=args.difix_ref_traj_type,
        )
    if len(traj_types) == 0:
        raise ValueError(
            "No novel trajectory types found. Pass --traj_types or ensure "
            "difix_distill_all_frames_* folders exist under experiment_dir."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DrivingDatasetNovelView(data_cfg=cfg.data)
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_full_images=len(dataset.train_indices + dataset.test_indices),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )

    camera_ids = args.cam_ids if len(args.cam_ids) > 0 else cfg.data.pixel_source.cameras
    downscales = (
        args.downscales
        if len(args.downscales) > 0
        else cfg.data.pixel_source.downscale_when_loading
    )
    camera_data_dict = dataset.load_specified_cameras(camera_ids, downscales)

    render_args = argparse.Namespace(
        distill_cam_ids=args.distill_cam_ids,
        distill_ref_cam_id=args.distill_ref_cam_id,
        traj_path=args.traj_path,
        difix_device=args.difix_device,
        difix_use_original_traj_ref=args.difix_use_original_traj_ref,
        difix_ref_video_path=args.difix_ref_video_path,
        difix_ref_video_dir=args.difix_ref_video_dir,
        difix_ref_traj_type=args.difix_ref_traj_type,
        difix_src_dir=args.difix_src_dir,
        difix_pretrained_dir=args.difix_pretrained_dir,
        difix_prompt=args.difix_prompt,
        difix_num_inference_steps=args.difix_num_inference_steps,
        difix_timesteps=args.difix_timesteps,
        difix_guidance_scale=args.difix_guidance_scale,
        difix_trust_remote_code=args.difix_trust_remote_code,
    )

    print(
        f"[RENDER] Full-frame eval export for {len(traj_types)} novel trajectories: "
        f"{', '.join(traj_types)}"
    )
    save_dirs = run_difix_eval_all_frames(
        cfg=cfg,
        trainer=trainer,
        dataset=dataset,
        camera_data_dict=camera_data_dict,
        args=render_args,
        traj_types=traj_types,
        ckpt_path_before=os.path.abspath(args.ckpt_path_before),
        ckpt_path_after=os.path.abspath(args.ckpt_path),
        eval_stage_prefix=args.eval_stage_prefix,
    )
    for save_dir in save_dirs:
        num_pngs = len([n for n in os.listdir(save_dir) if n.endswith(".png")])
        print(f"[RENDER] {save_dir}: {num_pngs} PNGs")


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate novel trajectory quality using DiFix outputs as pseudo GT")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="experiment output dir containing difix_distill_all_frames_* or difix_eval_all_frames_* folders",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="directory for result json; defaults to <experiment_dir>/metrics_novel_difix",
    )
    parser.add_argument(
        "--stage_glob_prefix",
        type=str,
        default="",
        help="prefix used to find stage folders; defaults to difix_eval_all_frames_ when --render_all_frames",
    )
    parser.add_argument(
        "--render_all_frames",
        action="store_true",
        help="render all timeline frames (with DiFix pseudo-GT) before computing metrics",
    )
    parser.add_argument(
        "--skip_render_all_frames",
        action="store_true",
        help="skip full-frame rendering and only read existing PNG folders",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="distilled checkpoint for after_distill renders (required with --render_all_frames)",
    )
    parser.add_argument(
        "--ckpt_path_before",
        type=str,
        default="",
        help="pre-distill checkpoint for before + DiFix pseudo-GT (required with --render_all_frames)",
    )
    parser.add_argument(
        "--eval_stage_prefix",
        type=str,
        default="difix_eval_all_frames_",
        help="output folder prefix for full-frame eval PNGs",
    )
    parser.add_argument(
        "--traj_types",
        nargs="+",
        default=None,
        help="novel trajectory names to evaluate; auto-inferred from distill folders if omitted",
    )
    parser.add_argument(
        "--cam_ids",
        nargs="*",
        type=int,
        default=[],
        help="camera ids for eval rendering; defaults to config",
    )
    parser.add_argument(
        "--downscales",
        nargs="*",
        type=float,
        default=[],
        help="downscales for eval rendering; defaults to config",
    )
    parser.add_argument(
        "--distill_cam_ids",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12],
        help="camera ids used for novel-view eval export",
    )
    parser.add_argument(
        "--distill_ref_cam_id",
        type=int,
        default=0,
        help="reference camera id for novel trajectory generation",
    )
    parser.add_argument(
        "--traj_path",
        type=str,
        default=None,
        help="directory of self-defined trajectories (cam2world)",
    )
    parser.add_argument(
        "--difix_src_dir",
        type=str,
        default="/nas/oldbak/ga/code/Difix3D-main/src",
    )
    parser.add_argument(
        "--difix_pretrained_dir",
        type=str,
        default="/nas/oldbak/ga/code/Difix3D-main/difix_ref",
    )
    parser.add_argument(
        "--difix_device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--difix_prompt",
        type=str,
        default="remove_degradation",
    )
    parser.add_argument(
        "--difix_num_inference_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--difix_timesteps",
        nargs="+",
        type=int,
        default=[199],
    )
    parser.add_argument(
        "--difix_guidance_scale",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--difix_use_original_traj_ref",
        action="store_true",
        help="use original-trajectory GT videos as DiFix ref_image",
    )
    parser.add_argument(
        "--difix_ref_traj_type",
        type=str,
        default="original_traj",
    )
    parser.add_argument(
        "--difix_ref_video_dir",
        type=str,
        default="",
        help="per-cam GT ref videos (cam<id>.mp4)",
    )
    parser.add_argument(
        "--difix_ref_video_path",
        type=str,
        default="",
        help="single GT ref video shared by all cams",
    )
    parser.add_argument(
        "--difix_trust_remote_code",
        action="store_true",
    )
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        help="extra OmegaConf overrides, e.g. log_dir=...",
    )
    parser.add_argument(
        "--lpips_device",
        type=str,
        default="cuda",
        help="device for LPIPS model: cuda or cpu",
    )
    args = parser.parse_args()

    exp_dir = os.path.abspath(args.experiment_dir)
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"experiment_dir not found: {exp_dir}")

    render_all_frames = args.render_all_frames
    if render_all_frames:
        if not args.ckpt_path:
            raise ValueError("--ckpt_path is required when rendering all frames for eval.")
        if not args.ckpt_path_before:
            raise ValueError(
                "--ckpt_path_before is required when rendering all frames for eval "
                "(used for before render + DiFix pseudo-GT)."
            )
        if not os.path.isfile(args.ckpt_path):
            raise FileNotFoundError(f"ckpt_path not found: {args.ckpt_path}")
        if not os.path.isfile(args.ckpt_path_before):
            raise FileNotFoundError(f"ckpt_path_before not found: {args.ckpt_path_before}")
        if args.difix_use_original_traj_ref and not args.difix_ref_video_dir and not args.difix_ref_video_path:
            default_ref_dir = os.path.join(exp_dir, "refer_gt_video")
            if os.path.isdir(default_ref_dir):
                args.difix_ref_video_dir = default_ref_dir
            else:
                raise ValueError(
                    "--difix_use_original_traj_ref is set but no ref video dir/path found. "
                    "Set --difix_ref_video_dir or ensure refer_gt_video exists under experiment_dir."
                )
        _render_all_frames_for_eval(args)

    stage_glob_prefix = args.stage_glob_prefix.strip()
    if not stage_glob_prefix:
        if render_all_frames or len(_collect_stage_dirs(exp_dir, args.eval_stage_prefix)) > 0:
            stage_glob_prefix = args.eval_stage_prefix
        else:
            stage_glob_prefix = "difix_distill_all_frames_"

    output_dir = args.output_dir.strip() or os.path.join(exp_dir, "metrics_novel_difix")
    os.makedirs(output_dir, exist_ok=True)

    stage_dirs = _collect_stage_dirs(exp_dir, stage_glob_prefix)
    if len(stage_dirs) == 0:
        raise FileNotFoundError(
            f"No stage folders found in {exp_dir} with prefix '{stage_glob_prefix}'"
        )

    lpips_computer = _LpipsComputer(args.lpips_device)
    all_before_psnr: List[float] = []
    all_before_ssim: List[float] = []
    all_before_lpips: List[float] = []
    all_after_psnr: List[float] = []
    all_after_ssim: List[float] = []
    all_after_lpips: List[float] = []

    stage_results = {}
    for stage_dir in stage_dirs:
        stage_name = os.path.basename(stage_dir)
        samples = _parse_stage_samples(stage_dir)
        before_psnr, before_ssim, before_lpips = _compute_pair_metrics(
            samples, pred_key="before", gt_key="fixed", lpips_computer=lpips_computer
        )
        after_psnr, after_ssim, after_lpips = _compute_pair_metrics(
            samples, pred_key="after_distill", gt_key="fixed", lpips_computer=lpips_computer
        )

        all_before_psnr.extend(before_psnr)
        all_before_ssim.extend(before_ssim)
        all_before_lpips.extend(before_lpips)
        all_after_psnr.extend(after_psnr)
        all_after_ssim.extend(after_ssim)
        all_after_lpips.extend(after_lpips)

        stage_results[stage_name] = {
            "num_samples_with_before_fixed": len(before_psnr),
            "num_samples_with_after_fixed": len(after_psnr),
            "before_vs_fixed": {
                "psnr": _mean_or_none(before_psnr),
                "ssim": _mean_or_none(before_ssim),
                "lpips": _mean_or_none(before_lpips),
            },
            "after_vs_fixed": {
                "psnr": _mean_or_none(after_psnr),
                "ssim": _mean_or_none(after_ssim),
                "lpips": _mean_or_none(after_lpips),
            },
        }

    summary = {
        "experiment_dir": exp_dir,
        "eval_mode": "all_frames" if render_all_frames else "existing_pngs",
        "stage_glob_prefix": stage_glob_prefix,
        "lpips_available": lpips_computer.available,
        "lpips_error": lpips_computer.error,
        "overall": {
            "num_samples_with_before_fixed": len(all_before_psnr),
            "num_samples_with_after_fixed": len(all_after_psnr),
            "before_vs_fixed": {
                "psnr": _mean_or_none(all_before_psnr),
                "ssim": _mean_or_none(all_before_ssim),
                "lpips": _mean_or_none(all_before_lpips),
            },
            "after_vs_fixed": {
                "psnr": _mean_or_none(all_after_psnr),
                "ssim": _mean_or_none(all_after_ssim),
                "lpips": _mean_or_none(all_after_lpips),
            },
        },
        "stages": stage_results,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"novel_difix_pseudo_gt_metrics_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Saved metrics: {out_path}")
    overall = summary["overall"]
    print(
        "[OVERALL] before_vs_fixed: "
        f"PSNR={overall['before_vs_fixed']['psnr']}, "
        f"SSIM={overall['before_vs_fixed']['ssim']}, "
        f"LPIPS={overall['before_vs_fixed']['lpips']}, "
        f"n={overall['num_samples_with_before_fixed']}"
    )
    print(
        "[OVERALL] after_vs_fixed: "
        f"PSNR={overall['after_vs_fixed']['psnr']}, "
        f"SSIM={overall['after_vs_fixed']['ssim']}, "
        f"LPIPS={overall['after_vs_fixed']['lpips']}, "
        f"n={overall['num_samples_with_after_fixed']}"
    )


if __name__ == "__main__":
    main()
