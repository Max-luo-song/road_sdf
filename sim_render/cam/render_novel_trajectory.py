from typing import Dict, List, Optional, Tuple
from omegaconf import OmegaConf
import os
import re
import time
import logging
import argparse
import numpy as np
import imageio
import sys
import glob

import torch
import torch.nn.functional as F
from PIL import Image
from datasets.driving_dataset_novel_view import DrivingDatasetNovelView
from utils.misc import import_str
from utils.logging_utils import setup_logging
from models.trainers import BasicTrainer
from models.video_utils import render_novel_views, save_single_camera_video, save_videos
from utils.visualization import get_layout
from chery_tools.lidar_simulation import (
    unproject_depth_to_pointcloud,
    remove_ground_points,
    save_pointcloud_pcd,
)

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def _log_timing(message: str, *args) -> None:
    text = message % args if args else message
    logger.info(text)
    timing_log = os.environ.get("PIPELINE_TIMING_LOG", "").strip()
    if timing_log:
        with open(timing_log, "a") as f:
            f.write(text + "\n")


def _safe_float_arg(value):
    """
    Parse float args robustly even if a trailing inline comment leaks in.
    Example: "0.0####" -> 0.0
    """
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if "#" in text:
        text = text.split("#", 1)[0].strip()
    return float(text)


class Difix3DRefiner:
    """
    Placeholder wrapper for DiFix3D refinement.
    Replace the internals with your own model invocation.
    """

    def __init__(
        self,
        model=None,
        prompt: str = "remove degradation",
        num_inference_steps: int = 1,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 0.0,
    ):
        # Placeholder requested by user:
        # self.difix3d = model
        self.difix3d = model
        self.prompt = prompt
        self.num_inference_steps = num_inference_steps
        self.timesteps = timesteps if timesteps is not None else [199]
        self.guidance_scale = guidance_scale

    def refine(
        self,
        image: np.ndarray,
        ref_image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Args:
            image: HxWx3 float image in [0, 1] (novel / degraded view)
            ref_image: optional HxWx3 float [0,1], e.g. same timestep on original trajectory
        Returns:
            refined image: HxWx3 float image in [0, 1]
        """
        if self.difix3d is None:
            logger.warning("Difix3D model is None, fallback to identity refinement.")
            return image

        image_uint8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_uint8)

        call_kw = dict(
            num_inference_steps=self.num_inference_steps,
            timesteps=self.timesteps,
            guidance_scale=self.guidance_scale,
        )
        if ref_image is not None:
            ref_u8 = (np.clip(ref_image, 0.0, 1.0) * 255).astype(np.uint8)
            call_kw["ref_image"] = Image.fromarray(ref_u8)

        try:
            output = self.difix3d(
                self.prompt,
                image=pil_image,
                **call_kw,
            )
        except TypeError as e:
            if "ref_image" in call_kw:
                logger.warning(
                    "DiFix pipeline rejected ref_image (%s); retry without ref_image.", e
                )
                del call_kw["ref_image"]
                output = self.difix3d(
                    self.prompt,
                    image=pil_image,
                    **call_kw,
                )
            else:
                raise

        refined = output.images[0]
        refined = np.asarray(refined).astype(np.float32) / 255.0
        refined = np.clip(refined, 0.0, 1.0)
        return refined


def _load_difix_pipeline(args: argparse.Namespace):
    """
    Load DiFix3D pipeline from local project.
    Equivalent to the user's test.py usage.
    """
    if args.difix_src_dir not in sys.path:
        sys.path.append(args.difix_src_dir)
    from pipeline_difix import DifixPipeline

    load_kw = {}
    if getattr(args, "difix_trust_remote_code", False):
        load_kw["trust_remote_code"] = True
    pipe = DifixPipeline.from_pretrained(args.difix_pretrained_dir, **load_kw)
    pipe.to(args.difix_device)
    logger.info(f"Loaded DiFix pipeline from: {args.difix_pretrained_dir}")
    return pipe


def _frame_data_to_device(frame_data: dict, device: str = "cuda") -> dict:
    cam_infos = {}
    image_infos = {}
    for key, value in frame_data["cam_infos"].items():
        if isinstance(value, torch.Tensor):
            cam_infos[key] = value.to(device, non_blocking=True)
        else:
            cam_infos[key] = value
    for key, value in frame_data["image_infos"].items():
        if isinstance(value, torch.Tensor):
            image_infos[key] = value.to(device, non_blocking=True)
        else:
            image_infos[key] = value
    return {"cam_infos": cam_infos, "image_infos": image_infos}


def _frame_data_to_cpu(frame_data: dict) -> dict:
    cam_infos = {}
    image_infos = {}
    for key, value in frame_data["cam_infos"].items():
        if isinstance(value, torch.Tensor):
            cam_infos[key] = value.detach().cpu()
        else:
            cam_infos[key] = value
    for key, value in frame_data["image_infos"].items():
        if isinstance(value, torch.Tensor):
            image_infos[key] = value.detach().cpu()
        else:
            image_infos[key] = value
    return {"cam_infos": cam_infos, "image_infos": image_infos}


@torch.no_grad()
def _render_single_novel_frame(trainer: BasicTrainer, frame_data: dict) -> torch.Tensor:
    trainer.set_eval()
    outputs = trainer(
        image_infos=frame_data["image_infos"],
        camera_infos=frame_data["cam_infos"],
        novel_view=True,
        render_only=True,
    )
    return outputs[0]["rgb"].clamp(0.0, 1.0)


def _distill_one_step(
    trainer: BasicTrainer,
    frame_data_device: dict,
    target_rgb: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    trainer.set_train()
    optimizer.zero_grad(set_to_none=True)
    outputs = trainer(
        image_infos=frame_data_device["image_infos"],
        camera_infos=frame_data_device["cam_infos"],
        novel_view=True,
        freeze_road=False,
    )
    pred_rgb = outputs[0]["rgb"].clamp(0.0, 1.0)
    distill_loss = F.l1_loss(pred_rgb, target_rgb)
    distill_loss.backward()
    optimizer.step()
    return float(distill_loss.item())


def _build_distill_optimizer(
    trainer: BasicTrainer,
    lr: float,
    lr_scale: float = 1.0,
) -> torch.optim.Optimizer:
    """
    Build a robust optimizer for distillation.
    It avoids crashing on model-specific lazy attributes (e.g., SMPLNodes.instances_quats).
    """
    param_groups = []
    seen_param_ids = set()

    for class_name, model in trainer.models.items():
        collected = []
        if hasattr(model, "get_param_groups"):
            try:
                model_param_groups = model.get_param_groups()
                for _, params in model_param_groups.items():
                    for p in params:
                        if (
                            isinstance(p, torch.Tensor)
                            and p.requires_grad
                            and id(p) not in seen_param_ids
                        ):
                            seen_param_ids.add(id(p))
                            collected.append(p)
            except Exception as e:
                logger.warning(
                    f"Skip get_param_groups for {class_name} during distill: {e}"
                )

        # Fallback to module.parameters() if no valid params from get_param_groups
        if len(collected) == 0:
            for p in model.parameters():
                if p.requires_grad and id(p) not in seen_param_ids:
                    seen_param_ids.add(id(p))
                    collected.append(p)

        if len(collected) > 0:
            param_groups.append({"params": collected, "lr": lr * lr_scale})

    if len(param_groups) == 0:
        raise RuntimeError("No trainable parameters found for DiFix distillation.")

    total_params = sum(len(g["params"]) for g in param_groups)
    logger.info(f"Built distill optimizer with {total_params} tensors.")
    return torch.optim.Adam(param_groups, lr=lr * lr_scale)


def _safe_trainer_state_dict_only_model(trainer: BasicTrainer) -> dict:
    """
    Build a robust trainer state_dict for save, skipping broken/uninitialized models.
    This is needed when some optional models (e.g., SMPLNodes) are present in
    config but not loaded from checkpoint.
    """
    state_dict = {}
    base_state = torch.nn.Module.state_dict(trainer)
    state_dict.update(base_state)

    safe_models = {}
    for class_name, model in trainer.models.items():
        try:
            safe_models[class_name] = model.state_dict()
        except Exception as e:
            logger.warning(
                f"Skip model '{class_name}' in final-fix.ckpt due to state_dict error: {e}"
            )
    state_dict["models"] = safe_models
    state_dict["step"] = getattr(trainer, "step", 0)
    return state_dict


def _difix_distill_dir_suffix(traj_type: str) -> str:
    """
    Maps trajectory names to difix_distill_all_frames_<suffix> tags.
    e.g. left_shift_0.2m -> 0_2, left_shift_1m -> 1_0; legacy left_shift_0_2_m still works.
    """
    m = re.match(r"^left_shift_([\d.]+)m$", traj_type)
    if m:
        num = m.group(1)
        if "." in num:
            whole, frac = num.split(".", 1)
            return f"{whole}_{frac}"
        return f"{num}_0"
    m = re.match(r"^left_shift_(\d+)_(\d+)_m$", traj_type)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    m = re.match(r"^left_shift_(\d+)_m$", traj_type)
    if m:
        return f"{m.group(1)}_0"
    safe = re.sub(r"[^\w\-.]+", "_", traj_type)
    return safe.strip("_") or "traj"


def _traj_type_from_difix_dir_suffix(suffix: str) -> str:
    """Reverse map difix_*_all_frames_<suffix> back to trajectory name."""
    m = re.match(r"^(\d+)_(\d+)$", suffix)
    if m:
        whole, frac = m.group(1), m.group(2)
        if frac == "0":
            return f"left_shift_{whole}m"
        return f"left_shift_{whole}.{frac}m"
    return suffix


def _infer_novel_traj_types_from_experiment(
    exp_dir: str,
    stage_glob_prefix: str,
    ref_traj_type: str = "original_traj",
) -> List[str]:
    traj_types: List[str] = []
    for name in sorted(os.listdir(exp_dir)):
        if not name.startswith(stage_glob_prefix):
            continue
        stage_path = os.path.join(exp_dir, name)
        if not os.path.isdir(stage_path):
            continue
        suffix = name[len(stage_glob_prefix) :]
        traj_type = _traj_type_from_difix_dir_suffix(suffix)
        if traj_type == ref_traj_type:
            continue
        traj_types.append(traj_type)
    return traj_types


def _difix_eval_dir_suffix(traj_type: str) -> str:
    return _difix_distill_dir_suffix(traj_type)


_LATERAL_SHIFT_TRAJ_RE = re.compile(r"^(left|right)_shift_([\d.]+)m$")


def _lateral_shift_meters(traj_type: str) -> Optional[float]:
    match = _LATERAL_SHIFT_TRAJ_RE.match(traj_type)
    if match is None:
        return None
    return float(match.group(2))


def _select_distill_render_traj_types(
    traj_stages: List[str],
    ref_traj_type: str = "original_traj",
) -> List[str]:
    """
    Pick trajectories for full-timeline rendering after distillation:
    original trajectory plus novel trajectory(ies) with the largest lateral shift.
    """
    novel_trajs: List[str] = []
    seen = set()
    for traj_type in traj_stages:
        if traj_type == ref_traj_type or traj_type in seen:
            continue
        seen.add(traj_type)
        novel_trajs.append(traj_type)

    max_offset_trajs: List[str] = []
    measurable = [
        (traj_type, meters)
        for traj_type in novel_trajs
        if (meters := _lateral_shift_meters(traj_type)) is not None
    ]
    if measurable:
        max_meters = max(meters for _, meters in measurable)
        max_offset_trajs = [
            traj_type for traj_type, meters in measurable if meters == max_meters
        ]
    elif novel_trajs:
        max_offset_trajs = [novel_trajs[-1]]
        logger.warning(
            "Could not parse lateral shift from novel trajectories %s; "
            "falling back to last novel stage for rendering: %s",
            novel_trajs,
            max_offset_trajs[0],
        )

    render_traj_types: List[str] = []
    if ref_traj_type in traj_stages and ref_traj_type not in render_traj_types:
        render_traj_types.append(ref_traj_type)
    for traj_type in max_offset_trajs:
        if traj_type not in render_traj_types:
            render_traj_types.append(traj_type)
    return render_traj_types


def _load_ref_video_frames(video_path: str) -> List[np.ndarray]:
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"ref video not found: {video_path}")
    reader = imageio.get_reader(video_path)
    frames = []
    for frame in reader:
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise RuntimeError(f"Unexpected ref video frame shape {arr.shape} in {video_path}")
        frames.append(np.clip(arr[..., :3].astype(np.float32) / 255.0, 0.0, 1.0))
    reader.close()
    if len(frames) == 0:
        raise RuntimeError(f"No frames decoded from ref video: {video_path}")
    return frames


def _resolve_per_cam_ref_video_path(ref_dir: str, cam_id: int) -> str:
    exact_path = os.path.join(ref_dir, f"cam{cam_id}.mp4")
    if os.path.isfile(exact_path):
        return exact_path

    matches = sorted(glob.glob(os.path.join(ref_dir, f"cam{cam_id}_*.mp4")))
    if len(matches) > 0:
        return matches[0]

    raise FileNotFoundError(
        f"reference video for cam{cam_id} not found under {ref_dir}. "
        f"Expected cam{cam_id}.mp4 or cam{cam_id}_*.mp4"
    )


def _load_per_cam_ref_video_frames(
    ref_dir: str,
    cam_ids: List[int],
) -> Dict[int, List[np.ndarray]]:
    ref_frames_by_cam = {}
    for cam_id in cam_ids:
        video_path = _resolve_per_cam_ref_video_path(ref_dir, cam_id)
        ref_frames_by_cam[cam_id] = _load_ref_video_frames(video_path)
        logger.info(
            "Using GT ref video for cam%s: %s (%d frames)",
            cam_id,
            video_path,
            len(ref_frames_by_cam[cam_id]),
        )
    return ref_frames_by_cam


def _copy_mask_or_zero(src_path: str, dst_path: str, shape_hw: tuple) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.isfile(src_path):
        mask = Image.open(src_path).convert("L")
    else:
        mask = Image.fromarray(np.zeros(shape_hw, dtype=np.uint8))
    mask.save(dst_path)


def _export_difix_pseudo_gt_sample(
    dataset: DrivingDatasetNovelView,
    traj_type: str,
    cam_id: int,
    frame_idx: int,
    fixed_u8: np.ndarray,
    frame_data: dict,
) -> None:
    """Export a DiFix-refined frame in the mixed-novel-view dataset layout."""
    frame_t = dataset.start_timestep + int(frame_idx)
    traj_dir = os.path.join(dataset.data_path, "novel_views", traj_type)
    image_dir = os.path.join(traj_dir, "images")
    cam_pose_dir = os.path.join(traj_dir, "cam_pose", f"cam{cam_id}")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(cam_pose_dir, exist_ok=True)

    imageio.imwrite(
        os.path.join(
            image_dir,
            f"{frame_t:06d}_{cam_id}_{traj_type}.00_scale0.3.png",
        ),
        fixed_u8,
    )

    c2w = frame_data["cam_infos"]["camera_to_world"]
    if isinstance(c2w, torch.Tensor):
        c2w = c2w.detach().cpu().numpy()
    np.savetxt(os.path.join(cam_pose_dir, f"{frame_t:06d}.txt"), c2w)

    h, w = fixed_u8.shape[:2]
    src_mask_root = dataset.data_path
    mask_specs = [
        (
            os.path.join(src_mask_root, "sky_masks", f"{frame_t:06d}_{cam_id}.png"),
            os.path.join(traj_dir, "sky_masks", f"{frame_t:06d}_{cam_id}_{traj_type}.00_scale0.3.png"),
        ),
        (
            os.path.join(src_mask_root, "dynamic_masks", "all", f"{frame_t:06d}_{cam_id}.png"),
            os.path.join(traj_dir, "dynamic_masks", "all", f"{frame_t:06d}_{cam_id}_{traj_type}.png"),
        ),
        (
            os.path.join(src_mask_root, "dynamic_masks", "human", f"{frame_t:06d}_{cam_id}.png"),
            os.path.join(traj_dir, "dynamic_masks", "human", f"{frame_t:06d}_{cam_id}_{traj_type}.png"),
        ),
        (
            os.path.join(src_mask_root, "dynamic_masks", "vehicle", f"{frame_t:06d}_{cam_id}.png"),
            os.path.join(traj_dir, "dynamic_masks", "vehicle", f"{frame_t:06d}_{cam_id}_{traj_type}.png"),
        ),
    ]
    for src_path, dst_path in mask_specs:
        _copy_mask_or_zero(src_path, dst_path, (h, w))


def get_timeline_frame_count(data_cfg) -> int:
    """Estimate timeline frame count from config + on-disk pose files (no full dataset load)."""
    try:
        data_path = os.path.join(data_cfg.data_root, f"{int(data_cfg.scene_idx):03d}")
    except (TypeError, ValueError):
        data_path = os.path.join(data_cfg.data_root, str(data_cfg.scene_idx))

    if os.path.exists(os.path.join(data_path, "lidar_pose")):
        total_frames = len(os.listdir(os.path.join(data_path, "lidar_pose")))
    elif os.path.exists(os.path.join(data_path, "ego_pose")):
        total_frames = len(os.listdir(os.path.join(data_path, "ego_pose")))
    else:
        raise ValueError(f"Cannot count frames: no ego_pose/lidar_pose under {data_path}")

    start = int(data_cfg.start_timestep)
    end = int(data_cfg.end_timestep)
    if end == -1:
        end = total_frames - 1
    return end + 1 - start


def _select_distill_frame_indices(num_frames: int, args: argparse.Namespace) -> List[int]:
    """Build frame indices for distillation with optional stride subsampling."""
    if num_frames <= 0:
        return []
    if getattr(args, "distill_use_all_frames", False):
        indices = list(range(num_frames))
    else:
        indices = [int(args.distill_frame_idx)]

    stride = max(1, int(getattr(args, "distill_frame_stride", 1)))
    if stride > 1:
        indices = indices[::stride]

    max_frames = int(getattr(args, "distill_max_frames", -1))
    if max_frames > 0:
        indices = indices[:max_frames]
    return indices


def _estimate_progressive_distill_steps(
    args: argparse.Namespace,
    traj_stages: List[str],
    stage_repeats: List[int],
    num_timeline_frames: int,
) -> dict:
    """Estimate total optimization steps under progressive stage-repeat mode."""
    frame_indices = _select_distill_frame_indices(num_timeline_frames, args)
    samples_per_stage = len(args.distill_cam_ids) * len(frame_indices)
    per_stage_steps = [
        samples_per_stage * int(repeat) for repeat in stage_repeats
    ]
    return {
        "num_timeline_frames": num_timeline_frames,
        "num_distill_frames": len(frame_indices),
        "frame_stride": max(1, int(getattr(args, "distill_frame_stride", 1))),
        "samples_per_stage": samples_per_stage,
        "per_stage_steps": per_stage_steps,
        "total_steps": int(sum(per_stage_steps)),
    }


def run_difix_distill_one_traj(
    cfg: OmegaConf,
    trainer: BasicTrainer,
    dataset: DrivingDatasetNovelView,
    camera_data_dict: dict,
    args: argparse.Namespace,
    traj_type: str,
    save_dir: str,
    difix_refiner: Difix3DRefiner,
    optimizer: Optional[torch.optim.Optimizer],
    stage_repeat: Optional[int] = None,
) -> None:
    """
    单个 traj_type：DiFix 全量帧 + 蒸馏；结果写入 save_dir。
    """
    stage_t0 = time.time()
    target_cam_ids = [int(cam_id) for cam_id in args.distill_cam_ids]
    os.makedirs(save_dir, exist_ok=True)

    stage_mode = "novel"
    if getattr(args, "distill_ref_traj_as_original_stage", False):
        ref_traj_type = str(getattr(args, "difix_ref_traj_type", "original_traj"))
        if traj_type == ref_traj_type:
            stage_mode = "original"

    render_traj = None
    if stage_mode == "novel":
        render_traj = dataset.get_novel_render_traj(
            traj_type=traj_type,
            target_frames=dataset.frame_num,
            traj_path=args.traj_path,
        )
        if render_traj is None:
            logger.warning("Skip distill stage '%s': get_novel_render_traj returned None", traj_type)
            return

    use_original_ref = getattr(args, "difix_use_original_traj_ref", False)
    ref_video_path = str(getattr(args, "difix_ref_video_path", "") or "").strip()
    ref_video_dir = str(getattr(args, "difix_ref_video_dir", "") or "").strip()
    ref_video_frames = None
    ref_video_frames_by_cam = None
    if use_original_ref:
        if ref_video_dir:
            ref_video_frames_by_cam = _load_per_cam_ref_video_frames(
                ref_video_dir, target_cam_ids
            )
        elif ref_video_path:
            ref_video_frames = _load_ref_video_frames(ref_video_path)
            logger.info(
                "Using single GT ref video for all cams: %s (%d frames)",
                ref_video_path,
                len(ref_video_frames),
            )
        else:
            raise ValueError(
                "--difix_use_original_traj_ref is true, but neither "
                "--difix_ref_video_dir nor --difix_ref_video_path is set"
            )

    original_replay_samples = []
    original_replay_t0 = time.time()
    if getattr(args, "distill_mix_original", False) or stage_mode == "original":
        if ref_video_frames_by_cam is None and ref_video_frames is None:
            raise ValueError(
                "--distill_mix_original requires original GT refs. Set "
                "--difix_use_original_traj_ref with --difix_ref_video_dir or --difix_ref_video_path."
            )
        original_traj = dataset.get_novel_render_traj(
            traj_type=getattr(args, "difix_ref_traj_type", "original_traj"),
            target_frames=dataset.frame_num,
            traj_path=args.traj_path,
        )
        if original_traj is None:
            raise RuntimeError("Could not build original trajectory replay data.")

        for replay_cam_id in target_cam_ids:
            target_cam_data = camera_data_dict[replay_cam_id]
            original_render_data = dataset.prepare_novel_view_render_data(
                original_traj, target_cam_data, is_original_traj=True
            )
            replay_frame_indices = _select_distill_frame_indices(
                len(original_render_data), args
            )
            logger.info(
                "Original replay cam=%s: using %d/%d frames (stride=%s)",
                replay_cam_id,
                len(replay_frame_indices),
                len(original_render_data),
                max(1, int(getattr(args, "distill_frame_stride", 1))),
            )

            cam_ref_video_frames = (
                ref_video_frames_by_cam[replay_cam_id]
                if ref_video_frames_by_cam is not None
                else ref_video_frames
            )
            for idx in replay_frame_indices:
                if idx >= len(cam_ref_video_frames):
                    logger.warning(
                        "Skip original replay frame: cam=%s idx=%s out of %s",
                        replay_cam_id,
                        idx,
                        len(cam_ref_video_frames),
                    )
                    continue
                original_replay_samples.append({
                    "frame_data": _frame_data_to_cpu(original_render_data[idx]),
                    "target_rgb_u8": (np.clip(cam_ref_video_frames[idx], 0.0, 1.0) * 255).astype(np.uint8),
                    "frame_idx": idx,
                    "cam_id": replay_cam_id,
                    "sample_type": "original",
                })
            del original_render_data
            torch.cuda.empty_cache()
        logger.info(
            "Prepared %d original trajectory replay samples for mixed distillation.",
            len(original_replay_samples),
        )
    if original_replay_samples:
        _log_timing(
            "TIMING distill_prepare_original traj=%s samples=%d: %.1f min",
            traj_type,
            len(original_replay_samples),
            (time.time() - original_replay_t0) / 60.0,
        )

    logger.info(
        "Distill stage traj_type=%s mode=%s -> save under %s",
        traj_type,
        stage_mode,
        save_dir,
    )
    novel_samples = []

    difix_t0 = time.time()
    if stage_mode == "novel":
        for distill_cam_id in target_cam_ids:
            target_cam_data = camera_data_dict[distill_cam_id]
            render_data = dataset.prepare_novel_view_render_data(
                render_traj, target_cam_data, is_original_traj=False
            )

            frame_indices = _select_distill_frame_indices(len(render_data), args)

            logger.info(
                "Processing DiFix for Cam %s (traj=%s), using %s/%s frames (stride=%s)...",
                distill_cam_id,
                traj_type,
                len(frame_indices),
                len(render_data),
                max(1, int(getattr(args, "distill_frame_stride", 1))),
            )

            for idx in frame_indices:
                frame_data = render_data[idx]
                frame_data_device = _frame_data_to_device(frame_data, device=args.difix_device)

                with torch.no_grad():
                    before_rgb = _render_single_novel_frame(trainer, frame_data_device)

                before_np = before_rgb.detach().cpu().numpy()
                del before_rgb
                del frame_data_device
                torch.cuda.empty_cache()

                ref_np = None
                previous_ref_dir = str(
                    getattr(args, "difix_previous_stage_ref_dir", "") or ""
                ).strip()
                if previous_ref_dir:
                    previous_ref_path = os.path.join(
                        previous_ref_dir,
                        f"cam{distill_cam_id}_frame{idx:06d}_fixed.png",
                    )
                    if os.path.isfile(previous_ref_path):
                        ref_np = (
                            np.asarray(Image.open(previous_ref_path).convert("RGB"))
                            .astype(np.float32)
                            / 255.0
                        )

                cam_ref_video_frames = None
                if ref_video_frames_by_cam is not None:
                    cam_ref_video_frames = ref_video_frames_by_cam[distill_cam_id]
                elif ref_video_frames is not None:
                    cam_ref_video_frames = ref_video_frames

                if ref_np is None and cam_ref_video_frames is not None:
                    if idx >= len(cam_ref_video_frames):
                        logger.warning(
                            "ref video shorter than trajectory for traj=%s cam=%s: idx=%s out of %s, skip ref_image for this frame.",
                            traj_type,
                            distill_cam_id,
                            idx,
                            len(cam_ref_video_frames),
                        )
                    else:
                        ref_np = cam_ref_video_frames[idx]

                fixed_np = difix_refiner.refine(before_np, ref_image=ref_np)
                fixed_u8 = (np.clip(fixed_np, 0.0, 1.0) * 255).astype(np.uint8)
                del fixed_np
                torch.cuda.empty_cache()

                if getattr(args, "difix_export_pseudo_gt", False):
                    _export_difix_pseudo_gt_sample(
                        dataset=dataset,
                        traj_type=traj_type,
                        cam_id=distill_cam_id,
                        frame_idx=idx,
                        fixed_u8=fixed_u8,
                        frame_data=frame_data,
                    )

                if not getattr(args, "skip_difix_distill_steps", False):
                    novel_samples.append({
                        "frame_data": _frame_data_to_cpu(frame_data),
                        "target_rgb_u8": fixed_u8,
                        "frame_idx": idx,
                        "cam_id": distill_cam_id,
                        "sample_type": "novel",
                    })

                prefix = f"cam{distill_cam_id}_frame{idx:06d}"
                imageio.imwrite(os.path.join(save_dir, f"{prefix}_before.png"), (before_np * 255).astype(np.uint8))
                imageio.imwrite(os.path.join(save_dir, f"{prefix}_fixed.png"), fixed_u8)
                if ref_np is not None:
                    imageio.imwrite(os.path.join(save_dir, f"{prefix}_ref.png"), (ref_np * 255).astype(np.uint8))

            del render_data
            torch.cuda.empty_cache()
    else:
        logger.info("Stage '%s' uses original trajectory GT replay only.", traj_type)
    if stage_mode == "novel":
        _log_timing(
            "TIMING difix_pseudo_gt_and_debug_png traj=%s samples=%d: %.1f min",
            traj_type,
            len(novel_samples),
            (time.time() - difix_t0) / 60.0,
        )

    if getattr(args, "skip_difix_distill_steps", False):
        logger.info(
            "DiFix pseudo-GT exported for traj=%s; skip_difix_distill_steps=true, so no model distillation is run.",
            traj_type,
        )
        _log_timing(
            "TIMING distill_stage_total traj=%s mode=%s: %.1f min",
            traj_type,
            stage_mode,
            (time.time() - stage_t0) / 60.0,
        )
        return

    num_novel_samples = len(novel_samples)
    # Progressive original stages only fill original_replay_samples; novel_samples stays empty.
    # Checking novel_samples alone would incorrectly skip original-trajectory GT replay.
    if stage_mode == "original":
        if len(original_replay_samples) == 0:
            logger.warning(
                "No original replay samples for traj=%s, skip distillation steps.",
                traj_type,
            )
            _log_timing(
                "TIMING distill_stage_total traj=%s mode=%s: %.1f min",
                traj_type,
                stage_mode,
                (time.time() - stage_t0) / 60.0,
            )
            return
    elif num_novel_samples == 0:
        logger.warning("No cached samples for traj=%s, skip distillation steps.", traj_type)
        _log_timing(
            "TIMING distill_stage_total traj=%s mode=%s: %.1f min",
            traj_type,
            stage_mode,
            (time.time() - stage_t0) / 60.0,
        )
        return
    if optimizer is None:
        raise RuntimeError("optimizer is required when skip_difix_distill_steps is false")

    distill_t0 = time.time()
    if stage_repeat is not None:
        if stage_repeat <= 0:
            raise ValueError(f"Invalid stage_repeat={stage_repeat} for traj={traj_type}")
        stage_samples = original_replay_samples if stage_mode == "original" else novel_samples
        if len(stage_samples) == 0:
            logger.warning("No cached samples for traj=%s stage_mode=%s, skip distillation.", traj_type, stage_mode)
            return
        actual_steps = len(stage_samples) * stage_repeat
        logger.info(
            "Starting progressive distillation for traj=%s mode=%s: samples=%d repeat=%d total_steps=%d",
            traj_type,
            stage_mode,
            len(stage_samples),
            stage_repeat,
            actual_steps,
        )
        step = 0
        for repeat_idx in range(stage_repeat):
            order = np.arange(len(stage_samples))
            np.random.shuffle(order)
            for sample_idx in order:
                sample = stage_samples[int(sample_idx)]
                frame_data_device = _frame_data_to_device(sample["frame_data"], device=args.difix_device)
                target_rgb = torch.from_numpy(sample["target_rgb_u8"]).to(args.difix_device, dtype=torch.float32) / 255.0
                loss_val = _distill_one_step(trainer, frame_data_device, target_rgb, optimizer)
                del frame_data_device
                del target_rgb
                step += 1
                if step % 500 == 0 or step == actual_steps:
                    logger.info(
                        "[%s] [Step %s/%s] repeat=%s/%s type=%s Loss: %.6f",
                        traj_type,
                        step,
                        actual_steps,
                        repeat_idx + 1,
                        stage_repeat,
                        sample.get("sample_type", stage_mode),
                        loss_val,
                    )
    else:
        min_required_steps = num_novel_samples * 100
        actual_steps = max(args.distill_steps, min_required_steps)

        logger.info(
            "Starting mixed distillation for traj=%s: total_steps=%s (approx %.1f novel steps per frame), original_replay_ratio=%.2f",
            traj_type,
            actual_steps,
            actual_steps / max(num_novel_samples, 1),
            args.distill_original_sample_ratio if original_replay_samples else 0.0,
        )

        novel_order = np.arange(num_novel_samples)
        np.random.shuffle(novel_order)
        original_order = np.arange(len(original_replay_samples))
        if len(original_order) > 0:
            np.random.shuffle(original_order)

        for step in range(actual_steps):
            use_original = (
                len(original_replay_samples) > 0
                and np.random.rand() < args.distill_original_sample_ratio
            )
            if use_original:
                if step > 0 and step % len(original_replay_samples) == 0:
                    np.random.shuffle(original_order)
                sample = original_replay_samples[original_order[step % len(original_replay_samples)]]
            else:
                if step > 0 and step % num_novel_samples == 0:
                    np.random.shuffle(novel_order)
                sample = novel_samples[novel_order[step % num_novel_samples]]

            frame_data_device = _frame_data_to_device(sample["frame_data"], device=args.difix_device)
            target_rgb = torch.from_numpy(sample["target_rgb_u8"]).to(args.difix_device, dtype=torch.float32) / 255.0

            loss_val = _distill_one_step(trainer, frame_data_device, target_rgb, optimizer)
            del frame_data_device
            del target_rgb

            if step % 500 == 0 or step == actual_steps - 1:
                logger.info(
                    "[%s] [Step %s/%s] type=%s Loss: %.6f",
                    traj_type,
                    step,
                    actual_steps,
                    sample.get("sample_type", "novel"),
                    loss_val,
                )

    _log_timing(
        "TIMING l1_distill_optimization traj=%s mode=%s: %.1f min",
        traj_type,
        stage_mode,
        (time.time() - distill_t0) / 60.0,
    )

    after_t0 = time.time()
    for item in novel_samples:
        frame_data_device = _frame_data_to_device(item["frame_data"], device=args.difix_device)
        after_rgb = _render_single_novel_frame(trainer, frame_data_device)
        imageio.imwrite(
            os.path.join(
                save_dir,
                f"cam{item['cam_id']}_frame{item['frame_idx']:06d}_after_distill.png",
            ),
            (after_rgb.detach().cpu().numpy() * 255).astype(np.uint8),
        )
        del frame_data_device
        del after_rgb
        torch.cuda.empty_cache()
    _log_timing(
        "TIMING after_distill_debug_png traj=%s samples=%d: %.1f min",
        traj_type,
        len(novel_samples),
        (time.time() - after_t0) / 60.0,
    )
    _log_timing(
        "TIMING distill_stage_total traj=%s mode=%s: %.1f min",
        traj_type,
        stage_mode,
        (time.time() - stage_t0) / 60.0,
    )


def _load_difix_ref_frames_for_eval(
    args: argparse.Namespace,
    target_cam_ids: List[int],
) -> Tuple[Optional[List[np.ndarray]], Optional[Dict[int, List[np.ndarray]]]]:
    use_original_ref = getattr(args, "difix_use_original_traj_ref", False)
    ref_video_path = str(getattr(args, "difix_ref_video_path", "") or "").strip()
    ref_video_dir = str(getattr(args, "difix_ref_video_dir", "") or "").strip()
    ref_video_frames = None
    ref_video_frames_by_cam = None
    if use_original_ref:
        if ref_video_dir:
            ref_video_frames_by_cam = _load_per_cam_ref_video_frames(
                ref_video_dir, target_cam_ids
            )
        elif ref_video_path:
            ref_video_frames = _load_ref_video_frames(ref_video_path)
            logger.info(
                "Using single GT ref video for all cams: %s (%d frames)",
                ref_video_path,
                len(ref_video_frames),
            )
        else:
            raise ValueError(
                "--difix_use_original_traj_ref is true, but neither "
                "--difix_ref_video_dir nor --difix_ref_video_path is set"
            )
    return ref_video_frames, ref_video_frames_by_cam


@torch.no_grad()
def export_difix_eval_one_traj(
    trainer_before: Optional[BasicTrainer],
    trainer_after: Optional[BasicTrainer],
    dataset: DrivingDatasetNovelView,
    camera_data_dict: dict,
    args: argparse.Namespace,
    traj_type: str,
    save_dir: str,
    difix_refiner: Difix3DRefiner,
) -> None:
    """
    Export before/fixed/after_distill PNGs on ALL timeline frames for evaluation.
    Distillation may subsample frames; evaluation always uses every frame.
    """
    ref_traj_type = str(getattr(args, "difix_ref_traj_type", "original_traj"))
    if traj_type == ref_traj_type:
        logger.info("Skip full-frame eval export for original trajectory stage: %s", traj_type)
        return
    if trainer_before is None and trainer_after is None:
        raise ValueError("At least one of trainer_before or trainer_after must be provided.")

    target_cam_ids = [int(cam_id) for cam_id in args.distill_cam_ids]
    os.makedirs(save_dir, exist_ok=True)

    render_traj = dataset.get_novel_render_traj(
        traj_type=traj_type,
        target_frames=dataset.frame_num,
        traj_path=args.traj_path,
    )
    if render_traj is None:
        logger.warning("Skip eval export for '%s': get_novel_render_traj returned None", traj_type)
        return

    ref_video_frames, ref_video_frames_by_cam = _load_difix_ref_frames_for_eval(
        args, target_cam_ids
    )

    logger.info(
        "Full-frame DiFix eval export traj_type=%s -> %s (before=%s after=%s)",
        traj_type,
        save_dir,
        trainer_before is not None,
        trainer_after is not None,
    )

    for distill_cam_id in target_cam_ids:
        target_cam_data = camera_data_dict[distill_cam_id]
        render_data = dataset.prepare_novel_view_render_data(
            render_traj, target_cam_data, is_original_traj=False
        )
        frame_indices = list(range(len(render_data)))
        logger.info(
            "Eval export cam=%s traj=%s: rendering %d/%d frames",
            distill_cam_id,
            traj_type,
            len(frame_indices),
            len(render_data),
        )

        cam_ref_video_frames = None
        if ref_video_frames_by_cam is not None:
            cam_ref_video_frames = ref_video_frames_by_cam[distill_cam_id]
        elif ref_video_frames is not None:
            cam_ref_video_frames = ref_video_frames

        for idx in frame_indices:
            frame_data = render_data[idx]
            prefix = f"cam{distill_cam_id}_frame{idx:06d}"

            if trainer_before is not None:
                frame_data_device = _frame_data_to_device(frame_data, device=args.difix_device)
                before_rgb = _render_single_novel_frame(trainer_before, frame_data_device)
                before_np = before_rgb.detach().cpu().numpy()
                del before_rgb
                del frame_data_device
                torch.cuda.empty_cache()

                ref_np = None
                if cam_ref_video_frames is not None:
                    if idx >= len(cam_ref_video_frames):
                        logger.warning(
                            "ref video shorter than trajectory for traj=%s cam=%s: idx=%s out of %s",
                            traj_type,
                            distill_cam_id,
                            idx,
                            len(cam_ref_video_frames),
                        )
                    else:
                        ref_np = cam_ref_video_frames[idx]

                fixed_np = difix_refiner.refine(before_np, ref_image=ref_np)
                fixed_u8 = (np.clip(fixed_np, 0.0, 1.0) * 255).astype(np.uint8)
                del fixed_np
                torch.cuda.empty_cache()

                imageio.imwrite(
                    os.path.join(save_dir, f"{prefix}_before.png"),
                    (before_np * 255).astype(np.uint8),
                )
                imageio.imwrite(os.path.join(save_dir, f"{prefix}_fixed.png"), fixed_u8)
                if ref_np is not None:
                    imageio.imwrite(
                        os.path.join(save_dir, f"{prefix}_ref.png"),
                        (ref_np * 255).astype(np.uint8),
                    )

            if trainer_after is not None:
                frame_data_device = _frame_data_to_device(frame_data, device=args.difix_device)
                after_rgb = _render_single_novel_frame(trainer_after, frame_data_device)
                imageio.imwrite(
                    os.path.join(save_dir, f"{prefix}_after_distill.png"),
                    (after_rgb.detach().cpu().numpy() * 255).astype(np.uint8),
                )
                del frame_data_device
                del after_rgb
                torch.cuda.empty_cache()

        del render_data
        torch.cuda.empty_cache()


def run_difix_eval_all_frames(
    cfg: OmegaConf,
    trainer: BasicTrainer,
    dataset: DrivingDatasetNovelView,
    camera_data_dict: dict,
    args: argparse.Namespace,
    traj_types: List[str],
    ckpt_path_before: str,
    ckpt_path_after: str,
    eval_stage_prefix: str = "difix_eval_all_frames_",
) -> List[str]:
    """
    Render all timeline frames for novel-trajectory evaluation.
    Uses ckpt_path_before for before/fixed pseudo-GT and ckpt_path_after for after_distill.
    """
    if not ckpt_path_before:
        raise ValueError(
            "ckpt_path_before is required for full-frame eval (before + DiFix pseudo-GT)."
        )
    if not ckpt_path_after:
        raise ValueError("ckpt_path_after is required for full-frame eval (after_distill).")

    difix_model = _load_difix_pipeline(args)
    difix_refiner = Difix3DRefiner(
        model=difix_model,
        prompt=args.difix_prompt,
        num_inference_steps=args.difix_num_inference_steps,
        timesteps=args.difix_timesteps,
        guidance_scale=args.difix_guidance_scale,
    )

    save_dirs: List[str] = []
    for traj_type in traj_types:
        suffix = _difix_eval_dir_suffix(traj_type)
        save_dir = os.path.join(cfg.log_dir, f"{eval_stage_prefix}{suffix}")
        save_dirs.append(save_dir)

        trainer.resume_from_checkpoint(ckpt_path=ckpt_path_before, load_only_model=True)
        export_difix_eval_one_traj(
            trainer_before=trainer,
            trainer_after=None,
            dataset=dataset,
            camera_data_dict=camera_data_dict,
            args=args,
            traj_type=traj_type,
            save_dir=save_dir,
            difix_refiner=difix_refiner,
        )

        trainer.resume_from_checkpoint(ckpt_path=ckpt_path_after, load_only_model=True)
        export_difix_eval_one_traj(
            trainer_before=None,
            trainer_after=trainer,
            dataset=dataset,
            camera_data_dict=camera_data_dict,
            args=args,
            traj_type=traj_type,
            save_dir=save_dir,
            difix_refiner=difix_refiner,
        )

    return save_dirs


def _layout_qcraft_with_black_missing(imgs, cam_names):
    """Preserve the full QCraft layout when rendering only one camera group."""
    from datasets.qcraft.qcraft_config import ALL_CAM_SPECS

    images = list(imgs)
    names = list(cam_names)
    # Include every slot so layout_qcraft does not crop away an absent group.
    template = max(images, key=lambda img: img.shape[0] * img.shape[1])
    for spec in ALL_CAM_SPECS:
        if spec.name not in names:
            images.append(np.zeros_like(template))
            names.append(spec.name)
    return get_layout("qcraft")(images, names)


def _merge_novel_render_results(
    render_results_by_cam: Dict[int, dict],
    camera_data_dict: dict,
) -> dict:
    """Interleave per-camera novel-view renders for multi-cam layout video export."""
    cam_ids_sorted = sorted(render_results_by_cam.keys())
    num_frames = len(render_results_by_cam[cam_ids_sorted[0]]["rgbs"])
    merged = {"rgbs": [], "depths": [], "opacities": [], "cam_names": []}
    has_depths = all("depths" in render_results_by_cam[c] for c in cam_ids_sorted)
    has_opacities = all(
        "opacities" in render_results_by_cam[c]
        and len(render_results_by_cam[c]["opacities"]) > 0
        for c in cam_ids_sorted
    )

    for frame_idx in range(num_frames):
        for cam_id in cam_ids_sorted:
            rr = render_results_by_cam[cam_id]
            merged["rgbs"].append(rr["rgbs"][frame_idx])
            if has_depths:
                merged["depths"].append(rr["depths"][frame_idx])
            if has_opacities:
                merged["opacities"].append(rr["opacities"][frame_idx])
            merged["cam_names"].append(camera_data_dict[cam_id].cam_name)
    return merged


@torch.no_grad()
def render_trajectory(
    step: int = 0,
    cfg: OmegaConf = None,
    trainer: BasicTrainer = None,
    dataset: DrivingDatasetNovelView = None,
    cam_ids: Optional[List[int]] = None,
    downscales: Optional[List[int]] = None,
    traj_types: List[str] = None,
    fps: int = 10,
    render_rgb: bool = True,
    render_depth: bool = False,
    save_images: bool = True,
    generate_lidar_pc: bool = False,
    args: Optional[argparse.Namespace] = None,
):
    trainer.set_eval()

    logger.info("Rendering novel views...")

    render_keys = []
    if render_rgb:
        render_keys.append("rgbs")
    if render_depth:
        render_keys.append("depths")

    logger.info(f"Render keys: {render_keys}")

    camera_data_dict = dataset.load_specified_cameras(cam_ids, downscales)

    for traj_type in traj_types:
        logger.info(f"Trajectory type: {traj_type}")

        render_traj = dataset.get_novel_render_traj(
            traj_type=traj_type,
            target_frames=dataset.frame_num,
            traj_path=args.traj_path if args is not None else None,
        )
        if render_traj is None:
            continue

        output_suffix = ""
        if args is not None and getattr(args, "enable_difix_distill", False):
            raw_suffix = str(getattr(args, "difix_output_suffix", "") or "").strip()
            if raw_suffix:
                raw_suffix = raw_suffix.lstrip("-_")
                output_suffix = f"_{raw_suffix}"
        output_dir = f"{cfg.log_dir}/novel_traj/{traj_type}_step{step}{output_suffix}"
        os.makedirs(output_dir, exist_ok=True)

        # Render and save video
        video_output_dir = os.path.join(output_dir, "videos")
        os.makedirs(video_output_dir, exist_ok=True)
        video_output_path = os.path.join(video_output_dir, f"{traj_type}.mp4")

        image_output_dir = None
        if save_images:
            image_output_dir = os.path.join(output_dir, "images")
            os.makedirs(image_output_dir, exist_ok=True)

        save_layout = args is not None and (
            getattr(args, "save_layout_video", False)
            or getattr(args, "save_catted_videos", False)
        )
        layout_fn = get_layout(dataset.type) if save_layout else None
        if save_layout and dataset.type == "qcraft":
            layout_fn = _layout_qcraft_with_black_missing

        depths_per_cam = {}
        render_results_by_cam = {}
        is_original = traj_type == "original_traj"
        for cam_id, target_cam_data in camera_data_dict.items():
            render_data = dataset.prepare_novel_view_render_data(
                traj=render_traj,
                target_cam_data=target_cam_data,
                is_original_traj=is_original,
            )
            render_results = render_novel_views(
                trainer,
                render_data,
                target_cam_data,
            )
            del render_data

            save_single_camera_video(
                render_results,
                cam_id,
                dataset.start_timestep,
                dataset.end_timestep,
                video_output_path,
                image_output_dir,
                keys=render_keys,
                fps=fps,
                verbose=True,
            )

            depths_per_cam[cam_id] = render_results["depths"]
            if save_layout:
                render_results_by_cam[cam_id] = render_results
            else:
                del render_results

        if save_layout and layout_fn is not None:
            merged_results = _merge_novel_render_results(
                render_results_by_cam, camera_data_dict
            )
            num_frames = len(render_results_by_cam[sorted(render_results_by_cam.keys())[0]]["rgbs"])
            layout_video_path = os.path.join(
                video_output_dir, f"{traj_type}_rgbs_layout.mp4"
            )
            save_videos(
                merged_results,
                layout_video_path,
                layout=layout_fn,
                num_timestamps=num_frames,
                keys=render_keys,
                num_cams=len(render_results_by_cam),
                save_seperate_video=False,
                save_images=False,
                fps=fps,
                verbose=True,
            )
            logger.info(f"Saved layout video to {layout_video_path}")
            del merged_results, render_results_by_cam

        logger.info(f"Saved novel view videos for trajectory type: {traj_type}")

        # [DEPRECATED] 生成雷达点云
        if generate_lidar_pc:
            pc_output_dir = os.path.join(output_dir, "lidar_point_clouds")
            os.makedirs(pc_output_dir, exist_ok=True)

            pinhole_cam_ids = [
                id for id, cam in camera_data_dict.items() if not cam.is_fisheye
            ]
            intrinsics = {
                cam_id: camera_data_dict[cam_id].intrinsics.cpu().numpy()
                for cam_id in pinhole_cam_ids
            }
            T_cam_to_lidars = {
                cam_id: camera_data_dict[cam_id].cam_to_main_lidar
                for cam_id in pinhole_cam_ids
            }

            for frame_id, t in enumerate(range(dataset.start_timestep, dataset.end_timestep)):
                all_points = []
                for cam_id in pinhole_cam_ids:
                    points = unproject_depth_to_pointcloud(
                        depths_per_cam[cam_id][frame_id],
                        intrinsics[cam_id][frame_id],
                        T_cam_to_lidars[cam_id],
                    )
                    if len(points) > 0:
                        all_points.append(points)

                if len(all_points) > 0:
                    # 合并所有相机的点云
                    all_points = np.vstack(all_points)
                    all_points = remove_ground_points(all_points, ground_height=-3)

                    # 保存为PCD文件
                    pcd_path = os.path.join(pc_output_dir, f"frame_{t:06d}.pcd")
                    save_pointcloud_pcd(all_points, pcd_path)

                    logger.debug(
                        f"Frame {t}: Merged {len(all_points)} points -> {pcd_path}"
                    )

        del depths_per_cam


def main(args):
    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    args.enable_wandb = False

    global logger
    setup_logging(level=logging.INFO, time_string=current_time)

    camera_ids = (
        args.cam_ids if len(args.cam_ids) > 0 else cfg.data.pixel_source.cameras
    )
    downscales = (
        args.downscales
        if len(args.downscales) > 0
        else cfg.data.pixel_source.downscale_when_loading
    )
    logger.info(f"Camera IDs: {camera_ids}")
    logger.info(f"Downscales: {downscales}")
    assert len(camera_ids) == len(downscales)

    if args.save_catted_videos:
        cfg.logging.save_seperate_video = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # build dataset
    dataset = DrivingDatasetNovelView(data_cfg=cfg.data)

    # setup trainer
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        # num_train_images=len(dataset.train_indices),
        num_full_images=len(dataset.train_indices + dataset.test_indices),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )

    # Resume from checkpoint
    trainer.resume_from_checkpoint(
        ckpt_path=args.resume_from,
        load_only_model=True,
    )
    logger.info(
        f"Resuming training from {args.resume_from}, starting at step {trainer.step}"
    )

    if args.enable_viewer:
        # a simple viewer for background visualization
        trainer.init_viewer(port=args.viewer_port)

    if args.enable_difix_distill:
        camera_data_dict = dataset.load_specified_cameras(camera_ids, downscales)
        traj_stages = list(args.traj_types or [])
        if len(traj_stages) == 0:
            raise ValueError(
                "--traj_types must list at least one trajectory when --enable_difix_distill is set"
            )
        logger.info(
            "DiFix distillation trajectory stages (%d): %s",
            len(traj_stages),
            ", ".join(traj_stages),
        )
        stage_repeats = list(args.distill_stage_repeats or [])
        if len(stage_repeats) == 1 and len(traj_stages) > 1:
            stage_repeats = stage_repeats * len(traj_stages)
        if len(stage_repeats) not in (0, len(traj_stages)):
            raise ValueError(
                f"--distill_stage_repeats must be empty, single value, or match --traj_types count "
                f"({len(traj_stages)}), but got {len(stage_repeats)}"
            )
        if any(int(v) <= 0 for v in stage_repeats):
            raise ValueError("--distill_stage_repeats values must all be > 0")

        distill_plan = _estimate_progressive_distill_steps(
            args=args,
            traj_stages=traj_stages,
            stage_repeats=stage_repeats,
            num_timeline_frames=dataset.frame_num,
        )
        logger.info(
            "Distill frame plan: timeline_frames=%d distill_frames=%d stride=%d "
            "samples_per_stage=%d total_steps=%d (stages=%d repeats=%s)",
            distill_plan["num_timeline_frames"],
            distill_plan["num_distill_frames"],
            distill_plan["frame_stride"],
            distill_plan["samples_per_stage"],
            distill_plan["total_steps"],
            len(traj_stages),
            stage_repeats,
        )

        difix_model = _load_difix_pipeline(args)
        difix_refiner = Difix3DRefiner(
            model=difix_model,
            prompt=args.difix_prompt,
            num_inference_steps=args.difix_num_inference_steps,
            timesteps=args.difix_timesteps,
            guidance_scale=args.difix_guidance_scale,
        )
        optimizer = None
        if not args.skip_difix_distill_steps:
            optimizer = _build_distill_optimizer(
                trainer, lr=args.distill_lr, lr_scale=args.distill_lr_scale
            )

        ref_traj_type = str(getattr(args, "difix_ref_traj_type", "original_traj"))
        render_per_stage = not getattr(
            args, "distill_render_max_offset_and_original", False
        )
        deferred_render_traj_types: List[str] = []
        if not render_per_stage:
            deferred_render_traj_types = _select_distill_render_traj_types(
                traj_stages, ref_traj_type=ref_traj_type
            )
            logger.info(
                "distill_render_max_offset_and_original enabled: skip per-stage full "
                "timeline render; will render %s after all distill stages",
                ", ".join(deferred_render_traj_types),
            )

        previous_stage_ref_dir = ""
        for stage_idx, traj_type in enumerate(traj_stages):
            suffix = _difix_distill_dir_suffix(traj_type)
            save_dir = os.path.join(cfg.log_dir, f"difix_distill_all_frames_{suffix}")
            stage_repeat = int(stage_repeats[stage_idx]) if len(stage_repeats) > 0 else None
            args.difix_previous_stage_ref_dir = (
                previous_stage_ref_dir
                if getattr(args, "difix_chain_previous_stage_ref", False)
                else ""
            )
            run_difix_distill_one_traj(
                cfg=cfg,
                trainer=trainer,
                dataset=dataset,
                camera_data_dict=camera_data_dict,
                args=args,
                traj_type=traj_type,
                save_dir=save_dir,
                difix_refiner=difix_refiner,
                optimizer=optimizer,
                stage_repeat=stage_repeat,
            )
            if (
                getattr(args, "difix_chain_previous_stage_ref", False)
                and traj_type != ref_traj_type
            ):
                previous_stage_ref_dir = save_dir
            if not args.skip_difix_distill_steps and render_per_stage:
                stage_render_t0 = time.time()
                render_trajectory(
                    step=trainer.step,
                    cfg=cfg,
                    trainer=trainer,
                    dataset=dataset,
                    cam_ids=camera_ids,
                    downscales=downscales,
                    traj_types=[traj_type],
                    fps=args.fps,
                    render_rgb=args.render_rgb,
                    render_depth=args.render_depth,
                    save_images=args.save_images,
                    generate_lidar_pc=args.generate_lidar_pc,
                    args=args,
                )
                _log_timing(
                    "TIMING stage_render_trajectory traj=%s: %.1f min",
                    traj_type,
                    (time.time() - stage_render_t0) / 60.0,
                )

        if (
            not args.skip_difix_distill_steps
            and not render_per_stage
            and deferred_render_traj_types
        ):
            final_render_t0 = time.time()
            render_trajectory(
                step=trainer.step,
                cfg=cfg,
                trainer=trainer,
                dataset=dataset,
                cam_ids=camera_ids,
                downscales=downscales,
                traj_types=deferred_render_traj_types,
                fps=args.fps,
                render_rgb=args.render_rgb,
                render_depth=args.render_depth,
                save_images=args.save_images,
                generate_lidar_pc=args.generate_lidar_pc,
                args=args,
            )
            _log_timing(
                "TIMING final_render_trajectory traj_types=%s: %.1f min",
                ",".join(deferred_render_traj_types),
                (time.time() - final_render_t0) / 60.0,
            )

        if not args.skip_final_fix_ckpt and not args.skip_difix_distill_steps:
            ckpt_t0 = time.time()
            if os.path.isabs(args.final_fix_ckpt):
                final_path = args.final_fix_ckpt
            elif os.path.dirname(args.final_fix_ckpt):
                final_path = os.path.normpath(args.final_fix_ckpt)
            else:
                final_path = os.path.join(cfg.log_dir, args.final_fix_ckpt)
            os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
            torch.save(_safe_trainer_state_dict_only_model(trainer), final_path)
            logger.info(f"Distilled model saved to {final_path}")
            _log_timing(
                "TIMING save_final_checkpoint: %.1f min",
                (time.time() - ckpt_t0) / 60.0,
            )
    else:
        render_trajectory(
            step=trainer.step,
            cfg=cfg,
            trainer=trainer,
            dataset=dataset,
            cam_ids=camera_ids,
            downscales=downscales,
            traj_types=args.traj_types or [],
            fps=args.fps,
            render_rgb=args.render_rgb,
            render_depth=args.render_depth,
            save_images=args.save_images,
            generate_lidar_pc=args.generate_lidar_pc,
            args=args,
        )

    if args.enable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Render novel trajectory for a single scene")
    # eval
    parser.add_argument(
        "--resume_from",
        default=None,
        help="path to checkpoint to resume from",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--traj_types",
        default=None,
        nargs="+",
        type=str,
        help=(
            "Novel trajectory names in order. With --enable_difix_distill, the same list drives "
            "each distill stage and its per-traj novel render."
        ),
    )
    parser.add_argument(
        "--traj_path",
        default=None,
        type=str,
        help="directory of self-defined trajectories (cam2world)",
    )

    parser.add_argument(
        "--cam_ids",
        default=None,
        nargs="*",
        type=int,
        help="Camera ID to render",
        required=True,
    )
    parser.add_argument(
        "--downscales",
        default=None,
        nargs="*",
        type=float,
        help="Downscales for each camera ID",
        required=True,
    )

    parser.add_argument(
        "--fps",
        default=10,
        type=float,
        help="Frame per second of the rendered video",
    )
    parser.add_argument(
        "--save_catted_videos",
        action="store_true",
        help="save multi-camera layout video (same as --save_layout_video)",
    )
    parser.add_argument(
        "--save_layout_video",
        action="store_true",
        help="save dataset-specific multi-camera tiled layout video",
    )

    parser.add_argument(
        "--render_rgb", action="store_true", help="render rgb novel views"
    )
    parser.add_argument(
        "--render_depth", action="store_true", help="render depth novel views"
    )
    parser.add_argument(
        "--save_images", action="store_true", help="save rendered images"
    )
    parser.add_argument(
        "--generate_lidar_pc",
        action="store_true",
        help="generate point cloud for each frame",
    )
    parser.add_argument(
        "--enable_difix_distill",
        action="store_true",
        help="enable single-view DiFix3D fix and distill back to 3DGS",
    )
    parser.add_argument(
        "--distill_ref_cam_id",
        type=int,
        default=0,
        help="reference camera id for novel trajectory generation",
    )
    parser.add_argument(
        "--distill_cam_ids",
        nargs="+",
        type=int,
        default=[0],
        help="target camera ids for distill (single or multiple)",
    )
    parser.add_argument(
        "--distill_frame_idx",
        type=int,
        default=0,
        help="frame index on the chosen trajectory to distill",
    )
    parser.add_argument(
        "--distill_use_all_frames",
        action="store_true",
        help="use all frames for selected distill camera(s)",
    )
    parser.add_argument(
        "--distill_max_frames",
        type=int,
        default=-1,
        help="limit number of distilled frames per camera after stride; -1 means all",
    )
    parser.add_argument(
        "--distill_frame_stride",
        type=int,
        default=1,
        help=(
            "subsample timeline frames for distillation/DiFix, e.g. 3 keeps every 3rd frame "
            "(0,3,6,...). Applied after --distill_use_all_frames and before --distill_max_frames."
        ),
    )
    parser.add_argument(
        "--distill_steps",
        type=int,
        default=200,
        help="number of optimization steps for distillation",
    )
    parser.add_argument(
        "--distill_lr_scale",
        type=float,
        default=1.0,
        help="global LR scale multiplier during distillation",
    )
    parser.add_argument(
        "--distill_lr",
        type=float,
        default=1.0e-4,
        help="base learning rate for DiFix distillation",
    )
    parser.add_argument(
        "--distill_mix_original",
        action="store_true",
        help="mix original-trajectory GT replay with DiFix-refined novel trajectory samples during distillation",
    )
    parser.add_argument(
        "--distill_original_sample_ratio",
        type=float,
        default=0.65,
        help="probability of sampling original-trajectory GT replay when --distill_mix_original is enabled",
    )
    parser.add_argument(
        "--distill_stage_repeats",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional per-stage repeat count for deterministic progressive training. "
            "If set, each stage uses (num_samples_of_stage * repeat) optimization steps. "
            "Provide one value to broadcast to all stages, or one value per --traj_types entry."
        ),
    )
    parser.add_argument(
        "--distill_ref_traj_as_original_stage",
        action="store_true",
        help=(
            "When enabled, any stage whose traj_type == --difix_ref_traj_type is trained as "
            "original-trajectory GT replay only (no DiFix refinement)."
        ),
    )
    parser.add_argument(
        "--distill_render_max_offset_and_original",
        action="store_true",
        help=(
            "After all distill stages, render full timeline only for the original trajectory "
            "and the novel trajectory with the largest lateral shift (e.g. left_shift_3m). "
            "Skips per-stage full-timeline rendering for intermediate offsets."
        ),
    )
    parser.add_argument(
        "--final_fix_ckpt",
        type=str,
        default="final-fix.ckpt",
        help="output checkpoint path (relative to log_dir or absolute)",
    )
    parser.add_argument(
        "--skip_final_fix_ckpt",
        action="store_true",
        help="skip saving final-fix checkpoint and only render with in-memory distilled model",
    )
    parser.add_argument(
        "--difix_export_pseudo_gt",
        action="store_true",
        help="export DiFix-refined frames into data_root/scene/novel_views/<traj_type> for mixed training",
    )
    parser.add_argument(
        "--skip_difix_distill_steps",
        action="store_true",
        help="run DiFix refinement/export only and skip optimizing the loaded 3DGS checkpoint",
    )
    parser.add_argument(
        "--difix_src_dir",
        type=str,
        default="/nas_thoru/oldbak/ga/code/Difix3D-main/src",
        help="path to Difix3D src directory containing pipeline_difix.py",
    )
    parser.add_argument(
        "--difix_pretrained_dir",
        type=str,
        default="/nas_thoru/oldbak/ga/code/Difix3D-main/difix",
        help="path to Difix3D pretrained directory",
    )
    parser.add_argument(
        "--difix_device",
        type=str,
        default="cuda",
        help="device for Difix pipeline",
    )
    parser.add_argument(
        "--difix_prompt",
        type=str,
        default="remove degradation",
        help="prompt for Difix pipeline",
    )
    parser.add_argument(
        "--difix_num_inference_steps",
        type=int,
        default=1,
        help="Difix num_inference_steps",
    )
    parser.add_argument(
        "--difix_timesteps",
        nargs="+",
        type=int,
        default=[199],
        help="Difix timesteps list",
    )
    parser.add_argument(
        "--difix_guidance_scale",
        type=_safe_float_arg,
        default=0.0,
        help="Difix guidance scale",
    )
    parser.add_argument(
        "--difix_use_original_traj_ref",
        action="store_true",
        help=(
            "Pass ref_image to DiFix: each frame uses the same frame index rendered on "
            "--difix_ref_traj_type (default original_traj), while image= shifted novel render."
        ),
    )
    parser.add_argument(
        "--difix_chain_previous_stage_ref",
        action="store_true",
        help=(
            "Use the previous novel stage's same-camera, same-frame fixed PNG as "
            "DiFix ref_image; the first stage falls back to the original trajectory ref."
        ),
    )
    parser.add_argument(
        "--difix_ref_traj_type",
        type=str,
        default="original_traj",
        help="Trajectory name for ref_image when --difix_use_original_traj_ref is set",
    )
    parser.add_argument(
        "--difix_ref_video_dir",
        type=str,
        default="",
        help=(
            "Directory containing per-camera GT reference videos for DiFix, "
            "for example cam0.mp4 ... cam12.mp4. "
            "When set, this takes priority over --difix_ref_video_path."
        ),
    )
    parser.add_argument(
        "--difix_ref_video_path",
        type=str,
        default="",
        help=(
            "Optional GT video path used as ref_image source. "
            "If set, each frame index uses this video's corresponding frame and skips rendered original_traj reference."
        ),
    )
    parser.add_argument(
        "--difix_trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to DifixPipeline.from_pretrained (e.g. nvidia/difix_ref)",
    )
    parser.add_argument(
        "--difix_output_suffix",
        type=str,
        default="difix",
        help="output dir suffix text appended when enable_difix_distill is true",
    )

    # viewer
    parser.add_argument("--enable_viewer", action="store_true", help="enable viewer")
    parser.add_argument("--viewer_port", type=int, default=8080, help="viewer port")

    # misc
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    main(args)
