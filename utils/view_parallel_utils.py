import logging
import os
from typing import List, Optional

import torch
from omegaconf import OmegaConf

logger = logging.getLogger()


DEFAULT_VIEW_GROUPS = {
    0: [0, 2, 5, 6, 7, 12],
    1: [1, 3, 9, 10, 11],
}


def parse_view_ids(view_ids: Optional[str]) -> Optional[List[int]]:
    if view_ids is None or view_ids == "":
        return None
    return [int(v.strip()) for v in view_ids.split(",") if v.strip()]


def _get_group_from_cfg(cfg: OmegaConf, group_id: int) -> List[int]:
    view_parallel = cfg.get("view_parallel", {})
    groups = view_parallel.get("groups", None)
    if groups is not None:
        key = f"group{group_id}"
        if key in groups:
            return [int(v) for v in groups[key]]
    if group_id in DEFAULT_VIEW_GROUPS:
        return DEFAULT_VIEW_GROUPS[group_id]
    raise ValueError(f"Cannot find view group {group_id}")


def _filter_downscale_when_loading(cfg: OmegaConf, original_cameras: List[int], view_ids: List[int]) -> None:
    pixel_cfg = cfg.data.pixel_source
    if "downscale_when_loading" not in pixel_cfg:
        return
    downscales = list(pixel_cfg.downscale_when_loading)
    if len(downscales) == len(view_ids):
        return
    if len(downscales) == len(original_cameras):
        downscale_map = {
            int(cam_id): downscales[idx] for idx, cam_id in enumerate(original_cameras)
        }
        pixel_cfg.downscale_when_loading = [downscale_map[int(cam_id)] for cam_id in view_ids]
        return
    if len(set(downscales)) == 1:
        pixel_cfg.downscale_when_loading = [downscales[0] for _ in view_ids]
        return
    raise ValueError(
        "data.pixel_source.downscale_when_loading length does not match original cameras "
        f"({len(original_cameras)}) or selected view ids ({len(view_ids)})."
    )


def apply_view_parallel_config(cfg: OmegaConf, args, append_group_to_log_dir: bool = True) -> List[int]:
    if not hasattr(args, "view_parallel"):
        return list(cfg.data.pixel_source.cameras)

    if args.view_parallel:
        if "view_parallel" not in cfg:
            cfg.view_parallel = {}
        cfg.view_parallel.enabled = True
        cfg.view_parallel.num_groups = int(getattr(args, "num_view_groups", 2))

    enabled = bool(cfg.get("view_parallel", {}).get("enabled", False))
    explicit_view_ids = parse_view_ids(getattr(args, "view_ids", None))
    group_id = getattr(args, "view_group_id", None)
    if group_id is None:
        group_id = cfg.get("view_parallel", {}).get("group_id", None)

    if not enabled and explicit_view_ids is None:
        return list(cfg.data.pixel_source.cameras)

    if explicit_view_ids is not None:
        view_ids = explicit_view_ids
    else:
        if group_id is None:
            raise ValueError("view_parallel is enabled but view_group_id is not specified.")
        view_ids = _get_group_from_cfg(cfg, int(group_id))

    original_cameras = [int(v) for v in cfg.data.pixel_source.cameras]
    _filter_downscale_when_loading(cfg, original_cameras, view_ids)
    cfg.data.pixel_source.cameras = view_ids

    if "view_parallel" not in cfg:
        cfg.view_parallel = {}
    cfg.view_parallel.enabled = enabled or explicit_view_ids is not None
    cfg.view_parallel.group_id = None if group_id is None else int(group_id)
    cfg.view_parallel.view_ids = view_ids
    if getattr(args, "gpu_id", None) is not None:
        cfg.view_parallel.gpu_id = int(args.gpu_id)

    if append_group_to_log_dir and cfg.view_parallel.enabled:
        group_name = f"group{cfg.view_parallel.group_id}"
        if not os.path.basename(str(cfg.log_dir).rstrip(os.sep)) == group_name:
            cfg.log_dir = os.path.join(cfg.log_dir, group_name)

    return view_ids


def log_view_parallel_status(cfg: OmegaConf, dataset=None) -> None:
    view_cfg = cfg.get("view_parallel", {})
    enabled = bool(view_cfg.get("enabled", False))
    gpu_id = view_cfg.get("gpu_id", None)
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    current_device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    logger.info(f"View parallel enabled: {enabled}")
    logger.info(f"Configured GPU id: {gpu_id}; CUDA_VISIBLE_DEVICES={visible_devices}; current device={current_device}")
    if enabled:
        logger.info(f"View group id: {view_cfg.get('group_id', None)}")
        logger.info(f"View ids: {list(cfg.data.pixel_source.cameras)}")
    logger.info(f"Output directory: {cfg.log_dir}")
    if dataset is not None:
        logger.info(f"Train image count: {len(dataset.train_image_set)}")
        logger.info(f"Full image count: {len(dataset.full_image_set)}")
        if dataset.test_image_set is not None:
            logger.info(f"Test image count: {len(dataset.test_image_set)}")
