"""Brightness matching utilities for digital asset insertion.

When adding a PLY asset (3DGS Gaussian) to a scene, its intrinsic brightness
may not match the target scene. These functions measure the average brightness
of the scene background and the asset, then scale the asset's DC color to match.
"""

from typing import Dict, List, Optional, Tuple

import torch

from models.gaussians.basics import RGB2SH, SH2RGB

try:
    from gsplat.cuda_legacy._torch_impl import quat_to_rotmat
except ImportError:
    quat_to_rotmat = None

EXCLUDE_MASK_KEYS = ("sky_masks", "egocar_masks", "human_masks", "dynamic_masks")
MIN_VISIBILITY = 0.05
MIN_VISIBLE_POINTS = 128
TRIM_RATIO = 0.05
SCALE_CLIP = (0.5, 2.0)


def _rgb_luma(rgb: torch.Tensor) -> torch.Tensor:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _mask_to_bool(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    return mask.bool() if mask.dtype == torch.bool else mask > 0.5


def _trimmed_mean(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_tensor(float("nan"))
    values = torch.sort(values.flatten()).values
    trim = int(values.numel() * TRIM_RATIO)
    if trim > 0 and values.numel() > 2 * trim:
        values = values[trim:-trim]
    return values.mean()


def _background_luma(dataset) -> torch.Tensor:
    frame_count = len(dataset)
    if frame_count == 0:
        return torch.tensor(float("nan"))
    frame_ids = sorted(set([0, frame_count // 2, frame_count - 1]))
    lumas = []
    for frame_id in frame_ids:
        image_infos, _ = dataset.get_image(frame_id, 1.0)
        pixels = image_infos.get("pixels", None)
        if pixels is None:
            continue

        valid = torch.ones_like(pixels[..., 0], dtype=torch.bool)
        for key in EXCLUDE_MASK_KEYS:
            if key in image_infos:
                valid &= ~_mask_to_bool(image_infos[key])
        if valid.any():
            lumas.append(_rgb_luma(pixels.float().clamp(0.0, 1.0))[valid].mean())

    if len(lumas) == 0:
        return torch.tensor(float("nan"))
    return torch.stack(lumas).mean()


def _gaussian_rgb(features_dc: torch.Tensor, sh_degree: int) -> torch.Tensor:
    if sh_degree > 0:
        return SH2RGB(features_dc).clamp(0.0, 1.0)
    return torch.sigmoid(features_dc).clamp(0.0, 1.0)


def _set_gaussian_rgb(
    gaussian: Dict[str, torch.Tensor], rgb: torch.Tensor, sh_degree: int
) -> Dict[str, torch.Tensor]:
    gaussian = dict(gaussian)
    if sh_degree > 0:
        gaussian["features_dc"] = RGB2SH(rgb)
    else:
        gaussian["features_dc"] = torch.logit(rgb, eps=1e-6)
    return gaussian


def _gaussian_luma(gaussian: Dict[str, torch.Tensor], sh_degree: int) -> torch.Tensor:
    rgb = _gaussian_rgb(gaussian["features_dc"], sh_degree)
    luma = _rgb_luma(rgb)

    visibility = torch.sigmoid(gaussian["opacities"]).squeeze(-1)
    visible_luma = luma[visibility >= MIN_VISIBILITY]
    if visible_luma.numel() >= MIN_VISIBLE_POINTS:
        luma = visible_luma
    return _trimmed_mean(luma)


def compute_rigid_instances_luma(
    rigid_model,
    sh_degree: int,
    min_instance_points: int = 32,
    min_total_points: int = 128,
) -> Optional[torch.Tensor]:
    """Compute average luma across existing rigid instances.

    Returns None if there are too few instances or too few total Gaussians.
    """
    point_ids = getattr(rigid_model, "point_ids", None)
    if point_ids is None or point_ids.numel() == 0:
        return None

    unique_ids = torch.unique(point_ids)
    if unique_ids.numel() < 1:
        return None

    instance_lumas = []
    total_points = 0
    for ins_id in unique_ids.tolist():
        mask = (point_ids.squeeze(-1) == ins_id)
        n_pts = mask.sum().item()
        total_points += n_pts
        if n_pts < min_instance_points:
            continue
        gaussian = {
            "features_dc": rigid_model._features_dc[mask],
            "opacities": rigid_model._opacities[mask],
        }
        luma = _gaussian_luma(gaussian, sh_degree)
        if torch.isfinite(luma):
            instance_lumas.append(luma)

    if len(instance_lumas) < 1 or total_points < min_total_points:
        return None

    return torch.stack(instance_lumas).mean()


def _weighted_mean(pairs: List[Tuple[torch.Tensor, int]]) -> Optional[torch.Tensor]:
    """Compute point-count-weighted average of (luma, n_points) pairs."""
    if not pairs:
        return None
    lumas = torch.stack([p[0] for p in pairs])
    weights = torch.tensor([p[1] for p in pairs], dtype=lumas.dtype, device=lumas.device)
    return (lumas * weights).sum() / weights.sum()


def compute_nearby_rigid_luma(
    rigid_model,
    positions_frames: List[Tuple[torch.Tensor, int]],
    radius: float = 3.0,
    sh_degree: int = 3,
    min_total_points: int = 128,
) -> Optional[torch.Tensor]:
    """Query only RigidNodes Gaussians near sampled positions.

    Returns point-count-weighted average luma, or None if too few nearby points.
    """
    if not positions_frames:
        return None

    pairs = []  # (luma, n_points)
    for position_3d, frame_idx in positions_frames:
        position_3d = position_3d.to(torch.float32)
        _query_rigid_model(rigid_model, position_3d, frame_idx, radius, sh_degree, pairs)

    total_points = sum(p[1] for p in pairs)
    if total_points < min_total_points:
        return None

    return _weighted_mean(pairs)


def compute_local_luma(
    models: dict,
    positions_frames: List[Tuple[torch.Tensor, int]],
    radius: float = 3.0,
    sh_degree: int = 3,
    min_total_points: int = 128,
) -> Optional[torch.Tensor]:
    """Compute point-count-weighted average luma of nearby Gaussians.

    For each (position, frame) pair, queries all Gaussian models (Background,
    RigidNodes, SMPLNodes) for primitives within ``radius`` of the position.
    Each luma is weighted by the number of Gaussians that contributed to it.

    Returns None if fewer than ``min_total_points`` nearby Gaussians in total.
    """
    if not positions_frames:
        return None

    pairs = []  # (luma, n_points)
    for position_3d, frame_idx in positions_frames:
        position_3d = position_3d.to(torch.float32)

        # --- Background: _means are in world space ---
        bg_model = models.get("Background")
        if bg_model is not None and hasattr(bg_model, "_means"):
            _query_static_model(bg_model, position_3d, radius, sh_degree, pairs)

        # --- RigidNodes: _means in local space, transform per-instance ---
        rigid_model = models.get("RigidNodes")
        if rigid_model is not None and hasattr(rigid_model, "_means"):
            _query_rigid_model(rigid_model, position_3d, frame_idx, radius, sh_degree, pairs)

        # --- SMPLNodes: approximate using instances_trans as proxy ---
        smpl_model = models.get("SMPLNodes")
        if smpl_model is not None and hasattr(smpl_model, "_means"):
            _query_smpl_model(smpl_model, position_3d, frame_idx, radius, sh_degree, pairs)

    total_points = sum(p[1] for p in pairs)
    if total_points < min_total_points:
        return None

    return _weighted_mean(pairs)


def _query_static_model(
    model, position: torch.Tensor, radius: float, sh_degree: int,
    out_pairs: List[Tuple[torch.Tensor, int]],
):
    """Query a model whose _means are already in world space. Appends (luma, n_points)."""
    means = model._means.float()
    dists = torch.norm(means - position.unsqueeze(0), dim=-1)
    mask = dists < radius
    n_pts = int(mask.sum().item())
    if n_pts == 0:
        return
    gaussian = {
        "features_dc": model._features_dc[mask],
        "opacities": model._opacities[mask],
    }
    luma = _gaussian_luma(gaussian, sh_degree)
    if torch.isfinite(luma):
        out_pairs.append((luma, n_pts))


def _query_rigid_model(
    rigid_model, position: torch.Tensor, frame_idx: int, radius: float,
    sh_degree: int, out_pairs: List[Tuple[torch.Tensor, int]],
):
    """Query RigidNodes model by transforming local _means to world at frame_idx."""
    if quat_to_rotmat is None:
        return
    for ins_id in range(rigid_model.instances_fv.shape[1]):
        if not rigid_model.instances_fv[frame_idx, ins_id]:
            continue
        mask = rigid_model.point_ids.squeeze(-1) == ins_id
        n_pts = mask.sum().item()
        if n_pts == 0:
            continue
        local_xyz = rigid_model._means[mask].float()
        quat = rigid_model.instances_quats[frame_idx, ins_id].float()
        trans = rigid_model.instances_trans[frame_idx, ins_id].float()
        rotmat = quat_to_rotmat(quat.unsqueeze(0) / quat.norm()).squeeze(0)
        world_xyz = local_xyz @ rotmat.T + trans
        dists = torch.norm(world_xyz - position.unsqueeze(0), dim=-1)
        close_mask = dists < radius
        n_close = int(close_mask.sum().item())
        if n_close == 0:
            continue
        orig_mask = torch.nonzero(mask).squeeze(-1)
        selected_indices = orig_mask[close_mask]
        gaussian = {
            "features_dc": rigid_model._features_dc[selected_indices],
            "opacities": rigid_model._opacities[selected_indices],
        }
        luma = _gaussian_luma(gaussian, sh_degree)
        if torch.isfinite(luma):
            out_pairs.append((luma, n_close))


def _query_smpl_model(
    smpl_model, position: torch.Tensor, frame_idx: int, radius: float,
    sh_degree: int, out_pairs: List[Tuple[torch.Tensor, int]],
):
    """Query SMPLNodes model: use instances_trans as proxy, include instance if
    its root position is within 2*radius."""
    if not hasattr(smpl_model, "instances_fv"):
        return
    for ins_id in range(smpl_model.instances_fv.shape[1]):
        if not smpl_model.instances_fv[frame_idx, ins_id]:
            continue
        root_pos = smpl_model.instances_trans[frame_idx, ins_id].float()
        if torch.norm(root_pos - position).item() > radius * 2:
            continue
        mask = smpl_model.point_ids.squeeze(-1) == ins_id
        n_pts = mask.sum().item()
        if n_pts == 0:
            continue
        gaussian = {
            "features_dc": smpl_model._features_dc[mask],
            "opacities": smpl_model._opacities[mask],
        }
        luma = _gaussian_luma(gaussian, sh_degree)
        if torch.isfinite(luma):
            out_pairs.append((luma, n_pts))


def match_gaussian_brightness(
    gaussian: Dict[str, torch.Tensor],
    dataset,
    sh_degree: int,
    scale_clip: Tuple[float, float] = SCALE_CLIP,
    eps: float = 1e-6,
    reference_luma: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, torch.Tensor], Optional[float]]:
    if dataset is None and reference_luma is None:
        return gaussian, None

    if reference_luma is not None:
        bg_luma = reference_luma.to(gaussian["features_dc"].device)
    else:
        bg_luma = _background_luma(dataset).to(gaussian["features_dc"].device)
    asset_luma = _gaussian_luma(gaussian, sh_degree)
    if not torch.isfinite(bg_luma) or not torch.isfinite(asset_luma):
        return gaussian, None

    scale = (bg_luma / (asset_luma + eps)).clamp(*scale_clip)
    rgb = _gaussian_rgb(gaussian["features_dc"], sh_degree)
    return _set_gaussian_rgb(gaussian, (rgb * scale).clamp(0.0, 1.0), sh_degree), float(scale.item())
