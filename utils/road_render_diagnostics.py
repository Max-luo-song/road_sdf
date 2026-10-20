"""Layer ablations and numerical diagnostics; no CUDA imports at module load."""

from dataclasses import replace

import numpy as np


def render_layers(trainer, gs, cam, full):
    """Reuse activated Gaussians and the exact camera from one normal forward pass.

    One-hot RGB preserves all opacities/depth ordering: its channels measure
    visible road/background/other alpha contributions in the complete scene.
    Isolated renders instead remove occluders and must not be added together.
    """
    import torch

    labels = trainer.pts_labels
    road = labels == trainer.gaussian_classes["RoadNodes"]
    background = labels == trainer.gaussian_classes["Background"]
    options = dict(
        near_plane=trainer.render_cfg.near_plane,
        far_plane=trainer.render_cfg.far_plane,
        render_mode="RGB+ED",
        radius_clip=trainer.render_cfg.get("radius_clip", 0.0),
    )

    def render(selection=None, colors=None):
        subset = gs
        if selection is not None:
            if not selection.any():
                return {key: torch.zeros_like(full[key])
                        for key in ("rgb_gaussians", "depth", "opacity")}
            subset = replace(gs, **{
                key: getattr(gs, key)[selection]
                for key in ("_means", "_quats", "_scales", "_opacities", "_rgbs")
            })
        if colors is not None:
            subset = replace(subset, _rgbs=colors)
        result, _ = trainer.render_gaussians(gs=subset, cam=cam, **options)
        return result

    layers = {"full": {key: full[key] for key in ("rgb_gaussians", "depth", "opacity")}}
    for name, selection in (("road", road), ("background", background),
                            ("no_background", ~background)):
        layers[name] = render(selection)
    colors = torch.stack((road, background, ~(road | background)), -1).to(gs.rgbs)
    contribution = render(colors=colors)["rgb_gaussians"]
    counts = {"road": int(road.sum()), "background": int(background.sum()),
              "other": int((~(road | background)).sum())}
    return layers, contribution, counts


def normalize_rgb(rgb, alpha, minimum_alpha=0.05):
    """Undo alpha premultiplication only where support is sufficient."""
    alpha = np.asarray(alpha).reshape(rgb.shape[:2])
    valid = alpha >= minimum_alpha
    result = np.zeros_like(rgb)
    result[valid] = rgb[valid] / alpha[valid, None]
    return np.clip(result, 0, 1), valid


def depth_edges(depth, valid):
    """Largest adjacent relative ED jump; invalid neighbors never create edges."""
    depth = np.asarray(depth).reshape(valid.shape)
    result = np.zeros_like(depth)
    for axis in (0, 1):
        a = [slice(None), slice(None)]
        b = [slice(None), slice(None)]
        a[axis], b[axis] = slice(1, None), slice(None, -1)
        a, b = tuple(a), tuple(b)
        good = valid[a] & valid[b]
        jump = np.abs(depth[a] - depth[b]) / np.maximum(
            np.minimum(np.abs(depth[a]), np.abs(depth[b])), 1e-6)
        jump = np.where(good, jump, 0)
        result[a] = np.maximum(result[a], jump)
        result[b] = np.maximum(result[b], jump)
    return result


def expected_depth_normals(depth, valid, intrinsics):
    """Camera-space normals of expected-depth image, not Gaussian/surface normals.

    Central differences require four valid neighbors; image borders are invalid.
    Callers should also reject depth discontinuities before using these normals.
    """
    height, width = depth.shape
    y, x = np.mgrid[:height, :width]
    pixels = np.stack((x + .5, y + .5, np.ones_like(x)), -1)
    points = (pixels @ np.linalg.inv(intrinsics).T) * depth[..., None]
    normals = np.zeros((height, width, 3), dtype=np.float32)
    mask = np.zeros_like(valid)
    cross = np.cross(points[1:-1, 2:] - points[1:-1, :-2],
                     points[2:, 1:-1] - points[:-2, 1:-1])
    length = np.linalg.norm(cross, axis=-1)
    good = (valid[1:-1, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2]
            & valid[2:, 1:-1] & valid[:-2, 1:-1] & np.isfinite(length) & (length > 1e-8))
    cross /= np.maximum(length[..., None], 1e-8)
    # Orient toward camera for consistent display.
    cross *= np.where((cross * points[1:-1, 1:-1]).sum(-1) > 0, -1, 1)[..., None]
    normals[1:-1, 1:-1] = np.where(good[..., None], cross, 0)
    mask[1:-1, 1:-1] = good
    return normals, mask


def roi_mask(shape, box):
    """Normalized half-open [x0,y0,x1,y1] rectangle, independent of resolution."""
    if len(box) != 4 or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
        raise ValueError(f"Invalid normalized ROI: {box}")
    height, width = shape
    x0, y0 = np.floor(np.array(box[:2]) * [width, height]).astype(int)
    x1, y1 = np.ceil(np.array(box[2:]) * [width, height]).astype(int)
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def summarize(layers, contribution, composed, no_background_composed, region, thresholds):
    """Descriptive associations, not automatic artifact/causality classification."""
    alpha = layers["road"]["opacity"].squeeze(-1)
    road_rgb, valid = normalize_rgb(layers["road"]["rgb_gaussians"], alpha,
                                    thresholds["support_alpha"])
    full_alpha = layers["full"]["opacity"].squeeze(-1)
    road_visible = contribution[..., 0] >= thresholds["support_alpha"]
    other_visible = contribution[..., 2] >= thresholds["other_alpha"]
    support = region & valid & ~other_visible
    visible = support & road_visible
    dark = visible & (composed @ np.array([0.2126, 0.7152, 0.0722]) < thresholds["dark_luma"])
    weak = alpha < thresholds["coverage_alpha"]
    bg = contribution[..., 1] >= thresholds["background_alpha"]
    depth = layers["road"]["depth"].squeeze(-1)
    depth_valid = valid & np.isfinite(depth) & (depth > 0)
    edges = depth_edges(depth, depth_valid) > thresholds["relative_depth_jump"]
    delta = np.abs(composed - no_background_composed).mean(-1)

    def fraction(flag, mask):
        return float(flag[mask].mean()) if mask.any() else None

    def mean(value, mask):
        return float(value[mask].mean()) if mask.any() else None

    return {
        "roi_pixels": int(region.sum()), "supported_road_pixels": int(support.sum()),
        "visible_road_pixels": int(visible.sum()), "dark_visible_road_pixels": int(dark.sum()),
        "supported_road_weak_coverage_fraction": fraction(weak, support),
        "visible_road_background_contribution_fraction": fraction(bg, visible),
        "visible_road_depth_jump_fraction": fraction(edges, visible),
        "visible_road_mean_alpha": mean(alpha, visible),
        "visible_road_normalized_luma": mean(road_rgb @ np.array([0.2126, 0.7152, 0.0722]), visible),
        "visible_road_background_removal_rgb_mae": mean(delta, visible),
        "dark_road_weak_coverage_fraction": fraction(weak, dark),
        "dark_road_background_contribution_fraction": fraction(bg, dark),
        "dark_road_depth_jump_fraction": fraction(edges, dark),
        "dark_road_background_removal_rgb_mae": mean(delta, dark),
        "contribution_alpha_max_abs_error": float(np.abs(contribution.sum(-1) - full_alpha).max()),
    }
