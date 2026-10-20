"""Export RoadNodes as separate road and lane-line Gaussian PLY files."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


GAUSSIAN_KEYS = (
    "_means",
    "_features_dc",
    "_features_rest",
    "_opacities",
    "_scales",
    "_quats",
)


def load_road_gaussians(checkpoint_path, allow_unsafe=False):
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except pickle.UnpicklingError as exc:
        if not allow_unsafe:
            raise RuntimeError(
                f"Restricted loading failed for {checkpoint_path}. "
                "If this is a trusted "
                "checkpoint produced by this project, rerun with "
                "--allow-unsafe-checkpoint."
            ) from exc
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    try:
        state = checkpoint["models"]["RoadNodes"]
    except KeyError as exc:
        raise KeyError(f"RoadNodes not found in checkpoint: {checkpoint_path}") from exc
    missing = [key for key in GAUSSIAN_KEYS if key not in state]
    if missing:
        raise KeyError(
            f"Missing RoadNodes fields in {checkpoint_path}: {', '.join(missing)}"
        )
    gaussians = {
        key: state[key].detach().cpu().numpy() for key in GAUSSIAN_KEYS
    }
    saved_mask = state.get("_lane_mask")
    lane_mask = (
        None
        if saved_mask is None
        else saved_mask.detach().cpu().numpy().astype(bool).reshape(-1)
    )
    del checkpoint
    return gaussians, lane_mask


def find_checkpoints(input_path, checkpoint_name):
    if input_path.is_file():
        if input_path.suffix != ".pth":
            raise ValueError(f"Expected a .pth checkpoint, got: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    direct = input_path / checkpoint_name
    if direct.is_file():
        return [direct]
    checkpoints = sorted(input_path.glob(f"group*/{checkpoint_name}"))
    if not checkpoints:
        checkpoints = sorted(input_path.rglob(checkpoint_name))
    if not checkpoints:
        raise FileNotFoundError(
            f"Could not find {checkpoint_name} under directory: {input_path}"
        )
    return checkpoints


def load_config(config_path):
    if not config_path.is_file():
        raise FileNotFoundError(
            "Config required to reconstruct the lane mask was not found: "
            f"{config_path}"
        )
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML config: {config_path}")
    return config


def resolve_data_dir(config, override):
    if override is not None:
        return override
    try:
        data_cfg = config["data"]
        return Path(data_cfg["data_root"]) / str(data_cfg["scene_idx"])
    except KeyError as exc:
        raise KeyError(
            "Config must contain data.data_root and data.scene_idx, or use --data-dir"
        ) from exc


def resolve_label_dir(data_dir, override):
    if override is not None:
        label_dir = override
    else:
        root = data_dir / "label" / "bev_road_geometry_label"
        candidates = sorted(root.glob("bevrg_label_*")) if root.is_dir() else []
        if not candidates:
            raise FileNotFoundError(f"No bevrg_label_* directory found under: {root}")
        label_dir = candidates[-1]
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Lane label directory not found: {label_dir}")
    return label_dir


def load_lane_samples(data_dir, label_dir, config, spacing, stride):
    paths = {
        int(path.stem): path
        for path in label_dir.glob("*.json")
        if path.stem.isdigit()
    }
    if not paths:
        raise ValueError(f"No numeric BEVRG label files found: {label_dir}")
    data_cfg = config.get("data", {})
    start = int(data_cfg.get("start_timestep", min(paths)))
    configured_end = int(data_cfg.get("end_timestep", -1))
    end = max(paths) + 1 if configured_end == -1 else configured_end + 1
    frame_ids = [frame_id for frame_id in sorted(paths) if start <= frame_id < end]
    if not frame_ids:
        raise ValueError(f"No BEVRG labels overlap configured frames [{start}, {end})")
    selected_ids = frame_ids[::max(1, stride)]
    if selected_ids[-1] != frame_ids[-1]:
        selected_ids.append(frame_ids[-1])

    origin_path = data_dir / "lidar_pose" / f"{start:06d}.txt"
    if not origin_path.is_file():
        raise FileNotFoundError(f"Training-origin LiDAR pose not found: {origin_path}")
    origin = np.linalg.inv(np.loadtxt(origin_path))

    lines = {}
    for frame_id in selected_ids:
        ego_pose_path = data_dir / "ego_pose" / f"{frame_id:06d}.txt"
        if not ego_pose_path.is_file():
            continue
        ego_to_scene = origin @ np.loadtxt(ego_pose_path)
        with paths[frame_id].open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for index, item in enumerate(payload.get("mapformer_lane_info_v2", [])):
            if (
                item.get("coarse_lane_type") != "LANE_LINE"
                or item.get("is_filtered", False)
            ):
                continue
            points = np.asarray(
                [
                    [point.get("x"), point.get("y"), point.get("z")]
                    for point in item.get("points", [])
                    if point.get("is_ground_point", True)
                ],
                dtype=np.float64,
            )
            if points.ndim != 2 or len(points) < 2:
                continue
            points = points[np.isfinite(points).all(axis=1)]
            if len(points) < 2:
                continue
            points = (ego_to_scene @ np.c_[points, np.ones(len(points))].T).T[:, :3]
            line_id = str(item.get("fsd_id", f"{frame_id}:{index}"))
            if line_id not in lines or len(points) > len(lines[line_id]):
                lines[line_id] = points

    samples = []
    for points in lines.values():
        for point0, point1 in zip(points[:-1], points[1:]):
            delta = point1 - point0
            length = np.linalg.norm(delta[:2])
            if not np.isfinite(length) or not 1e-4 < length <= 5.0:
                continue
            count = max(1, int(np.ceil(length / spacing)))
            samples.append(point0 + np.arange(count)[:, None] / count * delta)
    if not samples:
        raise ValueError(f"No valid LANE_LINE geometry found: {label_dir}")
    return np.concatenate(samples).astype(np.float32, copy=False)


def reconstruct_lane_mask(means, lane_samples, radius, height_tolerance):
    tree = cKDTree(lane_samples[:, :2])
    distances, indices = tree.query(means[:, :2], k=1, workers=-1)
    height_error = np.abs(means[:, 2] - lane_samples[indices, 2])
    return (distances <= radius) & (height_error <= height_tolerance)


def flatten_rows(values):
    columns = int(np.prod(values.shape[1:]))
    return values.reshape(len(values), columns)


def write_ply(gaussians, mask, output_path):
    rows = int(mask.sum())
    means = gaussians["_means"][mask]
    features_dc = flatten_rows(gaussians["_features_dc"][mask])
    features_rest = flatten_rows(gaussians["_features_rest"][mask])
    opacities = flatten_rows(gaussians["_opacities"][mask])
    scales = flatten_rows(gaussians["_scales"][mask])
    quats = flatten_rows(gaussians["_quats"][mask])

    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{index}" for index in range(features_dc.shape[1])]
    names += [f"f_rest_{index}" for index in range(features_rest.shape[1])]
    names += ["opacity"]
    names += [f"scale_{index}" for index in range(scales.shape[1])]
    names += [f"rot_{index}" for index in range(quats.shape[1])]

    elements = np.empty(rows, dtype=[(name, "f4") for name in names])
    columns = [means[:, 0], means[:, 1], means[:, 2]]
    columns += [np.zeros(rows, dtype=np.float32) for _ in range(3)]
    columns += [features_dc[:, index] for index in range(features_dc.shape[1])]
    columns += [
        features_rest[:, index] for index in range(features_rest.shape[1])
    ]
    columns += [opacities[:, 0]]
    columns += [scales[:, index] for index in range(scales.shape[1])]
    columns += [quats[:, index] for index in range(quats.shape[1])]
    for name, column in zip(names, columns):
        elements[name] = column

    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(output_path)


def export_checkpoint(
    checkpoint_path,
    output_dir,
    config_path=None,
    data_dir=None,
    label_dir=None,
    lane_radius=None,
    lane_height_tolerance=None,
    lane_spacing=0.05,
    lane_stride=10,
    allow_unsafe_checkpoint=False,
):
    gaussians, lane_mask = load_road_gaussians(
        checkpoint_path, allow_unsafe=allow_unsafe_checkpoint
    )
    num_gaussians = len(gaussians["_means"])
    if lane_mask is None:
        config = load_config(config_path or checkpoint_path.parent / "config.yaml")
        road_ctrl = config.get("model", {}).get("RoadNodes", {}).get("ctrl", {})
        radius = float(
            lane_radius
            if lane_radius is not None
            else road_ctrl.get("lane_influence_radius", 0.2)
        )
        height_tolerance = float(
            lane_height_tolerance
            if lane_height_tolerance is not None
            else road_ctrl.get("lane_height_tolerance", 0.3)
        )
        resolved_data_dir = resolve_data_dir(config, data_dir)
        resolved_label_dir = resolve_label_dir(resolved_data_dir, label_dir)
        lane_samples = load_lane_samples(
            resolved_data_dir, resolved_label_dir, config, lane_spacing, lane_stride
        )
        lane_mask = reconstruct_lane_mask(
            gaussians["_means"], lane_samples, radius, height_tolerance
        )
    elif len(lane_mask) != num_gaussians:
        raise ValueError(
            f"Saved lane mask has {len(lane_mask)} entries, expected {num_gaussians}"
        )

    road_mask = ~lane_mask
    road_path = output_dir / "road_gaussians.ply"
    lane_path = output_dir / "lane_gaussians.ply"
    write_ply(gaussians, road_mask, road_path)
    write_ply(gaussians, lane_mask, lane_path)
    return road_path, lane_path, int(road_mask.sum()), int(lane_mask.sum())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export RoadNodes as separate road and lane-line PLY files."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", type=Path, help="Checkpoint, group directory, or run directory"
    )
    input_group.add_argument("--model", type=Path, help="Deprecated alias for --input")
    parser.add_argument("--checkpoint-name", default="checkpoint_final.pth")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output root; defaults to point_clouds next to each checkpoint",
    )
    parser.add_argument(
        "--config", type=Path, help="Override config.yaml for a single checkpoint"
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--lane-label-dir", type=Path)
    parser.add_argument("--lane-radius", type=float)
    parser.add_argument("--lane-height-tolerance", type=float)
    parser.add_argument("--lane-spacing", type=float, default=0.05)
    parser.add_argument("--lane-stride", type=int, default=10)
    parser.add_argument(
        "--allow-unsafe-checkpoint",
        action="store_true",
        help="Allow full pickle loading for trusted project checkpoints",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoints = find_checkpoints(
        args.input or args.model, args.checkpoint_name
    )
    if args.config is not None and len(checkpoints) != 1:
        raise ValueError("--config requires exactly one selected checkpoint")

    for checkpoint_path in checkpoints:
        if args.output_dir is None:
            output_dir = checkpoint_path.parent / "point_clouds"
        elif len(checkpoints) == 1:
            output_dir = args.output_dir
        else:
            output_dir = args.output_dir / checkpoint_path.parent.name
        road_path, lane_path, road_count, lane_count = export_checkpoint(
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            config_path=args.config,
            data_dir=args.data_dir,
            label_dir=args.lane_label_dir,
            lane_radius=args.lane_radius,
            lane_height_tolerance=args.lane_height_tolerance,
            lane_spacing=args.lane_spacing,
            lane_stride=args.lane_stride,
            allow_unsafe_checkpoint=args.allow_unsafe_checkpoint,
        )
        print(f"[{checkpoint_path}] road: {road_count} -> {road_path}")
        print(f"[{checkpoint_path}] lane: {lane_count} -> {lane_path}")


if __name__ == "__main__":
    main()
