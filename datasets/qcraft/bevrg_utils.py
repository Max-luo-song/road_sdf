import json
import logging
from pathlib import Path

import numpy as np
import torch


logger = logging.getLogger()


def load_lane_line_samples(data_path, start_frame, end_frame, device, spacing=0.05, stride=10):
    """Load BEVRG LANE_LINE polylines in the training-start LiDAR frame."""
    label_dir = Path(data_path) / "label" / "bev_road_geometry_label" / "bevrg_label_20260320"
    if not label_dir.is_dir():
        raise FileNotFoundError(f"BEVRG label directory not found: {label_dir}")

    paths = {int(p.stem): p for p in label_dir.glob("*.json") if p.stem.isdigit()}
    frame_ids = [i for i in sorted(paths) if start_frame <= i < end_frame]
    if not frame_ids:
        raise ValueError(f"No BEVRG labels overlap training frames: {label_dir}")
    frame_ids = frame_ids[::max(1, int(stride))]
    if frame_ids[-1] != max(i for i in paths if start_frame <= i < end_frame):
        frame_ids.append(max(i for i in paths if start_frame <= i < end_frame))

    origin = np.linalg.inv(np.loadtxt(Path(data_path) / "lidar_pose" / f"{start_frame:06d}.txt"))
    lines = {}
    for frame_id in frame_ids:
        ego_pose = Path(data_path) / "ego_pose" / f"{frame_id:06d}.txt"
        if not ego_pose.exists():
            continue
        ego_to_scene = origin @ np.loadtxt(ego_pose)
        payload = json.loads(paths[frame_id].read_text(encoding="utf-8"))
        for index, item in enumerate(payload.get("mapformer_lane_info_v2", [])):
            if item.get("coarse_lane_type") != "LANE_LINE" or item.get("is_filtered", False):
                continue
            points = np.asarray([
                [p.get("x"), p.get("y"), p.get("z")]
                for p in item.get("points", []) if p.get("is_ground_point", True)
            ], dtype=np.float64)
            if points.ndim != 2 or len(points) < 2:
                continue
            points = points[np.isfinite(points).all(axis=1)]
            if len(points) < 2:
                continue
            points = (ego_to_scene @ np.c_[points, np.ones(len(points))].T).T[:, :3]
            line_id = str(item.get("fsd_id", f"{frame_id}:{index}"))
            if line_id not in lines or len(points) > len(lines[line_id]):
                lines[line_id] = points

    samples, tangents = [], []
    for points in lines.values():
        for p0, p1 in zip(points[:-1], points[1:]):
            delta, length = p1 - p0, np.linalg.norm((p1 - p0)[:2])
            if not np.isfinite(length) or not 1e-4 < length <= 5.0:
                continue
            count = max(1, int(np.ceil(length / spacing)))
            samples.append(p0 + np.arange(count)[:, None] / count * delta)
            tangents.append(np.repeat((delta / length)[None], count, axis=0))

    if not samples:
        raise ValueError(f"No valid LANE_LINE geometry found: {label_dir}")
    points = torch.from_numpy(np.concatenate(samples)).float().to(device)
    tangents = torch.from_numpy(np.concatenate(tangents)).float().to(device)
    logger.info(f"Loaded {len(lines)} BEVRG lane lines ({len(points)} samples) from {label_dir}")
    return {"points": points, "tangents": tangents}
