import argparse
import glob
import os
import re
import sys
from typing import List

import imageio.v2 as imageio
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datasets.qcraft.qcraft_config import ALL_CAM_SPECS
from utils.visualization import get_layout


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def find_latest_traj_dir(novel_traj_dir: str, traj_type: str) -> str:
    candidates = [
        path
        for path in glob.glob(os.path.join(novel_traj_dir, f"{traj_type}_step*"))
        if os.path.isdir(path)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No rendered directory found for {traj_type} under {novel_traj_dir}"
        )
    step_pattern = re.compile(rf"^{re.escape(traj_type)}_step(\d+)")

    def sort_key(path: str):
        match = step_pattern.match(os.path.basename(path))
        step = int(match.group(1)) if match else -1
        return step, os.path.getmtime(path)

    return max(candidates, key=sort_key)


def make_layout_video(
    traj_dir: str,
    traj_type: str,
    view_order: List[int],
    fps: int,
    strict: bool,
) -> str:
    image_root = os.path.join(traj_dir, "images")
    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"Novel-trajectory images not found: {image_root}")

    frame_dirs = sorted(
        path
        for path in glob.glob(os.path.join(image_root, "*"))
        if os.path.isdir(path) and os.path.basename(path).isdigit()
    )
    if not frame_dirs:
        raise FileNotFoundError(f"No rendered frames found under: {image_root}")

    if strict:
        for frame_dir in frame_dirs:
            missing = [
                cam_id
                for cam_id in view_order
                if not os.path.isfile(os.path.join(frame_dir, f"{cam_id}_rgbs.png"))
            ]
            if missing:
                raise FileNotFoundError(
                    f"Frame {os.path.basename(frame_dir)} is missing cameras {missing}"
                )

    video_dir = os.path.join(traj_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    output_path = os.path.join(video_dir, f"{traj_type}_rgbs_layout.mp4")

    layout = get_layout("qcraft")
    cam_names = [ALL_CAM_SPECS[cam_id].name for cam_id in view_order]
    writer = imageio.get_writer(output_path, mode="I", fps=fps)
    written = 0
    try:
        for frame_dir in frame_dirs:
            missing = [
                cam_id
                for cam_id in view_order
                if not os.path.isfile(os.path.join(frame_dir, f"{cam_id}_rgbs.png"))
            ]
            if missing:
                message = f"Frame {os.path.basename(frame_dir)} is missing cameras {missing}"
                if strict:
                    raise FileNotFoundError(message)
                print(f"Warning: {message}; skipping frame")
                continue

            frames = [
                imageio.imread(os.path.join(frame_dir, f"{cam_id}_rgbs.png")).astype(
                    np.float32
                )
                / 255.0
                for cam_id in view_order
            ]
            # Preserve the full layout and ignore stale images from unselected groups.
            names = list(cam_names)
            template = max(frames, key=lambda frame: frame.shape[0] * frame.shape[1])
            for spec in ALL_CAM_SPECS:
                if spec.name not in names:
                    frames.append(np.zeros_like(template))
                    names.append(spec.name)
            tiled = layout(frames, names)
            writer.append_data((255 * np.clip(tiled, 0.0, 1.0)).astype(np.uint8))
            written += 1
    finally:
        writer.close()

    if written == 0:
        raise RuntimeError(f"No complete frames were written for {traj_type}")
    print(f"Saved full-camera layout video: {output_path} ({written} frames)")
    return output_path


def main(args: argparse.Namespace) -> None:
    view_order = parse_int_list(args.view_order)
    for traj_type in args.traj_types:
        traj_dir = find_latest_traj_dir(args.novel_traj_dir, traj_type)
        make_layout_video(
            traj_dir=traj_dir,
            traj_type=traj_type,
            view_order=view_order,
            fps=args.fps,
            strict=args.strict,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Merge grouped novel-trajectory images into full-camera layout videos"
    )
    parser.add_argument("--novel_traj_dir", required=True, type=str)
    parser.add_argument("--traj_types", required=True, nargs="+")
    parser.add_argument(
        "--view_order", default="0,1,2,3,5,6,7,9,10,11,12", type=str
    )
    parser.add_argument("--fps", default=10, type=int)
    parser.add_argument("--strict", action="store_true")
    main(parser.parse_args())
