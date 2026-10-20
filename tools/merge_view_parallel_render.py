import argparse
import os
import shutil
from typing import Dict, List

import imageio
import numpy as np

from datasets.qcraft.qcraft_config import ALL_CAM_SPECS
from utils.visualization import get_layout


def parse_int_list(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def collect_key_dirs(group_dirs: List[str]) -> List[str]:
    keys = set()
    for group_dir in group_dirs:
        if not os.path.isdir(group_dir):
            continue
        for name in os.listdir(group_dir):
            path = os.path.join(group_dir, name)
            if os.path.isdir(path) and not name.endswith("_layout"):
                keys.add(name)
    return sorted(keys)


def index_images(group_dirs: List[str], key: str) -> Dict[int, Dict[int, str]]:
    indexed: Dict[int, Dict[int, str]] = {}
    for group_dir in group_dirs:
        key_dir = os.path.join(group_dir, key)
        if not os.path.isdir(key_dir):
            continue
        for filename in os.listdir(key_dir):
            if not filename.endswith(".png"):
                continue
            stem = os.path.splitext(filename)[0]
            try:
                frame_id_str, cam_id_str = stem.split("_")
                frame_id = int(frame_id_str)
                cam_id = int(cam_id_str)
            except ValueError:
                continue
            indexed.setdefault(frame_id, {})[cam_id] = os.path.join(key_dir, filename)
    return indexed


def make_mp4(image_dir: str, output_path: str, fps: int) -> None:
    frames = [
        os.path.join(image_dir, name)
        for name in sorted(os.listdir(image_dir))
        if name.endswith(".png")
    ]
    if not frames:
        return
    writer = imageio.get_writer(output_path, mode="I", fps=fps)
    for frame_path in frames:
        writer.append_data(imageio.imread(frame_path))
    writer.close()


def merge_key(
    group_dirs: List[str],
    key: str,
    output_dir: str,
    view_order: List[int],
    layout,
    fps: int,
    strict: bool,
) -> None:
    indexed = index_images(group_dirs, key)
    if not indexed:
        return

    key_output_dir = os.path.join(output_dir, key)
    layout_output_dir = os.path.join(output_dir, f"{key}_layout")
    os.makedirs(key_output_dir, exist_ok=True)
    os.makedirs(layout_output_dir, exist_ok=True)

    cam_names = [ALL_CAM_SPECS[cam_id].name for cam_id in view_order]
    for frame_id in sorted(indexed):
        missing = [cam_id for cam_id in view_order if cam_id not in indexed[frame_id]]
        if missing:
            message = f"Frame {frame_id:06d} key {key} is missing cameras {missing}"
            if strict:
                raise FileNotFoundError(message)
            print(f"Warning: {message}; skipping frame")
            continue

        frames = []
        for cam_id in view_order:
            src = indexed[frame_id][cam_id]
            dst = os.path.join(key_output_dir, f"{frame_id:06d}_{cam_id:03d}.png")
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            frames.append(imageio.imread(src).astype(np.float32) / 255.0)

        names = list(cam_names)
        if layout.__name__ == "layout_qcraft":
            template = max(frames, key=lambda frame: frame.shape[0] * frame.shape[1])
            for spec in ALL_CAM_SPECS:
                if spec.name not in names:
                    frames.append(np.zeros_like(template))
                    names.append(spec.name)
        tiled = layout(frames, names)
        tiled = (255 * np.clip(tiled, 0, 1)).astype(np.uint8)
        imageio.imwrite(os.path.join(layout_output_dir, f"{frame_id:06d}.png"), tiled)

    make_mp4(layout_output_dir, os.path.join(output_dir, f"{key}_layout.mp4"), fps)


def main(args):
    group_dirs = [args.group0_dir, args.group1_dir]
    view_order = parse_int_list(args.view_order)
    layout = get_layout(args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.keys:
        keys = [key.strip() for key in args.keys.split(",") if key.strip()]
    else:
        keys = collect_key_dirs(group_dirs)

    print(f"Merging keys: {keys}")
    print(f"View order: {view_order}")
    print(f"Output dir: {args.output_dir}")

    for key in keys:
        merge_key(
            group_dirs=group_dirs,
            key=key,
            output_dir=args.output_dir,
            view_order=view_order,
            layout=layout,
            fps=args.fps,
            strict=args.strict,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Merge view-parallel render outputs into one camera layout")
    parser.add_argument("--group0_dir", required=True, type=str)
    parser.add_argument("--group1_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--view_order", default="0,1,2,3,5,6,7,9,10,11,12", type=str)
    parser.add_argument("--dataset", default="qcraft", type=str)
    parser.add_argument("--keys", default=None, type=str, help="comma-separated render keys; default merges all keys")
    parser.add_argument("--fps", default=10, type=int)
    parser.add_argument("--strict", action="store_true", help="fail if any frame is missing a camera")
    main(parser.parse_args())
