from typing import List
from omegaconf import OmegaConf
import os
import time
import logging
import argparse
import imageio
import numpy as np

from datasets.driving_dataset import DrivingDataset
from utils.logging_utils import setup_logging


logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def _to_uint8_image(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def export_reference_videos(
    cfg: OmegaConf,
    cam_ids: List[int],
    fps: int,
    output_dir: str,
    overwrite: bool,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = DrivingDataset(data_cfg=cfg.data)

    for cam_id in cam_ids:
        if cam_id not in dataset.pixel_source.camera_data:
            raise KeyError(f"cam_id {cam_id} not found in dataset.pixel_source.camera_data")

        camera = dataset.pixel_source.camera_data[cam_id]
        video_path = os.path.join(output_dir, f"cam{cam_id}.mp4")
        if os.path.exists(video_path) and not overwrite:
            logger.info("Skip existing reference video: %s", video_path)
            continue

        logger.info(
            "Exporting reference video for cam%s (%s) -> %s",
            cam_id,
            camera.cam_name,
            video_path,
        )
        writer = imageio.get_writer(video_path, mode="I", fps=fps)
        try:
            for frame_idx in range(len(camera)):
                image_infos, _ = camera.get_image(frame_idx)
                rgb = image_infos["pixels"].detach().cpu().numpy()
                writer.append_data(_to_uint8_image(rgb))
        finally:
            writer.close()


def main(args: argparse.Namespace) -> None:
    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))

    if len(args.cam_ids) > 0:
        cfg.data.pixel_source.cameras = list(args.cam_ids)
    if len(args.downscales) > 0:
        if len(args.cam_ids) != len(args.downscales):
            raise ValueError("--cam_ids and --downscales must have the same length")
        cfg.data.pixel_source.downscale_when_loading = list(args.downscales)

    global logger
    setup_logging(level=logging.INFO, time_string=current_time)

    cam_ids = args.cam_ids if len(args.cam_ids) > 0 else list(cfg.data.pixel_source.cameras)
    export_reference_videos(
        cfg=cfg,
        cam_ids=cam_ids,
        fps=args.fps,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Export per-camera GT reference videos")
    parser.add_argument(
        "--resume_from",
        type=str,
        required=True,
        help="checkpoint path used only to locate the matching config.yaml",
    )
    parser.add_argument(
        "--cam_ids",
        nargs="*",
        type=int,
        default=[],
        help="camera ids to export; defaults to cfg.data.pixel_source.cameras",
    )
    parser.add_argument(
        "--downscales",
        nargs="*",
        type=float,
        default=[],
        help="optional per-camera downscales overriding config loading scale",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="fps of exported reference videos",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/refer_gt_video",
        help="directory to save per-camera reference videos",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing reference videos",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    main(parser.parse_args())
