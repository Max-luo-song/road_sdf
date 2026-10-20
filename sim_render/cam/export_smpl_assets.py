from typing import List, Optional
from omegaconf import OmegaConf
import os
import logging
import argparse

import torch
from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.trainers import BasicTrainer

logger = logging.getLogger()


def parse_instance_ids(instance_ids: Optional[List[int]], max_instances: int) -> List[int]:
    if instance_ids is None or len(instance_ids) == 0:
        return list(range(max_instances))
    valid_ids = []
    for instance_id in instance_ids:
        if instance_id < 0 or instance_id >= max_instances:
            raise ValueError(f"Invalid instance id {instance_id}, valid range: [0, {max_instances - 1}]")
        valid_ids.append(instance_id)
    return valid_ids


def main(args):
    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DrivingDataset(data_cfg=cfg.data)

    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )
    trainer.resume_from_checkpoint(
        ckpt_path=args.resume_from,
        load_only_model=True,
    )
    logger.info(f"Resuming from {args.resume_from}, step {trainer.step}")

    if "SMPLNodes" not in trainer.models:
        raise ValueError("SMPLNodes is not enabled in this checkpoint.")
    smpl_model = trainer.models["SMPLNodes"]

    if not args.export_ply and not args.export_motion:
        args.export_ply = True
        args.export_motion = True

    output_dir = args.output_dir if args.output_dir is not None else os.path.join(log_dir, "smpl_assets")
    os.makedirs(output_dir, exist_ok=True)

    target_ids = parse_instance_ids(args.instance_ids, smpl_model.num_instances)
    logger.info(f"Exporting SMPL instances: {target_ids}")
    logger.info(f"Output dir: {output_dir}")

    for instance_id in target_ids:
        stem = f"smpl_instance_{instance_id:03d}"
        if args.export_ply:
            smpl_model.export_instance_to_ply(
                path=os.path.join(output_dir, f"{stem}.ply"),
                instance_id=instance_id,
                alpha_thresh=args.alpha_thresh,
            )
        if args.export_motion:
            smpl_model.export_instance_motion_to_pt(
                path=os.path.join(output_dir, f"{stem}_motion.pt"),
                instance_id=instance_id,
            )
    logger.info("Export finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Export SMPL assets from checkpoint")
    parser.add_argument(
        "--resume_from",
        type=str,
        required=True,
        help="path to checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="output directory, default: <log_dir>/smpl_assets",
    )
    parser.add_argument(
        "--instance_ids",
        nargs="*",
        type=int,
        default=None,
        help="instance ids to export, default: all",
    )
    parser.add_argument(
        "--alpha_thresh",
        type=float,
        default=0.001,
        help="opacity threshold for Gaussian PLY export",
    )
    parser.add_argument("--export_ply", action="store_true", help="export Gaussian ply assets")
    parser.add_argument("--export_motion", action="store_true", help="export motion pt assets")
    parser.add_argument(
        "opts",
        help="Modify config options using command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    main(parser.parse_args())
