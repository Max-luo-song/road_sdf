from typing import List, Optional
from omegaconf import OmegaConf
import os
import shutil
import time
import json
import wandb
import logging
import argparse

import torch, numpy as np, subprocess
from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.trainers import BasicTrainer
from models.video_utils import (
    render_images,
    render_legend,
    save_videos,
    save_images as safe_save_images,
)

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def save_bboxes_json(instance_bboxes, save_path):
    serializable_data = []
    for item in instance_bboxes:
        serializable_data.append(
            {
                "frame": int(item["frame"]),
                "type": item["type"],
                "bbox_min": item["bbox_min"].detach().cpu().tolist(),
                "bbox_max": item["bbox_max"].detach().cpu().tolist(),
            }
        )

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)


def images_to_video(image_path, output_path, fps=10):
    """
    ffmpeg -framerate 10 -start_number 1 -i ./output/qcraft_20250902_163634_Q3703_1035_1055/20251216_lidar+cam0_1_2_3_5_6_7_9_10_11_12_test_with_novelview_2/videos_eval/full_set_40000_depths_layout/%06d.png -vf "scale=ceil(iw/2)*2:ceil(ih/2)*2:flags=lanczos,pad=ceil(iw/2)*2:ceil(ih/2)*2:(ow-iw)/2:(oh-ih)/2" -c:v libx264 -pix_fmt yuv420p ./output/qcraft_20250902_163634_Q3703_1035_1055/20251216_lidar+cam0_1_2_3_5_6_7_9_10_11_12_test_with_novelview_2/full_set_40000_depths.mp4
    """
    cmd = [
        "ffmpeg -framerate",
        str(fps),
        "-start_number 1 -i",
        os.path.join(image_path, "%06d.png"),
        '-vf "scale=ceil(iw/2)*2:ceil(ih/2)*2:flags=lanczos,pad=ceil(iw/2)*2:ceil(ih/2)*2:(ow-iw)/2:(oh-ih)/2"',
        "-c:v libx264 -pix_fmt yuv420p -y",
        output_path,
    ]
    os.system(" ".join(cmd))


@torch.no_grad()
def do_evaluation(
    step: int = 0,
    cfg: OmegaConf = None,
    trainer: BasicTrainer = None,
    dataset: DrivingDataset = None,
    args: argparse.Namespace = None,
    render_keys: Optional[List[str]] = None,
    post_fix: str = "",
    log_metrics: bool = True,
    edit_cfg: OmegaConf = None,
    render_cam_indices: Optional[List[int]] = None,
):
    trainer.set_eval()

    logger.info("Evaluating Pixels...")

    if cfg.render.render_full:
        logger.info("Evaluating Full Set...")

        # get self trajectory
        trajectory = dataset.full_image_set.front_camera_trajectory()
        # edit gaussians
        if "Nodes" in edit_cfg and edit_cfg.Nodes != None:
            instance_bboxes = trainer.edit_gaussians(
                edit_cfg=edit_cfg, self_trajectory=trajectory, dataset=dataset.full_image_set
            )
            # save_bboxes_json(instance_bboxes, f"{cfg.log_dir}/videos{post_fix}/barrier_instance_bboxes.json")

            # 保存编辑后的 checkpoint
            output_ckpt_dir = getattr(edit_cfg, "output_ckpt_dir", None)
            if output_ckpt_dir is not None and output_ckpt_dir != "":
                os.makedirs(output_ckpt_dir, exist_ok=True)
                shutil.copy(
                    os.path.join(os.path.dirname(args.resume_from), "config.yaml"),
                    os.path.join(output_ckpt_dir, "config.yaml"),
                )
                shutil.copy(
                    args.edit_config,
                    os.path.join(output_ckpt_dir, "edit_config.yaml"),
                )
                trainer.save_checkpoint(
                    log_dir=output_ckpt_dir,
                    save_only_model=True,
                    is_final=True,
                )
                print(f"[info] 编辑后的 checkpoint 已保存至 {output_ckpt_dir}")
        else:
            print(
                "Warning: 'Nodes' key not found in the configuration or No values found in 'Nodes'."
            )

        # 保存编辑后的 checkpoint 到视频输出目录
        save_ckpt = getattr(edit_cfg, "save_ckpt", False)
        if save_ckpt:
            ckpt_output_dir = f"{cfg.log_dir}/videos{post_fix}"
            os.makedirs(ckpt_output_dir, exist_ok=True)
            shutil.copy(
                os.path.join(os.path.dirname(args.resume_from), "config.yaml"),
                os.path.join(ckpt_output_dir, "config.yaml"),
            )
            shutil.copy(
                args.edit_config,
                os.path.join(ckpt_output_dir, "edit_config.yaml"),
            )
            trainer.save_checkpoint(
                log_dir=ckpt_output_dir,
                save_only_model=True,
                is_final=True,
            )
            print(f"[info] 编辑后的 checkpoint 已保存至 {ckpt_output_dir}")

        if args.render_video_postfix is None:
            video_output_pth = f"{cfg.log_dir}/videos{post_fix}/full_set_{step}.mp4"
        else:
            video_output_pth = f"{cfg.log_dir}/videos{post_fix}/full_set_{step}_{args.render_video_postfix}.mp4"

        all_metrics = [
            "psnr",
            "ssim",
            "lpips",
            "psnr_no_ego",
            "ssim_no_ego",
            "lpips_no_ego",
            "psnr_with_ego",
            "ssim_with_ego",
            "lpips_with_ego",
            "occupied_psnr",
            "occupied_ssim",
            "masked_psnr",
            "masked_ssim",
            "human_psnr",
            "human_ssim",
            "vehicle_psnr",
            "vehicle_ssim",
        ]
        eval_dict = dict(zip(all_metrics, [[] for _ in range(len(all_metrics))]))

        for i in range(0, dataset.num_img_timesteps):
            print(f"crrent render {i} batch, total length {dataset.num_img_timesteps}")
            if render_cam_indices is not None:
                vis_timestep = np.array([i * dataset.num_cams + ci for ci in render_cam_indices])
            else:
                vis_timestep = np.arange(0, dataset.num_img_timesteps * dataset.num_cams)[
                    i * dataset.num_cams : (i + 1) * dataset.num_cams
                ]
            with torch.no_grad():
                render_results = render_images(
                    trainer=trainer,
                    dataset=dataset.full_image_set,
                    compute_metrics=True,
                    compute_error_map=cfg.render.vis_error,
                    vis_indices=vis_timestep,
                    edit_cfg=edit_cfg,
                )

                if log_metrics:
                    for k, v in render_results.items():
                        if k in all_metrics:
                            eval_dict[k] += [v]

                vis_frame_dict = safe_save_images(
                    render_results,
                    video_output_pth,
                    layout=dataset.layout,
                    timestamps=i,
                    keys=render_keys,
                    verbose=True,
                )

                if args.enable_wandb:
                    for k, v in vis_frame_dict.items():
                        wandb.log({"image_rendering/full/" + k: wandb.Image(v)})

        torch.cuda.empty_cache()

        for dirs in os.listdir(f"{cfg.log_dir}/videos{post_fix}"):
            dir_path = f"{cfg.log_dir}/videos{post_fix}/{dirs}"
            if (
                "layout" in dirs
                and os.path.isdir(dir_path)  # 先判断是目录
                and len(os.listdir(dir_path)) > 0
            ):
                img_dir = dir_path
                save_mp4 = f"{cfg.log_dir}/videos{post_fix}/{dirs}.mp4"
                # 如果 mp4 已存在，先删除
                if os.path.exists(save_mp4):
                    os.remove(save_mp4)
                images_to_video(img_dir, save_mp4, cfg.render.fps)

        fine_eval_dict = dict()
        if log_metrics:
            for k, v in eval_dict.items():
                if k in all_metrics:
                    fine_eval_dict[f"image_metrics/full/{k}"] = float(
                        sum(eval_dict[k]) / len(eval_dict[k])
                        if len(eval_dict[k]) > 0
                        else -1
                    )

    legend_cfg = getattr(edit_cfg, "legend", None)
    if legend_cfg and any(getattr(legend_cfg, k, False) for k in ["rigid", "smpl"]):
        image_output_pth = f"{cfg.log_dir}/videos{post_fix}"
        video_output_pth = f"{cfg.log_dir}/videos{post_fix}/rigid_legend.mp4"

        trainer.color_legend(edit_cfg=edit_cfg, image_output_pth=image_output_pth)

        for i in range(0, dataset.num_img_timesteps):
            if render_cam_indices is not None:
                vis_timestep = np.array([i * dataset.num_cams + ci for ci in render_cam_indices])
            else:
                vis_timestep = np.arange(0, dataset.num_img_timesteps * dataset.num_cams)[
                    i * dataset.num_cams : (i + 1) * dataset.num_cams
                ]
            legend_results = render_legend(
                trainer=trainer,
                dataset=dataset.full_image_set,
                compute_error_map=cfg.render.vis_error,
                vis_indices=vis_timestep,
            )
            safe_save_images(
                legend_results,
                video_output_pth,
                layout=dataset.layout,
                timestamps=i,
                keys=["rgbs"],
                verbose=True,
            )

        os.makedirs(os.path.dirname(video_output_pth), exist_ok=True)
        for dirs in os.listdir(f"{cfg.log_dir}/videos{post_fix}"):
            if (
                "rigid_legend" in dirs
                and "mp4" not in dirs
                and os.path.isdir(f"{cfg.log_dir}/videos{post_fix}/{dirs}")
            ):
                img_dir = f"{cfg.log_dir}/videos{post_fix}/{dirs}"
                save_mp4 = f"{cfg.log_dir}/videos{post_fix}/{dirs}.mp4"
                images_to_video(img_dir, save_mp4, cfg.render.fps)


def main(args):
    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    args.enable_wandb = False
    output_path = os.path.join(cfg.log_dir, f"videos{args.post_fix}")
    os.makedirs(output_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # edit config
    edit_cfg = OmegaConf.load(args.edit_config)
    # weather particle config
    particle_cfg = OmegaConf.load("./configs/particle_config.yaml")
    # build dataset
    if edit_cfg.weather_type != "":
        dataset = DrivingDataset(data_cfg=cfg.data, weather_type=edit_cfg.weather_type)
    else:
        dataset = DrivingDataset(data_cfg=cfg.data)

    # setup trainer
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
        particle_cfg=particle_cfg,
        weather_type=edit_cfg.weather_type,
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

    # define render keys
    render_keys = [
        "gt_rgbs",
        "rgbs",
        # "Background_rgbs",
        # "RigidNodes_rgbs",
        # "DeformableNodes_rgbs",
        # "SMPLNodes_rgbs",
        # "depths",
        # "Background_depths",
        # "RigidNodes_depths",
        # "DeformableNodes_depths",
        # "SMPLNodes_depths",
        # "mask"
    ]
    if cfg.render.vis_lidar:
        render_keys.insert(0, "lidar_on_images")
    if cfg.render.vis_sky:
        render_keys += ["rgb_sky_blend", "rgb_sky"]
    if cfg.render.vis_error:
        render_keys.insert(render_keys.index("rgbs") + 1, "rgb_error_maps")

    if args.save_catted_videos:
        cfg.logging.save_seperate_video = False

    # map requested cam_ids to unique_cam_idx
    render_cam_indices = None
    if args.render_cam_ids is not None:
        cam_id_to_unique = {}
        for cam_key, cam_data in dataset.pixel_source.camera_data.items():
            cam_id_to_unique[cam_data.cam_id] = cam_data.unique_cam_idx
        render_cam_indices = []
        for cid in args.render_cam_ids:
            if cid not in cam_id_to_unique:
                logger.warning(f"Camera ID {cid} not found in dataset, skipping. Available: {sorted(cam_id_to_unique.keys())}")
            else:
                render_cam_indices.append(cam_id_to_unique[cid])
        if not render_cam_indices:
            logger.warning("No valid camera IDs provided, falling back to render all.")
            render_cam_indices = None
        else:
            logger.info(f"Rendering cameras: {args.render_cam_ids} -> unique_cam_idx: {render_cam_indices}")

    do_evaluation(
        step=trainer.step,
        cfg=cfg,
        trainer=trainer,
        dataset=dataset,
        render_keys=render_keys,
        args=args,
        post_fix=args.post_fix,
        log_metrics=False,
        edit_cfg=edit_cfg,
        render_cam_indices=render_cam_indices,
    )

    if args.enable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Gaussian Splatting for a single scene")
    # eval
    parser.add_argument(
        "--resume_from",
        default=None,
        help="path to checkpoint to resume from",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--render_video_postfix",
        type=str,
        default=None,
        help="an optional postfix for video",
    )
    parser.add_argument(
        "--save_catted_videos",
        type=bool,
        default=False,
        help="visualize lidar on image",
    )
    parser.add_argument(
        "--edit_config",
        type=str,
        default=False,
        help="config for edit",
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

    # output
    parser.add_argument(
        "--post_fix",
        type=str,
        default="_edit",
        help="Postfix for output foldernames (default: _edit_barrier)",
    )

    # camera filter
    parser.add_argument(
        "--render_cam_ids",
        nargs="*",
        type=int,
        default=None,
        help="Camera IDs to render (e.g. --render_cam_ids 0 1 5). Default: render all.",
    )

    args = parser.parse_args()
    main(args)
