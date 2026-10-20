#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np

import subprocess

cmd = "nvidia-smi -q -d Memory |grep -A4 GPU|grep Used"
result = (
    subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split("\n")
)
os.environ["CUDA_VISIBLE_DEVICES"] = str(
    np.argmin([int(x.split()[2]) for x in result[:-1]])
)

os.system("echo $CUDA_VISIBLE_DEVICES")

from skimage.metrics import structural_similarity
import torch
import torchvision
import json
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
from random import randint
from utils.loss_utils import l1_loss, ssim, raydrop_lossf, l2_loss
from gaussian_renderer import prefilter_voxel, render, renderComposite
import sys
from scene import Scene, GaussianModel
from scene.cameras import Camera

from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.lidar_utils import pano_to_lidar_with_intensities, pano_to_lidar
from utils.lidar_utils import PointsMeter, filter_pcd
from utils.data_partition_utils import (
    dataPartition,
    getBlockInfo,
    dataPartitionSimple,
    dataPartitionChery,
)

from typing import NamedTuple
import time
import pdb
from scipy.spatial.transform import Rotation

cpu_count = os.cpu_count()
# Reduce PyTorch intra-op threads to avoid saturating CPU and improve GPU throughput
torch.set_num_threads(max(1, cpu_count // 2))

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")


class GaussianView(NamedTuple):
    gaussians: GaussianModel
    scene: Scene
    time_poses: dict


class TimePose(NamedTuple):
    camera_pose: np.array
    view: Camera


class ValidModeInfo(NamedTuple):
    model_id: int
    model_pose: np.array
    model_view: Camera
    model_gaussians: GaussianModel
    need_train: bool
    model_world_rotation: torch.Tensor = None
    model_world_translation: torch.Tensor = None
    model_world_quaternion: torch.Tensor = None


def build_dynamic_model_id_list(gt_dynamic_model):
    model_id_list = [0]
    dynamic_ids = gt_dynamic_model.get_dynamic_obj_id_list()
    if dynamic_ids is not None:
        model_id_list.extend(dynamic_ids)
    return list(dict.fromkeys(model_id_list))


def training(
    gt_dynamic_model,
    dataset,
    opt,
    pipe,
    dataset_name,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    logger=None,
    ply_path=None,
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    model_id_list = build_dynamic_model_id_list(gt_dynamic_model)

    static_views = None
    model_id_scene_info = {}
    for model_id in model_id_list:
        model_gaussians = GaussianModel(
            dataset.feat_dim,
            dataset.n_offsets,
            dataset.voxel_size,
            dataset.update_depth,
            dataset.update_init_factor,
            dataset.update_hierachy_factor,
            dataset.use_feat_bank,
            dataset.appearance_dim,
            dataset.ratio,
            dataset.add_opacity_dist,
            dataset.add_cov_dist,
            dataset.add_color_dist,
            dataset.color_channel,
        )
        model_scene = Scene(
            dataset,
            model_id,
            gt_dynamic_model,
            model_gaussians,
            ply_path=ply_path,
            shuffle=False,
        )

        if model_scene.init_status:
            model_gaussians.training_setup(opt)
            if checkpoint:
                (model_params, first_iter) = torch.load(checkpoint)
                model_gaussians.restore(model_params, opt)

            time_poses = {}
            total_views = model_scene.getTotalCameras()
            print("total_views ", len(total_views))

            if model_id == 0:
                static_views = total_views

            for view in total_views:
                timestamp = view.image_name
                camera_pose = np.eye(4)
                camera_pose[:3, :3] = np.transpose(view.R)
                camera_pose[:3, 3] = view.T
                time_poses[timestamp] = TimePose(
                    camera_pose=camera_pose,
                    view=view,
                )

            model_id_scene_info[model_id] = GaussianView(
                gaussians=model_gaussians, scene=model_scene, time_poses=time_poses
            )

    test_timestamp = []
    if static_views is None:
        print("ERROR: empty static_views")
        sys.exit(1)
    for i in range(1, len(static_views), dataset.test_frames_interval):
        test_timestamp.append(int(static_views[i].image_name))

    train_views = []
    test_views = []
    for idx, scene_view in enumerate(static_views):
        render_timestamp = int(scene_view.image_name)
        if render_timestamp in test_timestamp:
            test_views.append(scene_view)
        else:
            train_views.append(scene_view)
    if test_views is None or train_views is None:
        print("Error: test_views or train_views is None")
        return

    logger.info(f"block id: {dataset.block_id}")
    logger.info(f"model id: {list(model_id_scene_info.keys())}")
    valid_timestamp_model = {}
    static_camera_to_world = {}
    for view in static_views:
        timestamp = view.image_name
        world_to_camera_pose = np.eye(4)
        world_to_camera_pose[:3, :3] = np.transpose(view.R)
        world_to_camera_pose[:3, 3] = view.T
        static_camera_to_world[timestamp] = np.linalg.inv(world_to_camera_pose)
        valid_timestamp_model[timestamp] = []
        for model_id in model_id_scene_info.keys():
            if timestamp in model_id_scene_info[model_id].time_poses:
                valid_timestamp_model[timestamp].append(
                    model_id
                )  # 记录每一帧timestep（0-50）下出现的obj的id

    model_world_transform_cache = {}
    for model_id, scene_info in model_id_scene_info.items():
        world_transforms = {}
        for timestamp, time_pose in scene_info.time_poses.items():
            model_to_world = static_camera_to_world[timestamp] @ time_pose.camera_pose
            model_world_rotation = torch.tensor(
                model_to_world[:3, :3], dtype=torch.float32, device="cuda"
            )
            model_world_translation = torch.tensor(
                model_to_world[:3, 3], dtype=torch.float32, device="cuda"
            )
            model_world_quaternion_np = Rotation.from_matrix(
                model_to_world[:3, :3]
            ).as_quat()  # x, y, z, w
            model_world_quaternion = torch.tensor(
                [
                    model_world_quaternion_np[3],
                    model_world_quaternion_np[0],
                    model_world_quaternion_np[1],
                    model_world_quaternion_np[2],
                ],
                dtype=torch.float32,
                device="cuda",
            )
            world_transforms[timestamp] = (
                model_world_rotation,
                model_world_translation,
                model_world_quaternion,
            )
        model_world_transform_cache[model_id] = world_transforms

    viewpoint_stack = []
    ema_loss_for_log = 0.0
    ema_depth_loss_for_log = 0.0
    ema_intensity_loss_for_log = 0.0
    ema_scale_for_log = 0.0
    ema_dp_for_log = 0.0
    ema_dgx_for_log = 0.0
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )
    raydrop_loss_fn = torch.nn.BCELoss()
    progress_bar = tqdm(
        range(first_iter, opt.iterations),
        leave=True,
        dynamic_ncols=True,
        desc="Training progress",
    )
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        # Pick a random Camera
        if len(viewpoint_stack) == 0:
            viewpoint_stack = train_views.copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        render_timestamp = viewpoint_cam.image_name

        # if iteration == (opt.update_until+1) and opt.multistep: # 3k iter 后就不优化主要的gs属性了
        #     for model_id, scene_info in model_id_scene_info.items():
        #         model_gaussian = scene_info.gaussians
        #         model_gaussian.freeze_param() # 冻结除了raydrop外其他属性

        valid_model_info = []
        for model_id in valid_timestamp_model[render_timestamp]:
            time_pose = model_id_scene_info[model_id].time_poses[render_timestamp]
            model_to_camera_pose = time_pose.camera_pose
            camera_to_world_pose = static_camera_to_world[render_timestamp]
            object_view = time_pose.view

            model_gaussian = model_id_scene_info[model_id].gaussians
            model_gaussian.update_learning_rate(iteration)
            model_to_world = (
                camera_to_world_pose @ model_to_camera_pose
            )  # for id=0, static scene, this is c2w @ w2c
            (
                model_world_rotation,
                model_world_translation,
                model_world_quaternion,
            ) = model_world_transform_cache[model_id][render_timestamp]
            valid_model_info.append(
                ValidModeInfo(
                    model_id=model_id,
                    model_pose=model_to_world,
                    model_view=object_view,
                    model_gaussians=model_gaussian,
                    need_train=True,
                    model_world_rotation=model_world_rotation,
                    model_world_translation=model_world_translation,
                    model_world_quaternion=model_world_quaternion,
                )
            )

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        retain_grad = iteration < opt.update_until and iteration >= 0
        render_pkg = renderComposite(
            viewpoint_cam,
            background,
            pipe,
            valid_model_info,
            dataset.max_depth,
            retain_grad=retain_grad,
        )
        (
            image,
            depth,
            viewspace_point_tensor,
            visibility_filter,
            offset_selection_mask,
            radii,
            scaling,
            opacity,
        ) = (
            render_pkg["render"],
            render_pkg["depth"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["selection_mask"],
            render_pkg["radii"],
            render_pkg["scaling"],
            render_pkg["neural_opacity"],
        )
        if radii.sum() == 0:
            print("No Gaussians are visible!")
        points_with_model_id = render_pkg["points_with_model_id"]
        visable_mask = render_pkg["visable_mask"]

        # --------------- intensity & depth loss -----------------------#
        gt_image = viewpoint_cam.original_image.cuda()
        gt_objmask = viewpoint_cam.img_mask
        ray_drop = gt_image[0:1, ...]
        gt_intensity = gt_image[1:2, ...] * ray_drop * gt_objmask
        gt_depth = gt_image[2:3, ...] * ray_drop * gt_objmask
        render_intensity = image[0:1, ...]
        render_raydrop = image[1:2, ...]

        if False:
            render_raydrop_mask = torch.where(render_raydrop > 0.5, 1, 0)
            render_intensity = render_intensity * render_raydrop_mask * gt_objmask
            depth = depth * render_raydrop_mask * gt_objmask
            mse_loss = torch.nn.MSELoss()
            raydrop_loss = mse_loss(render_raydrop, ray_drop)
        else:
            render_intensity = (
                render_intensity * ray_drop * gt_objmask
            )  # 直接使用gt的raydrop mask
            depth = depth * ray_drop * gt_objmask
            # Fix: render_raydrop is already in [0,1] (sigmoid applied), so use BCELoss
            render_raydrop = torch.clamp(render_raydrop, 1e-6, 1.0 - 1e-6)
            raydrop_loss = raydrop_loss_fn(render_raydrop, ray_drop)

        Ll1 = l1_loss(render_intensity, gt_intensity)
        depth_loss = l1_loss(depth, gt_depth)
        ssim_loss = 1.0 - ssim(render_intensity, gt_intensity)

        # ---------------------- scale reg ----------------------------#
        scaling_reg = 0.02 * scaling.prod(dim=1).mean()
        # if visibility_filter.sum() > 0:
        #     scale = scaling[visibility_filter]
        #     sorted_scale, _ = torch.sort(scale, dim=-1)
        #     min_scale_loss = sorted_scale[...,0]
        #     plane_reg = 10.0*min_scale_loss.mean()
        #     scaling_reg = scaling_reg + plane_reg

        # ---------------------- occ reg ------------------------------#
        # alpha = render_pkg['occ']
        # o = alpha.clamp(1e-6, 1-1e-6)
        # loss_opa = -(o*torch.log(o)).mean() - ((1-o) * torch.log(1 - o)).mean() # opacity要么为0要么为1
        # loss_opa = 0.05 * loss_opa
        # ------------------------ grad loss -------------------------#
        pred_grad_x = torch.abs(depth[:, :, :-1] - depth[:, :, 1:])
        gt_grad_x = torch.abs(gt_depth[:, :, :-1] - gt_depth[:, :, 1:])
        grad_clip_x = 0.1
        grad_mask_x = torch.where(gt_grad_x < grad_clip_x, 1, 0)
        mask_dx = ray_drop[:, :, :-1] * grad_mask_x
        grad_loss = l1_loss(pred_grad_x * mask_dx, gt_grad_x * mask_dx)

        # dist_loss = lambda_dist * (rend_dist).mean()
        intensity_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss

        loss = depth_loss + scaling_reg + intensity_loss + grad_loss + raydrop_loss

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"[ERROR] Loss is NaN or Inf at iteration {iteration}")
            print(
                f"Loss components: depth={depth_loss.item()}, scaling_reg={scaling_reg.item()}, intensity={intensity_loss.item()}, grad={grad_loss.item()}"
            )
            # 打印 scaling 的统计信息
            # 保存当前模型状态用于调试
            torch.save(
                {
                    "iteration": iteration,
                    "model_state": model_gaussian.state_dict(),
                    "render_stats": {
                        "depth_nan": torch.isnan(depth).sum().item(),
                        "image_nan": torch.isnan(image).sum().item(),
                    },
                },
                f"debug_model_nan_{iteration}.pth",
            )
            raise ValueError("Loss is NaN or Inf")

        loss.backward()

        if iteration % 100 == 0:
            with open(os.path.join(dataset.model_path, "grad_log.txt"), "a") as f:
                f.write(f"Iteration {iteration}:\n")
                f.write(f"Losses: Total={loss.item():.6f}, RayDrop={raydrop_loss.item():.6f}, Intensity={intensity_loss.item():.6f}, Depth={depth_loss.item():.6f}, Grad={grad_loss.item():.6f}\n")
                if 0 in model_id_scene_info:
                    model_gaussian = model_id_scene_info[0].gaussians
                    for name, param in model_gaussian.named_parameters():
                        if param.grad is not None:
                            grad_mean = param.grad.mean().item()
                            grad_std = param.grad.std().item()
                            grad_abs_max = param.grad.abs().max().item()
                            f.write(f"{name}: mean={grad_mean:.2e}, std={grad_std:.2e}, abs_max={grad_abs_max:.2e}\n")
                f.write("-" * 20 + "\n")

        # 打印各参数梯度, 观察是否有nan，过大或者过小的梯度（每100步检查一次，避免每步同步拖慢训练）
        if iteration % 100 == 0:
            model_gaussian = model_id_scene_info[0].gaussians
            for name, param in model_gaussian.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(
                            f"[WARNING] NaN/Inf found in gradient of {name} at iteration {iteration}"
                        )

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dp_for_log = 0.4 * raydrop_loss.item() + 0.6 * ema_dp_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.{7}f}",
                        "i_L": f"{intensity_loss.item():.{7}f}",
                        "d_L": f"{depth_loss.item():.{7}f}",
                        "dp_L": f"{ema_dp_for_log:.{7}f}",
                        "anchor": f"{len(model_id_scene_info[0].gaussians.get_anchor)}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration in testing_iterations:
                save_unet_refine_data = (
                    iteration == opt.iterations
                ) and dataset.enable_raydrop_unet
                print("---------------------train----------------------")
                train_composite_report(
                    tb_writer,
                    dataset,
                    dataset_name,
                    iteration,
                    train_views,
                    test_timestamp,
                    model_id_scene_info,
                    model_world_transform_cache,
                    valid_timestamp_model,
                    pipe,
                    background,
                    logger,
                    is_train_views=True,
                    save_unet_refine_data=save_unet_refine_data,
                )

                print("---------------------test----------------------")
                train_composite_report(
                    tb_writer,
                    dataset,
                    dataset_name,
                    iteration,
                    test_views,
                    test_timestamp,
                    model_id_scene_info,
                    model_world_transform_cache,
                    valid_timestamp_model,
                    pipe,
                    background,
                    logger,
                    is_train_views=False,
                    save_unet_refine_data=save_unet_refine_data,
                )
            for model_id, model_info in model_id_scene_info.items():
                densify_until_num_points = opt.densify_until_num_points

                if iteration in saving_iterations:
                    model_scene = model_info.scene
                    model_scene.save(iteration, model_id)
                    if model_id == 0:
                        xyz_numpy = render_pkg["xyz"].detach().cpu().numpy()
                        opacity_numpy = render_pkg["opa"].detach().cpu().numpy()
                        out_save = np.concatenate((xyz_numpy, opacity_numpy), axis=1)
                        np.savetxt(
                            os.path.join(model_info.scene.model_path, "all_offset.txt"),
                            out_save,
                            fmt="%.4f",
                            comments="",
                        )

                model_gaussian = model_info.gaussians

                # densification
                if (
                    iteration < opt.update_until
                    and iteration > opt.start_stat
                    and model_gaussian.get_anchor.shape[0] < densify_until_num_points
                ):
                    if model_id in valid_timestamp_model[render_timestamp]:
                        curr_viewspace_point_tensor_grad = viewspace_point_tensor.grad[
                            points_with_model_id == model_id
                        ]
                        curr_visibility_filter = visibility_filter[
                            points_with_model_id == model_id
                        ]
                        model_gaussian.training_statis(
                            curr_viewspace_point_tensor_grad,
                            opacity[model_id],
                            curr_visibility_filter,
                            offset_selection_mask[model_id],
                            visable_mask[model_id],
                        )

                    # densification
                    if (
                        iteration > opt.update_from
                        and iteration % opt.update_interval == 0
                    ):  # TODO 如果影响震荡 就把update_interval放大
                        densify_grad_threshold = opt.densify_grad_threshold
                        model_gaussian.adjust_anchor(
                            check_interval=opt.update_interval,
                            success_threshold=opt.success_threshold,
                            grad_threshold=densify_grad_threshold,
                            min_opacity=opt.min_opacity,
                        )

                elif iteration == opt.update_until:
                    del model_gaussian.opacity_accum
                    del model_gaussian.offset_gradient_accum
                    del model_gaussian.offset_denom
                    torch.cuda.empty_cache()

                # Optimizer step
                if iteration < opt.iterations:
                    model_gaussian.optimizer.step()
                    model_gaussian.optimizer.zero_grad(set_to_none=True)


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def train_composite_report(
    tb_writer,
    model_args,
    dataset_name,
    iteration,
    static_views,
    test_timestamp,
    model_id_scene_info,
    model_world_transform_cache,
    valid_timestamp_model,
    pipe,
    background,
    logger,
    is_train_views=True,
    save_unet_refine_data=False,
):
    l1_test = 0.0
    psnr_test = 0.0
    in_mae = 0.0
    in_rmse = 0.0
    in_medae = 0.0
    in_ssim = 0.0
    # point
    cd_test = 0.0
    fscore_test = 0.0
    rmse = 0.0
    mae = 0.0
    medae = 0.0

    Ll1 = 0
    depth_loss = 0
    ssim_loss = 0

    total_number = len(static_views)

    for idx, scene_view in enumerate(static_views):
        render_timestamp = scene_view.image_name
        world_to_camera_pose = np.eye(4)
        world_to_camera_pose[:3, :3] = np.transpose(scene_view.R)
        world_to_camera_pose[:3, 3] = scene_view.T
        camera_to_world_pose = np.linalg.inv(world_to_camera_pose)
        valid_model_info = []
        for model_id in valid_timestamp_model[render_timestamp]:
            time_pose = model_id_scene_info[model_id].time_poses[render_timestamp]
            model_to_camera_pose = time_pose.camera_pose
            object_view = time_pose.view

            model_gaussian = model_id_scene_info[model_id].gaussians
            model_gaussian.eval()
            model_to_world = camera_to_world_pose @ model_to_camera_pose
            (
                model_to_world_r,
                model_to_world_t,
                model_to_world_q,
            ) = model_world_transform_cache[model_id][render_timestamp]
            valid_model_info.append(
                ValidModeInfo(
                    model_id=model_id,
                    model_pose=model_to_world,
                    model_view=object_view,
                    model_gaussians=model_gaussian,
                    need_train=False,
                    model_world_rotation=model_to_world_r,
                    model_world_translation=model_to_world_t,
                    model_world_quaternion=model_to_world_q,
                )
            )

        render_pkg = renderComposite(
            scene_view,
            background,
            pipe,
            valid_model_info,
            model_args.max_depth,
            retain_grad=False,
        )
        image, depth = render_pkg["render"], render_pkg["depth"]

        gt_image = scene_view.original_image.cuda()
        gt_objmask = scene_view.img_mask
        ray_drop = gt_image[0:1, ...]
        gt_intensity = gt_image[1:2, ...] * ray_drop * gt_objmask
        gt_depth = gt_image[2:3, ...] * ray_drop * gt_objmask

        render_intensity = image[0:1, ...]
        render_raydrop = image[1:2, ...]
        render_raydrop_mask = torch.where(render_raydrop > 0.5, 1, 0)
        render_intensity = (
            render_intensity * render_raydrop_mask
        )  # Align with dynfl without considering raydrop
        depth = depth * render_raydrop_mask

        # Save data for UNet refine if needed
        if save_unet_refine_data:
            ray_drop_datasets_dir = os.path.join(
                model_args.model_path, "ray_drop_datasets"
            )
            gt_dir = os.path.join(ray_drop_datasets_dir, "gt")
            render_dir = os.path.join(
                ray_drop_datasets_dir,
                "render_train" if is_train_views else "render_test",
            )

            gt_save_path = os.path.join(gt_dir, f"{render_timestamp}.pt")
            # gt_image: [raydrop, intensity, depth, beam_inclinations]
            # Save as dict to include beam_inclinations for evaluation
            save_dict = {
                "gt_image": gt_image.detach().cpu(),
                "beam_inclinations": scene_view.beam_inclinations.detach().cpu(),
                "gt_objmask": gt_objmask.detach().cpu(),
            }
            torch.save(save_dict, gt_save_path)

            # Save rendered data: [raydrop, intensity, depth, feats, mid_depth_diff]
            render_save_path = os.path.join(render_dir, f"{render_timestamp}.pt")
            
            # [Add depth_distortion_aware to input channels]
            # This allows UNet to see the "filtered" suggested mask and learn to use it
            depth_distortion_aware = render_pkg["mid_depth_diff"]
            
            # We explicitly detach and move to cpu to avoid memory issues
            render_data = torch.cat(
                [
                    render_pkg["render"][1:2, ...],
                    render_pkg["render"][0:1, ...],
                    render_pkg["depth"],
                    #render_pkg["rendered_feat"],
                    depth_distortion_aware, # Add mid_depth_diff as explicit feature for Unet
                ],
                dim=0,
            )
            torch.save(render_data.detach().cpu(), render_save_path)

        # [Modified by Instruction] Apply depth distortion TRICK to depth metrics only
        # User observed that applying this to intensity hurts intensity metrics,
        # but NOT applying it to depth hurts depth metrics (CD).
        # So we split the logic: Filter Depth, Keep Intensity Raw.
        if True: 
            depth_distortion_aware = render_pkg["mid_depth_diff"]
            # Filter out points with large distortion (noise)
            depth_distortion_aware = torch.where(depth_distortion_aware < 0.3, 1.0, 0.0)
            # Apply to DEPTH 
            depth = depth * depth_distortion_aware
            # render_intensity = render_intensity * depth_distortion_aware 
        else:
            depth_distortion_aware = None

        l1_test += l1_loss(render_intensity, gt_intensity)
        psnr_test += psnr(render_intensity, gt_intensity).mean().double()
        curr_depth_loss = l1_loss(depth, gt_depth)
        depth_loss += curr_depth_loss
        ssim_loss += 1.0 - ssim(render_intensity, gt_intensity)
        points_meter = PointsMeter(
            scale=1,
            intrinsics=None,
            beam_inclinations=scene_view.beam_inclinations.detach(),
        )

        points_meter.update(depth, gt_depth, Filter=False)
        cd_fs = points_meter.measure()
        cd_test += cd_fs[0]

        error_in_abs = torch.abs(render_intensity - gt_intensity)
        in_mae += error_in_abs.mean()
        in_rmse += torch.sqrt((error_in_abs * error_in_abs).mean())
        in_medae += error_in_abs.median()

        in_ssim += structural_similarity(
            render_intensity[0].detach().cpu().numpy(),
            gt_intensity[0].detach().cpu().numpy(),
            data_range=1.0,
        )

        fscore_test += cd_fs[1]
         # 计算深度误差时，只考虑有效区域（有 raydrop 且经过 depth_distortion_aware 筛选的区域）
        valid_depth_mask = (ray_drop > 0.5) & (gt_objmask > 0.5)
        if depth_distortion_aware is not None:  # 如果使用了 depth_distortion_aware
            valid_depth_mask = valid_depth_mask & (depth_distortion_aware > 0.5)
        
        error_depth_abs = torch.abs(depth - gt_depth)
        if valid_depth_mask.sum() > 0:
            valid_error = error_depth_abs[valid_depth_mask]
            mae += valid_error.mean()
            rmse += torch.sqrt((valid_error * valid_error).mean())
            medae += valid_error.median()
        else:
            # 如果没有有效像素，记录为 0（或者你可以选择跳过这一帧）
            mae += 0
            rmse += 0
            medae += 0

    if total_number == 0:
        logger.warning(f"[ITER {iteration}] No test views in this block, skipping evaluation.")
        return
    psnr_test /= total_number
    l1_test /= total_number
    in_ssim /= total_number
    in_mae /= total_number
    in_rmse /= total_number
    in_medae /= total_number

    cd_test /= total_number
    fscore_test /= total_number
    mae /= total_number
    rmse /= total_number
    medae /= total_number

    logger.info(
        "\n[ITER {}] Evaluating: intensity: L1 {} PSNR {} SSIM {} MAE {} RMSE {} MadAE {} // depth: CD {} Fscore {} MAE {} MedAE {} RMSE {}".format(
            iteration,
            l1_test,
            psnr_test,
            in_ssim,
            in_mae,
            in_rmse,
            in_medae,
            cd_test,
            fscore_test,
            mae,
            medae,
            rmse,
        )
    )

    torch.cuda.empty_cache()
    for model_id in valid_timestamp_model[render_timestamp]:
        model_gaussian = model_id_scene_info[model_id].gaussians
        model_gaussian.train()


def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO)
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--warmup", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[1000, 3000, 5000,8000,10000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[1000, 3000, 5000,8000,10000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--test_frames_interval", type=int, default=10)
    parser.add_argument("--gpu", type=str, default="-1")
    parser.add_argument("--dataset", type=str, default="chery")
    parser.add_argument("--block_size", type=int, default=50)
    parser.add_argument("--enable_raydrop_unet", action="store_true", default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # enable logging
    dataset = args.source_path.split("/")[-1]
    os.makedirs(args.model_path, exist_ok=True)
    logger = get_logger(args.model_path)
    logger.info(f"args: {args}")
    logger.info("SourcePath " + args.source_path)
    logger.info("Optimizing " + args.model_path)

    if args.gpu != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f"using GPU {args.gpu}")

    # Initialize system state (RNG)
    safe_state(args.quiet)
    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)  #

    # multi-block render for large scene
    model_args = lp.extract(args)
    model_args.enable_raydrop_unet = args.enable_raydrop_unet

    # load dynamic info
    if args.dataset == "chery":
        from scene.chery_dataloader import Chery_Dataloader as GT_Dataloader
    elif args.dataset == "zdrive":
        from scene.zdrive_dataloader import ZDrive_Dataloader as GT_Dataloader
    elif args.dataset == "chery_lidar_360":
        from scene.adaptive_dataloader import AdaptiveDataloader as GT_Dataloader
    else:
        print("ERROR: Unsupported data format.")
        logger.info("\nUnsupported data format.")
        sys.exit(1)

    # **dataPartitionSimple**: Divide into a block every 50 frames (simple implementation)
    # **dataPartition** : Divide blocks according to scene scale （You need to adjust the parameters according to the data set）
    if (
        args.dataset == "chery"
        or args.dataset == "zdrive"
        or args.dataset == "chery_lidar_360"
    ):
        block_info_with_extend, block_info_without_extend = dataPartitionChery(
            model_args, args.block_size, single_block_test= False # 增加一个快速验证调试点
        )
    else:
        block_info_with_extend, block_info_without_extend = dataPartitionSimple(
            model_args, single_block_test= False # 开的话就只训练1block，此时可以把block从50改很低快速验证
        )

    if model_args.enable_raydrop_unet:
        # Create directories for UNet refine
        ray_drop_datasets_dir = os.path.join(args.model_path, "ray_drop_datasets")
        os.makedirs(os.path.join(ray_drop_datasets_dir, "gt"), exist_ok=True)
        os.makedirs(os.path.join(ray_drop_datasets_dir, "render_train"), exist_ok=True)
        os.makedirs(os.path.join(ray_drop_datasets_dir, "render_test"), exist_ok=True)

    for block_id, train_frame_times in block_info_with_extend.items():
        model_args.block_id = block_id  # update block id
        model_args.test_frames_interval = args.test_frames_interval
        gt_dynamic_model = GT_Dataloader(
            model_args, train=True, train_frame_times=train_frame_times
        )
        if hasattr(gt_dynamic_model, "get_instance_id_list"):
            instance_map = {
                "background_model_id": 0,
                "instances": [
                    {
                        "instance_id": int(instance_id),
                        "model_id": int(gt_dynamic_model.instance_id_to_model_id(instance_id)),
                        "raw_id": gt_dynamic_model.get_raw_id_from_instance_id(instance_id),
                    }
                    for instance_id in gt_dynamic_model.get_instance_id_list()
                ],
            }
            with open(
                os.path.join(args.model_path, f"instance_id_map_block_{block_id}.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(instance_map, f, indent=2)
        training(
            gt_dynamic_model,
            model_args,
            op.extract(args),
            pp.extract(args),
            dataset,
            args.test_iterations,
            args.save_iterations,
            args.checkpoint_iterations,
            args.start_checkpoint,
            args.debug_from,
            logger,
        )

    # All done
    logger.info("\nTraining complete.")
