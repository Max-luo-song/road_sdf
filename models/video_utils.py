from typing import Literal, Dict, List, Optional, Callable
from tqdm import tqdm, trange
import numpy as np
import os
import logging
import imageio, cv2
from omegaconf import OmegaConf

import torch
from torch import Tensor
from torch.nn import functional as F
from skimage.metrics import structural_similarity as ssim

from datasets.base import SplitWrapper
from datasets.base.pixel_source import CameraData
from models.trainers.base import BasicTrainer
from utils.visualization import (
    to8b,
    depth_visualizer,
)

# NOTE(syc): pinhole2fisheye方法存在bug，暂时改用新的distort_image方法
# from chery_tools.pinhole2fisheye.utils.pinhole2fisheye import pinhole2fisheye
from chery_tools.distort import distort_image

logger = logging.getLogger()


def get_numpy(x: Tensor) -> np.ndarray:
    return x.squeeze().cpu().numpy()


def non_zero_mean(x: Tensor) -> float:
    x = [v for v in x if np.isfinite(v)]
    return float(sum(x) / len(x) if len(x) > 0 else -1)


def compute_psnr(prediction: Tensor, target: Tensor) -> float:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between the prediction and target tensors.

    Args:
        prediction (torch.Tensor): The predicted tensor.
        target (torch.Tensor): The target tensor.

    Returns:
        float: The PSNR value between the prediction and target tensors.
    """
    if not isinstance(prediction, Tensor):
        prediction = Tensor(prediction)
    if not isinstance(target, Tensor):
        target = Tensor(target).to(prediction.device)
    return (-10 * torch.log10(F.mse_loss(prediction, target))).item()


def render_images(
    trainer: BasicTrainer,
    dataset: SplitWrapper,
    compute_metrics: bool = False,
    compute_error_map: bool = False,
    vis_indices: Optional[List[int]] = None,
    edit_cfg: OmegaConf = None,
):
    """
    Render pixel-related outputs from a model.

    Args:
        ....skip obvious args
        compute_metrics (bool, optional): Whether to compute metrics. Defaults to False.
        vis_indices (Optional[List[int]], optional): Indices to visualize. Defaults to None.
    """
    trainer.set_eval()
    render_results = render(
        dataset,
        trainer=trainer,
        compute_metrics=compute_metrics,
        compute_error_map=compute_error_map,
        vis_indices=vis_indices,
        edit_cfg=edit_cfg,
    )
    if compute_metrics:
        num_samples = len(dataset) if vis_indices is None else len(vis_indices)
        logger.info(f"Eval over {num_samples} images:")
        logger.info(f"\t Full Image  PSNR: {render_results['psnr']:.4f}")
        logger.info(f"\t Full Image  SSIM: {render_results['ssim']:.4f}")
        logger.info(f"\t Full Image LPIPS: {render_results['lpips']:.4f}")
        logger.info(
            f"\t Full Image  PSNR (no ego): {render_results['psnr_no_ego']:.4f}"
        )
        logger.info(
            f"\t Full Image  SSIM (no ego): {render_results['ssim_no_ego']:.4f}"
        )
        logger.info(
            f"\t Full Image LPIPS (no ego): {render_results['lpips_no_ego']:.4f}"
        )
        logger.info(
            f"\t Full Image  PSNR (with ego): {render_results['psnr_with_ego']:.4f}"
        )
        logger.info(
            f"\t Full Image  SSIM (with ego): {render_results['ssim_with_ego']:.4f}"
        )
        logger.info(
            f"\t Full Image LPIPS (with ego): {render_results['lpips_with_ego']:.4f}"
        )
        logger.info(f"\t     Non-Sky PSNR: {render_results['occupied_psnr']:.4f}")
        logger.info(f"\t     Non-Sky SSIM: {render_results['occupied_ssim']:.4f}")
        logger.info(f"\tDynamic-Only PSNR: {render_results['masked_psnr']:.4f}")
        logger.info(f"\tDynamic-Only SSIM: {render_results['masked_ssim']:.4f}")
        logger.info(f"\t  Human-Only PSNR: {render_results['human_psnr']:.4f}")
        logger.info(f"\t  Human-Only SSIM: {render_results['human_ssim']:.4f}")
        logger.info(f"\tVehicle-Only PSNR: {render_results['vehicle_psnr']:.4f}")
        logger.info(f"\tVehicle-Only SSIM: {render_results['vehicle_ssim']:.4f}")

    return render_results


def render_edit_rigid(
    dataset: SplitWrapper,
    trainer: BasicTrainer = None,
    compute_metrics: bool = False,
    compute_error_map: bool = False,
    edit_cfg: OmegaConf = None,
    vis_indices: Optional[List[int]] = None,
):
    """
    Renders a dataset utilizing a specified render function.

    Parameters:
        dataset: Dataset to render.
        trainer: Gaussian trainer, includes gaussian models and rendering modules
        compute_metrics: Optional; if True, the function will compute and return metrics. Default is False.
        compute_error_map: Optional; if True, the function will compute and return error maps. Default is False.
        vis_indices: Optional; if not None, the function will only render the specified indices. Default is None.
    """
    # rgbs
    rgbs, gt_rgbs, rgb_sky_blend, rgb_sky = [], [], [], []
    (
        Background_rgbs,
        RigidNodes_rgbs,
        DeformableNodes_rgbs,
        SMPLNodes_rgbs,
        Dynamic_rgbs,
    ) = ([], [], [], [], [])
    error_maps = []

    # depths
    depths, lidar_on_images = [], []
    (
        Background_depths,
        RigidNodes_depths,
        DeformableNodes_depths,
        SMPLNodes_depths,
        Dynamic_depths,
    ) = ([], [], [], [], [])

    # sky
    opacities, sky_masks = [], []
    (
        Background_opacities,
        RigidNodes_opacities,
        DeformableNodes_opacities,
        SMPLNodes_opacities,
        Dynamic_opacities,
    ) = ([], [], [], [], [])

    # misc
    cam_names, cam_ids = [], []

    if compute_metrics:
        psnrs, ssim_scores, lpipss = [], [], []
        psnrs_no_ego, ssim_scores_no_ego, lpipss_no_ego = [], [], []
        psnrs_with_ego, ssim_scores_with_ego, lpipss_with_ego = [], [], []
        masked_psnrs, masked_ssims = [], []
        human_psnrs, human_ssims = [], []
        vehicle_psnrs, vehicle_ssims = [], []
        occupied_psnrs, occupied_ssims = [], []

    with torch.no_grad():
        indices = vis_indices if vis_indices is not None else range(len(dataset))
        camera_downscale = trainer._get_downscale_factor()

        # get self trajectory
        trajectory = dataset.datasource.front_camera_trajectory
        # edit gaussians
        if 'Nodes' in edit_cfg and edit_cfg.Nodes != None:
            trainer.edit_gaussians(edit_cfg=edit_cfg, self_trajectory=trajectory, dataset=dataset)
        else:
            print("Warning: 'Nodes' key not found in the configuration or No values found in 'Nodes'.")

        for i in tqdm(indices, desc=f"rendering {dataset.split}", dynamic_ncols=True):
            # get image and camera infos
            image_infos, cam_infos = dataset.get_image(i, camera_downscale)
            for k, v in image_infos.items():
                if isinstance(v, Tensor):
                    image_infos[k] = v.cuda(non_blocking=True)
            for k, v in cam_infos.items():
                if isinstance(v, Tensor):
                    cam_infos[k] = v.cuda(non_blocking=True)
            # render the image
            results, _ = trainer(image_infos, cam_infos)

            if isinstance(results, tuple):
                results = results[0]

            # ------------- clip rgb ------------- #
            for k, v in results.items():
                if isinstance(v, Tensor) and "rgb" in k:
                    results[k] = v.clamp(0.0, 1.0)

            # ------------- cam names ------------- #
            cam_names.append(cam_infos["cam_name"])
            cam_ids.append(cam_infos["cam_id"].flatten()[0].cpu().numpy())

            # ------------- rgb ------------- #
            rgb = results["rgb"]
            ### NOTE(gls): 增加ego_mask原始
            rgb = torch.where(image_infos["egocar_masks"][..., None], image_infos["pixels"], rgb)
            rgbs.append(get_numpy(rgb))
            if "pixels" in image_infos:
                gt_rgbs.append(get_numpy(image_infos["pixels"]))

            green_background = torch.tensor([0.0, 177, 64]) / 255.0
            green_background = green_background.to(rgb.device)
            if "Background_rgb" in results:
                Background_rgb = results["Background_rgb"] * results[
                    "Background_opacity"
                ] + green_background * (1 - results["Background_opacity"])
                Background_rgbs.append(get_numpy(Background_rgb))
            if "RigidNodes_rgb" in results:
                RigidNodes_rgb = results["RigidNodes_rgb"] * results[
                    "RigidNodes_opacity"
                ] + green_background * (1 - results["RigidNodes_opacity"])
                RigidNodes_rgbs.append(get_numpy(RigidNodes_rgb))
            if "DeformableNodes_rgb" in results:
                DeformableNodes_rgb = results["DeformableNodes_rgb"] * results[
                    "DeformableNodes_opacity"
                ] + green_background * (1 - results["DeformableNodes_opacity"])
                DeformableNodes_rgbs.append(get_numpy(DeformableNodes_rgb))
            if "SMPLNodes_rgb" in results:
                SMPLNodes_rgb = results["SMPLNodes_rgb"] * results[
                    "SMPLNodes_opacity"
                ] + green_background * (1 - results["SMPLNodes_opacity"])
                SMPLNodes_rgbs.append(get_numpy(SMPLNodes_rgb))
            if "Dynamic_rgb" in results:
                Dynamic_rgb = results["Dynamic_rgb"] * results[
                    "Dynamic_opacity"
                ] + green_background * (1 - results["Dynamic_opacity"])
                Dynamic_rgbs.append(get_numpy(Dynamic_rgb))
            if compute_error_map:
                # cal mean squared error
                error_map = (rgb - image_infos["pixels"]) ** 2
                error_map = error_map.mean(dim=-1, keepdim=True)
                # scale
                error_map = (error_map - error_map.min()) / (
                    error_map.max() - error_map.min()
                )
                error_map = error_map.repeat_interleave(3, dim=-1)
                error_maps.append(get_numpy(error_map))
            if "rgb_sky_blend" in results:
                rgb_sky_blend.append(get_numpy(results["rgb_sky_blend"]))
            if "rgb_sky" in results:
                rgb_sky.append(get_numpy(results["rgb_sky"]))
            # ------------- depth ------------- #
            depth = results["depth"]
            depths.append(get_numpy(depth))
            # ------------- mask ------------- #
            if "opacity" in results:
                opacities.append(get_numpy(results["opacity"]))
            if "Background_depth" in results:
                Background_depths.append(get_numpy(results["Background_depth"]))
                Background_opacities.append(get_numpy(results["Background_opacity"]))
            if "RigidNodes_depth" in results:
                RigidNodes_depths.append(get_numpy(results["RigidNodes_depth"]))
                RigidNodes_opacities.append(get_numpy(results["RigidNodes_opacity"]))
            if "DeformableNodes_depth" in results:
                DeformableNodes_depths.append(
                    get_numpy(results["DeformableNodes_depth"])
                )
                DeformableNodes_opacities.append(
                    get_numpy(results["DeformableNodes_opacity"])
                )
            if "SMPLNodes_depth" in results:
                SMPLNodes_depths.append(get_numpy(results["SMPLNodes_depth"]))
                SMPLNodes_opacities.append(get_numpy(results["SMPLNodes_opacity"]))
            if "Dynamic_depth" in results:
                Dynamic_depths.append(get_numpy(results["Dynamic_depth"]))
                Dynamic_opacities.append(get_numpy(results["Dynamic_opacity"]))
            if "sky_masks" in image_infos:
                sky_masks.append(get_numpy(image_infos["sky_masks"]))

            # ------------- lidar ------------- #
            if "lidar_depth_map" in image_infos:
                depth_map = image_infos["lidar_depth_map"]
                depth_img = depth_map.cpu().numpy()
                depth_img = depth_visualizer(depth_img, depth_img > 0)
                mask = (depth_map.unsqueeze(-1) > 0).cpu().numpy()
                lidar_on_image = np.where(mask, depth_img, image_infos["pixels"].cpu().numpy())
                lidar_on_images.append(lidar_on_image)

            if compute_metrics:
                psnr = compute_psnr(rgb, image_infos["pixels"])
                ssim_score = ssim(
                    get_numpy(rgb),
                    get_numpy(image_infos["pixels"]),
                    data_range=1.0,
                    channel_axis=-1,
                )
                lpips = trainer.lpips(
                    rgb[None, ...].permute(0, 3, 1, 2),
                    image_infos["pixels"][None, ...].permute(0, 3, 1, 2),
                )
                logger.info(f"Frame {i}: PSNR {psnr:.4f}, SSIM {ssim_score:.4f}")
                psnrs.append(psnr)
                ssim_scores.append(ssim_score)
                lpipss.append(lpips.item())

                # 不计入 ego 影响的指标 (syc)
                if "egocar_masks" in image_infos:
                    ego_mask = get_numpy(image_infos["egocar_masks"])

                    # 去除车身区域 >>>
                    mask_no_ego = ~ego_mask

                    psnr_no_ego = compute_psnr(
                        rgb[mask_no_ego], image_infos["pixels"][mask_no_ego]
                    )
                    ssim_score_no_ego = ssim(
                        get_numpy(rgb),
                        get_numpy(image_infos["pixels"]),
                        data_range=1.0,
                        channel_axis=-1,
                        full=True,
                    )[1][mask_no_ego].mean()

                    rgb_masked = rgb.clone()
                    pixels_masked = image_infos["pixels"].clone()
                    rgb_masked[ego_mask] = 0
                    pixels_masked[ego_mask] = 0
                    lpips_no_ego = trainer.lpips(
                        rgb_masked[None, ...].permute(0, 3, 1, 2),
                        pixels_masked[None, ...].permute(0, 3, 1, 2),
                    )
                    # <<<

                    # 加上车身 GT >>>
                    rgb_with_ego = rgb.clone()
                    rgb_with_ego[ego_mask] = image_infos["pixels"][ego_mask]

                    psnr_with_ego = compute_psnr(rgb_with_ego, image_infos["pixels"])
                    ssim_score_with_ego = ssim(
                        get_numpy(rgb_with_ego),
                        get_numpy(image_infos["pixels"]),
                        data_range=1.0,
                        channel_axis=-1,
                    )
                    lpips_with_ego = trainer.lpips(
                        rgb_with_ego[None, ...].permute(0, 3, 1, 2),
                        image_infos["pixels"][None, ...].permute(0, 3, 1, 2),
                    )
                    # <<<

                    psnrs_no_ego.append(psnr_no_ego)
                    ssim_scores_no_ego.append(ssim_score_no_ego)
                    lpipss_no_ego.append(lpips_no_ego.item())
                    logger.info(
                        f"Frame {i}: PSNR (no ego) {psnr_no_ego:.4f}, SSIM (no ego) {ssim_score_no_ego:.4f}"
                    )

                    psnrs_with_ego.append(psnr_with_ego)
                    ssim_scores_with_ego.append(ssim_score_with_ego)
                    lpipss_with_ego.append(lpips_with_ego.item())
                    logger.info(
                        f"Frame {i}: PSNR (with ego) {psnr_with_ego:.4f}, SSIM (with ego) {ssim_score_with_ego:.4f}"
                    )

                if "sky_masks" in image_infos:
                    occupied_mask = ~get_numpy(image_infos["sky_masks"])
                    if occupied_mask.sum() > 0:
                        occupied_psnrs.append(
                            compute_psnr(
                                rgb[occupied_mask], image_infos["pixels"][occupied_mask]
                            )
                        )
                        occupied_ssims.append(
                            ssim(
                                get_numpy(rgb),
                                get_numpy(image_infos["pixels"]),
                                data_range=1.0,
                                channel_axis=-1,
                                full=True,
                            )[1][occupied_mask].mean()
                        )

                if "dynamic_masks" in image_infos:
                    dynamic_mask = get_numpy(image_infos["dynamic_masks"])
                    if dynamic_mask.sum() > 0:
                        masked_psnrs.append(
                            compute_psnr(
                                rgb[dynamic_mask], image_infos["pixels"][dynamic_mask]
                            )
                        )
                        masked_ssims.append(
                            ssim(
                                get_numpy(rgb),
                                get_numpy(image_infos["pixels"]),
                                data_range=1.0,
                                channel_axis=-1,
                                full=True,
                            )[1][dynamic_mask].mean()
                        )

                if "human_masks" in image_infos:
                    human_mask = get_numpy(image_infos["human_masks"])
                    if human_mask.sum() > 0:
                        human_psnrs.append(
                            compute_psnr(
                                rgb[human_mask], image_infos["pixels"][human_mask]
                            )
                        )
                        human_ssims.append(
                            ssim(
                                get_numpy(rgb),
                                get_numpy(image_infos["pixels"]),
                                data_range=1.0,
                                channel_axis=-1,
                                full=True,
                            )[1][human_mask].mean()
                        )

                if "vehicle_masks" in image_infos:
                    vehicle_mask = get_numpy(image_infos["vehicle_masks"])
                    if vehicle_mask.sum() > 0:
                        vehicle_psnrs.append(
                            compute_psnr(
                                rgb[vehicle_mask], image_infos["pixels"][vehicle_mask]
                            )
                        )
                        vehicle_ssims.append(
                            ssim(
                                get_numpy(rgb),
                                get_numpy(image_infos["pixels"]),
                                data_range=1.0,
                                channel_axis=-1,
                                full=True,
                            )[1][vehicle_mask].mean()
                        )

    # messy aggregation...
    results_dict = {}
    results_dict["psnr"] = non_zero_mean(psnrs) if compute_metrics else -1
    results_dict["ssim"] = non_zero_mean(ssim_scores) if compute_metrics else -1
    results_dict["lpips"] = non_zero_mean(lpipss) if compute_metrics else -1
    # syc >>>
    results_dict["psnr_no_ego"] = non_zero_mean(psnrs_no_ego) if compute_metrics else -1
    results_dict["ssim_no_ego"] = (
        non_zero_mean(ssim_scores_no_ego) if compute_metrics else -1
    )
    results_dict["lpips_no_ego"] = (
        non_zero_mean(lpipss_no_ego) if compute_metrics else -1
    )
    results_dict["psnr_with_ego"] = (
        non_zero_mean(psnrs_with_ego) if compute_metrics else -1
    )
    results_dict["ssim_with_ego"] = (
        non_zero_mean(ssim_scores_with_ego) if compute_metrics else -1
    )
    results_dict["lpips_with_ego"] = (
        non_zero_mean(lpipss_with_ego) if compute_metrics else -1
    )
    # <<<
    results_dict["occupied_psnr"] = (
        non_zero_mean(occupied_psnrs) if compute_metrics else -1
    )
    results_dict["occupied_ssim"] = (
        non_zero_mean(occupied_ssims) if compute_metrics else -1
    )
    results_dict["masked_psnr"] = non_zero_mean(masked_psnrs) if compute_metrics else -1
    results_dict["masked_ssim"] = non_zero_mean(masked_ssims) if compute_metrics else -1
    results_dict["human_psnr"] = non_zero_mean(human_psnrs) if compute_metrics else -1
    results_dict["human_ssim"] = non_zero_mean(human_ssims) if compute_metrics else -1
    results_dict["vehicle_psnr"] = (
        non_zero_mean(vehicle_psnrs) if compute_metrics else -1
    )
    results_dict["vehicle_ssim"] = (
        non_zero_mean(vehicle_ssims) if compute_metrics else -1
    )
    results_dict["rgbs"] = rgbs
    results_dict["depths"] = depths
    results_dict["cam_names"] = cam_names
    results_dict["cam_ids"] = cam_ids
    if len(opacities) > 0:
        results_dict["opacities"] = opacities
    if len(gt_rgbs) > 0:
        results_dict["gt_rgbs"] = gt_rgbs
    if len(error_maps) > 0:
        results_dict["rgb_error_maps"] = error_maps
    if len(rgb_sky_blend) > 0:
        results_dict["rgb_sky_blend"] = rgb_sky_blend
    if len(rgb_sky) > 0:
        results_dict["rgb_sky"] = rgb_sky
    if len(sky_masks) > 0:
        results_dict["gt_sky_masks"] = sky_masks
    if len(lidar_on_images) > 0:
        results_dict["lidar_on_images"] = lidar_on_images
    if len(Background_rgbs) > 0:
        results_dict["Background_rgbs"] = Background_rgbs
    if len(RigidNodes_rgbs) > 0:
        results_dict["RigidNodes_rgbs"] = RigidNodes_rgbs
    if len(DeformableNodes_rgbs) > 0:
        results_dict["DeformableNodes_rgbs"] = DeformableNodes_rgbs
    if len(SMPLNodes_rgbs) > 0:
        results_dict["SMPLNodes_rgbs"] = SMPLNodes_rgbs
    if len(Dynamic_rgbs) > 0:
        results_dict["Dynamic_rgbs"] = Dynamic_rgbs
    if len(Background_depths) > 0:
        results_dict["Background_depths"] = Background_depths
    if len(RigidNodes_depths) > 0:
        results_dict["RigidNodes_depths"] = RigidNodes_depths
    if len(DeformableNodes_depths) > 0:
        results_dict["DeformableNodes_depths"] = DeformableNodes_depths
    if len(SMPLNodes_depths) > 0:
        results_dict["SMPLNodes_depths"] = SMPLNodes_depths
    if len(Dynamic_depths) > 0:
        results_dict["Dynamic_depths"] = Dynamic_depths
    if len(Background_opacities) > 0:
        results_dict["Background_opacities"] = Background_opacities
    if len(RigidNodes_opacities) > 0:
        results_dict["RigidNodes_opacities"] = RigidNodes_opacities
    if len(DeformableNodes_opacities) > 0:
        results_dict["DeformableNodes_opacities"] = DeformableNodes_opacities
    if len(SMPLNodes_opacities) > 0:
        results_dict["SMPLNodes_opacities"] = SMPLNodes_opacities
    if len(Dynamic_opacities) > 0:
        results_dict["Dynamic_opacities"] = Dynamic_opacities
    return results_dict


def render_legend(
    dataset: SplitWrapper,
    trainer: BasicTrainer = None,
    compute_error_map: bool = False,
    edit_cfg: OmegaConf = None,
    vis_indices: Optional[List[int]] = None,
    image_output_pth: Optional[str] = None,
):
    """
    Renders a dataset utilizing a specified render function.

    Parameters:
        dataset: Dataset to render.
        trainer: Gaussian trainer, includes gaussian models and rendering modules
        compute_metrics: Optional; if True, the function will compute and return metrics. Default is False.
        compute_error_map: Optional; if True, the function will compute and return error maps. Default is False.
        vis_indices: Optional; if not None, the function will only render the specified indices. Default is None.
    """
    # rgbs
    rgbs, gt_rgbs, rgb_sky_blend, rgb_sky = [], [], [], []
    (
        Background_rgbs,
        RigidNodes_rgbs,
        DeformableNodes_rgbs,
        SMPLNodes_rgbs,
        Dynamic_rgbs,
    ) = ([], [], [], [], [])
    error_maps = []

    # depths
    depths, lidar_on_images = [], []
    (
        Background_depths,
        RigidNodes_depths,
        DeformableNodes_depths,
        SMPLNodes_depths,
        Dynamic_depths,
    ) = ([], [], [], [], [])

    # sky
    opacities, sky_masks = [], []
    (
        Background_opacities,
        RigidNodes_opacities,
        DeformableNodes_opacities,
        SMPLNodes_opacities,
        Dynamic_opacities,
    ) = ([], [], [], [], [])

    # misc
    cam_names, cam_ids = [], []

    with torch.no_grad():
        indices = vis_indices if vis_indices is not None else range(len(dataset))
        camera_downscale = trainer._get_downscale_factor()

        for i in tqdm(indices, desc=f"rendering {dataset.split}", dynamic_ncols=True):
            # get image and camera infos
            image_infos, cam_infos = dataset.get_image(i, camera_downscale)
            for k, v in image_infos.items():
                if isinstance(v, Tensor):
                    image_infos[k] = v.cuda(non_blocking=True)
            for k, v in cam_infos.items():
                if isinstance(v, Tensor):
                    cam_infos[k] = v.cuda(non_blocking=True)
            # render the image
            results, _ = trainer(
                image_infos,
                cam_infos,
            )

            # ------------- clip rgb ------------- #
            for k, v in results.items():
                if isinstance(v, Tensor) and "rgb" in k:
                    results[k] = v.clamp(0.0, 1.0)

            # ------------- cam names ------------- #
            cam_names.append(cam_infos["cam_name"])
            cam_ids.append(cam_infos["cam_id"].flatten()[0].cpu().numpy())

            # ------------- rgb ------------- #
            rgb = results["rgb"]
            rgbs.append(get_numpy(rgb))
            if "pixels" in image_infos:
                gt_rgbs.append(get_numpy(image_infos["pixels"]))

            green_background = torch.tensor([0.0, 177, 64]) / 255.0
            green_background = green_background.to(rgb.device)
            if "Background_rgb" in results:
                Background_rgb = results["Background_rgb"] * results[
                    "Background_opacity"
                ] + green_background * (1 - results["Background_opacity"])
                Background_rgbs.append(get_numpy(Background_rgb))
            if "RigidNodes_rgb" in results:
                RigidNodes_rgb = results["RigidNodes_rgb"] * results[
                    "RigidNodes_opacity"
                ] + green_background * (1 - results["RigidNodes_opacity"])
                RigidNodes_rgbs.append(get_numpy(RigidNodes_rgb))
            if "DeformableNodes_rgb" in results:
                DeformableNodes_rgb = results["DeformableNodes_rgb"] * results[
                    "DeformableNodes_opacity"
                ] + green_background * (1 - results["DeformableNodes_opacity"])
                DeformableNodes_rgbs.append(get_numpy(DeformableNodes_rgb))
            if "SMPLNodes_rgb" in results:
                SMPLNodes_rgb = results["SMPLNodes_rgb"] * results[
                    "SMPLNodes_opacity"
                ] + green_background * (1 - results["SMPLNodes_opacity"])
                SMPLNodes_rgbs.append(get_numpy(SMPLNodes_rgb))
            if "Dynamic_rgb" in results:
                Dynamic_rgb = results["Dynamic_rgb"] * results[
                    "Dynamic_opacity"
                ] + green_background * (1 - results["Dynamic_opacity"])
                Dynamic_rgbs.append(get_numpy(Dynamic_rgb))
            if compute_error_map:
                # cal mean squared error
                error_map = (rgb - image_infos["pixels"]) ** 2
                error_map = error_map.mean(dim=-1, keepdim=True)
                # scale
                error_map = (error_map - error_map.min()) / (
                    error_map.max() - error_map.min()
                )
                error_map = error_map.repeat_interleave(3, dim=-1)
                error_maps.append(get_numpy(error_map))
            if "rgb_sky_blend" in results:
                rgb_sky_blend.append(get_numpy(results["rgb_sky_blend"]))
            if "rgb_sky" in results:
                rgb_sky.append(get_numpy(results["rgb_sky"]))
            # ------------- depth ------------- #
            depth = results["depth"]
            depths.append(get_numpy(depth))
            # ------------- mask ------------- #
            if "opacity" in results:
                opacities.append(get_numpy(results["opacity"]))
            if "Background_depth" in results:
                Background_depths.append(get_numpy(results["Background_depth"]))
                Background_opacities.append(get_numpy(results["Background_opacity"]))
            if "RigidNodes_depth" in results:
                RigidNodes_depths.append(get_numpy(results["RigidNodes_depth"]))
                RigidNodes_opacities.append(get_numpy(results["RigidNodes_opacity"]))
            if "DeformableNodes_depth" in results:
                DeformableNodes_depths.append(
                    get_numpy(results["DeformableNodes_depth"])
                )
                DeformableNodes_opacities.append(
                    get_numpy(results["DeformableNodes_opacity"])
                )
            if "SMPLNodes_depth" in results:
                SMPLNodes_depths.append(get_numpy(results["SMPLNodes_depth"]))
                SMPLNodes_opacities.append(get_numpy(results["SMPLNodes_opacity"]))
            if "Dynamic_depth" in results:
                Dynamic_depths.append(get_numpy(results["Dynamic_depth"]))
                Dynamic_opacities.append(get_numpy(results["Dynamic_opacity"]))
            if "sky_masks" in image_infos:
                sky_masks.append(get_numpy(image_infos["sky_masks"]))

            # ------------- lidar ------------- #
            if "lidar_depth_map" in image_infos:
                depth_map = image_infos["lidar_depth_map"]
                depth_img = depth_map.cpu().numpy()
                depth_img = depth_visualizer(depth_img, depth_img > 0)
                mask = (depth_map.unsqueeze(-1) > 0).cpu().numpy()
                lidar_on_image = np.where(mask, depth_img, image_infos["pixels"].cpu().numpy())
                lidar_on_images.append(lidar_on_image)

    # messy aggregation...
    results_dict = {}
    results_dict["rgbs"] = rgbs
    results_dict["depths"] = depths
    results_dict["cam_names"] = cam_names
    results_dict["cam_ids"] = cam_ids
    if len(opacities) > 0:
        results_dict["opacities"] = opacities
    if len(gt_rgbs) > 0:
        results_dict["gt_rgbs"] = gt_rgbs
    if len(error_maps) > 0:
        results_dict["rgb_error_maps"] = error_maps
    if len(rgb_sky_blend) > 0:
        results_dict["rgb_sky_blend"] = rgb_sky_blend
    if len(rgb_sky) > 0:
        results_dict["rgb_sky"] = rgb_sky
    if len(sky_masks) > 0:
        results_dict["gt_sky_masks"] = sky_masks
    if len(lidar_on_images) > 0:
        results_dict["lidar_on_images"] = lidar_on_images
    if len(Background_rgbs) > 0:
        results_dict["Background_rgbs"] = Background_rgbs
    if len(RigidNodes_rgbs) > 0:
        results_dict["RigidNodes_rgbs"] = RigidNodes_rgbs
    if len(DeformableNodes_rgbs) > 0:
        results_dict["DeformableNodes_rgbs"] = DeformableNodes_rgbs
    if len(SMPLNodes_rgbs) > 0:
        results_dict["SMPLNodes_rgbs"] = SMPLNodes_rgbs
    if len(Dynamic_rgbs) > 0:
        results_dict["Dynamic_rgbs"] = Dynamic_rgbs
    if len(Background_depths) > 0:
        results_dict["Background_depths"] = Background_depths
    if len(RigidNodes_depths) > 0:
        results_dict["RigidNodes_depths"] = RigidNodes_depths
    if len(DeformableNodes_depths) > 0:
        results_dict["DeformableNodes_depths"] = DeformableNodes_depths
    if len(SMPLNodes_depths) > 0:
        results_dict["SMPLNodes_depths"] = SMPLNodes_depths
    if len(Dynamic_depths) > 0:
        results_dict["Dynamic_depths"] = Dynamic_depths
    if len(Background_opacities) > 0:
        results_dict["Background_opacities"] = Background_opacities
    if len(RigidNodes_opacities) > 0:
        results_dict["RigidNodes_opacities"] = RigidNodes_opacities
    if len(DeformableNodes_opacities) > 0:
        results_dict["DeformableNodes_opacities"] = DeformableNodes_opacities
    if len(SMPLNodes_opacities) > 0:
        results_dict["SMPLNodes_opacities"] = SMPLNodes_opacities
    if len(Dynamic_opacities) > 0:
        results_dict["Dynamic_opacities"] = Dynamic_opacities
    return results_dict


def render(
    dataset: SplitWrapper,
    trainer: BasicTrainer = None,
    compute_metrics: bool = False,
    compute_error_map: bool = False,
    vis_indices: Optional[List[int]] = None,
    edit_cfg: OmegaConf = None,
):
    """
    Renders a dataset utilizing a specified render function.

    Parameters:
        dataset: Dataset to render.
        trainer: Gaussian trainer, includes gaussian models and rendering modules
        compute_metrics: Optional; if True, the function will compute and return metrics. Default is False.
        compute_error_map: Optional; if True, the function will compute and return error maps. Default is False.
        vis_indices: Optional; if not None, the function will only render the specified indices. Default is None.
    """
    # rgbs
    rgbs, gt_rgbs, rgb_sky_blend, rgb_sky = [], [], [], []
    (
        Background_rgbs,
        RigidNodes_rgbs,
        DeformableNodes_rgbs,
        SMPLNodes_rgbs,
        Dynamic_rgbs,
    ) = ([], [], [], [], [])
    road_rgbs, gt_road_rgbs = [], []
    error_maps = []

    # depths
    depths, lidar_on_images = [], []
    (
        Background_depths,
        RigidNodes_depths,
        DeformableNodes_depths,
        SMPLNodes_depths,
        Dynamic_depths,
    ) = ([], [], [], [], [])

    # sky
    opacities, sky_masks = [], []
    (
        Background_opacities,
        RigidNodes_opacities,
        DeformableNodes_opacities,
        SMPLNodes_opacities,
        Dynamic_opacities,
    ) = ([], [], [], [], [])

    # misc
    cam_names, cam_ids = [], []

    if compute_metrics:
        psnrs, ssim_scores, lpipss = [], [], []
        psnrs_no_ego, ssim_scores_no_ego, lpipss_no_ego = [], [], []
        psnrs_with_ego, ssim_scores_with_ego, lpipss_with_ego = [], [], []
        masked_psnrs, masked_ssims = [], []
        human_psnrs, human_ssims = [], []
        vehicle_psnrs, vehicle_ssims = [], []
        occupied_psnrs, occupied_ssims = [], []
        road_psnrs, road_ssims = [], []

    with torch.no_grad():
        indices = vis_indices if vis_indices is not None else range(len(dataset))
        camera_downscale = trainer._get_downscale_factor()
        # Initialize movement matrix and pose
        movement = torch.eye(4)
        prev_pose = None  
        is_new_batch = True
        for i in tqdm(indices, desc=f"rendering {dataset.split}", dynamic_ncols=True):
            # get image and camera infos
            image_infos, cam_infos = dataset.get_image(i, camera_downscale)
            
            for k, v in image_infos.items():
                if isinstance(v, Tensor):
                    image_infos[k] = v.cuda(non_blocking=True)
            for k, v in cam_infos.items():
                if isinstance(v, Tensor):
                    cam_infos[k] = v.cuda(non_blocking=True)
            # render the image
            if edit_cfg is not None and edit_cfg.weather_type !=  "":
                # Initialize defaults
                front_cam = (i % 3 == 0)
                first_frame = True
                if i % 3 == 0:
                    process_cam = trainer.process_camera(cam_infos, image_infos["img_idx"].flatten()[0]).camtoworlds
                    if i != 0:
                        prev_image_infos, prev_cam_infos = dataset.get_image(i-1, camera_downscale)
                        prev_pose = trainer.process_camera(cam_infos, image_infos["img_idx"].flatten()[0]).camtoworlds
                        movement[:3, 3] = process_cam[:3, 3] - prev_pose[:3, 3]
                        movement[:3, :3] = prev_pose[:3, :3].T @ process_cam[:3, :3]
                        first_frame = False
                results, _ = trainer(image_infos, cam_infos, movement=movement, front_cam=front_cam, first_frame=first_frame, edit_cfg=edit_cfg)
            else:
                results, _ = trainer(image_infos, cam_infos)

            is_new_batch = False

            # ------------- clip rgb ------------- #
            for k, v in results.items():
                if isinstance(v, Tensor) and "rgb" in k:
                    results[k] = v.clamp(0.0, 1.0)

            # ------------- cam names ------------- #
            cam_names.append(cam_infos["cam_name"])
            cam_ids.append(cam_infos["cam_id"].flatten()[0].cpu().numpy())

            # ------------- rgb ------------- #
            rgb = results["rgb"]
            ### NOTE(gls): 增加ego_mask原始
            rgb = torch.where(image_infos["egocar_masks"][..., None], image_infos["pixels"], rgb)
            rgbs.append(get_numpy(rgb))
            if "pixels" in image_infos:
                gt_rgbs.append(get_numpy(image_infos["pixels"]))


            if "road_masks" in image_infos:
                if "egocar_masks" in image_infos:
                    # in the case of egocar, we need to mask out the egocar region
                    valid_loss_mask = ~image_infos["egocar_masks"]
                else:
                    valid_loss_mask = torch.ones_like(image_infos["sky_masks"], dtype=torch.bool)
                road_mask = image_infos["road_masks"]

                gt_road_rgb = image_infos["pixels"] * valid_loss_mask[..., None] * road_mask[..., None]
                gt_road_rgbs.append(get_numpy(gt_road_rgb))
                road_rgb = results["road_rgb"]
                road_rgb = road_rgb * valid_loss_mask[..., None]
                road_rgbs.append(get_numpy(road_rgb))

            green_background = torch.tensor([0.0, 177, 64]) / 255.0
            green_background = green_background.to(rgb.device)
            if "Background_rgb" in results:
                Background_rgb = results["Background_rgb"] * results[
                    "Background_opacity"
                ] + green_background * (1 - results["Background_opacity"])
                ### NOTE(gls): 增加ego_mask原始
                Background_rgb = torch.where(image_infos["egocar_masks"][..., None], image_infos["pixels"], Background_rgb)
                Background_rgbs.append(get_numpy(Background_rgb))
            if "RigidNodes_rgb" in results:
                RigidNodes_rgb = results["RigidNodes_rgb"] * results[
                    "RigidNodes_opacity"
                ] + green_background * (1 - results["RigidNodes_opacity"])
                ### NOTE(gls): 增加ego_mask原始
                RigidNodes_rgb = torch.where(image_infos["egocar_masks"][..., None], image_infos["pixels"], RigidNodes_rgb)
                RigidNodes_rgbs.append(get_numpy(RigidNodes_rgb))
            if "DeformableNodes_rgb" in results:
                DeformableNodes_rgb = results["DeformableNodes_rgb"] * results[
                    "DeformableNodes_opacity"
                ] + green_background * (1 - results["DeformableNodes_opacity"])
                ### NOTE(gls): 增加ego_mask原始
                DeformableNodes_rgb = torch.where(image_infos["egocar_masks"][..., None], image_infos["pixels"], DeformableNodes_rgb)
                DeformableNodes_rgbs.append(get_numpy(DeformableNodes_rgb))
            if "SMPLNodes_rgb" in results:
                SMPLNodes_rgb = results["SMPLNodes_rgb"] * results[
                    "SMPLNodes_opacity"
                ] + green_background * (1 - results["SMPLNodes_opacity"])
                ### NOTE(gls): 增加ego_mask原始
                SMPLNodes_rgb = torch.where(image_infos["egocar_masks"][..., None], image_infos["pixels"], SMPLNodes_rgb)
                SMPLNodes_rgbs.append(get_numpy(SMPLNodes_rgb))
            if "Dynamic_rgb" in results:
                Dynamic_rgb = results["Dynamic_rgb"] * results[
                    "Dynamic_opacity"
                ] + green_background * (1 - results["Dynamic_opacity"])
                ### NOTE(gls): 增加ego_mask原始
                Dynamic_rgb = torch.where(image_infos["egocar_masks"][..., None], image_infos["pixels"], Dynamic_rgb)
                Dynamic_rgbs.append(get_numpy(Dynamic_rgb))
            if compute_error_map:
                # cal mean squared error
                error_map = (rgb - image_infos["pixels"]) ** 2
                error_map = error_map.mean(dim=-1, keepdim=True)
                # scale
                error_map = (error_map - error_map.min()) / (
                    error_map.max() - error_map.min()
                )
                error_map = error_map.repeat_interleave(3, dim=-1)
                error_maps.append(get_numpy(error_map))
            if "rgb_sky_blend" in results:
                rgb_sky_blend.append(get_numpy(results["rgb_sky_blend"]))
            if "rgb_sky" in results:
                rgb_sky.append(get_numpy(results["rgb_sky"]))
            # ------------- depth ------------- #
            depth = results["depth"]
            ### NOTE(gls): 增加ego_mask区域，深度设置为极端颜色
            depth = torch.where(image_infos["egocar_masks"][..., None], 200, depth)
            depths.append(get_numpy(depth))
            # ------------- mask ------------- #
            if "opacity" in results:
                opacities.append(get_numpy(results["opacity"]))
            if "Background_depth" in results:
                Background_depths.append(get_numpy(results["Background_depth"]))
                Background_opacities.append(get_numpy(results["Background_opacity"]))
            if "RigidNodes_depth" in results:
                RigidNodes_depths.append(get_numpy(results["RigidNodes_depth"]))
                RigidNodes_opacities.append(get_numpy(results["RigidNodes_opacity"]))
            if "DeformableNodes_depth" in results:
                DeformableNodes_depths.append(
                    get_numpy(results["DeformableNodes_depth"])
                )
                DeformableNodes_opacities.append(
                    get_numpy(results["DeformableNodes_opacity"])
                )
            if "SMPLNodes_depth" in results:
                SMPLNodes_depths.append(get_numpy(results["SMPLNodes_depth"]))
                SMPLNodes_opacities.append(get_numpy(results["SMPLNodes_opacity"]))
            if "Dynamic_depth" in results:
                Dynamic_depths.append(get_numpy(results["Dynamic_depth"]))
                Dynamic_opacities.append(get_numpy(results["Dynamic_opacity"]))
            if "sky_masks" in image_infos:
                sky_masks.append(get_numpy(image_infos["sky_masks"]))

            # ------------- lidar ------------- #
            if "lidar_depth_map" in image_infos:
                depth_map = image_infos["lidar_depth_map"]
                depth_img = depth_map.cpu().numpy()
                depth_img = depth_visualizer(depth_img, depth_img > 0)
                mask = (depth_map.unsqueeze(-1) > 0).cpu().numpy()
                lidar_on_image = np.where(mask, depth_img, image_infos["pixels"].cpu().numpy())
                lidar_on_images.append(lidar_on_image)

            if compute_metrics:
                psnr = compute_psnr(rgb, image_infos["pixels"])
                ssim_score = ssim(
                    get_numpy(rgb),
                    get_numpy(image_infos["pixels"]),
                    data_range=1.0,
                    channel_axis=-1,
                )
                lpips = trainer.lpips(
                    rgb[None, ...].permute(0, 3, 1, 2),
                    image_infos["pixels"][None, ...].permute(0, 3, 1, 2),
                )
                logger.info(f"Frame {i}: PSNR {psnr:.4f}, SSIM {ssim_score:.4f}")
                psnrs.append(psnr)
                ssim_scores.append(ssim_score)
                lpipss.append(lpips.item())

                # 不计入 ego 影响的指标 (syc)
                if "egocar_masks" in image_infos:
                    ego_mask = get_numpy(image_infos["egocar_masks"])

                    # 去除车身区域 >>>
                    mask_no_ego = ~ego_mask

                    psnr_no_ego = compute_psnr(
                        rgb[mask_no_ego], image_infos["pixels"][mask_no_ego]
                    )
                    ssim_score_no_ego = ssim(
                        get_numpy(rgb),
                        get_numpy(image_infos["pixels"]),
                        data_range=1.0,
                        channel_axis=-1,
                        full=True,
                    )[1][mask_no_ego].mean()

                    rgb_masked = rgb.clone()
                    pixels_masked = image_infos["pixels"].clone()
                    rgb_masked[ego_mask] = 0
                    pixels_masked[ego_mask] = 0
                    lpips_no_ego = trainer.lpips(
                        rgb_masked[None, ...].permute(0, 3, 1, 2),
                        pixels_masked[None, ...].permute(0, 3, 1, 2),
                    )
                    # <<<

                    # 加上车身 GT >>>
                    rgb_with_ego = rgb.clone()
                    rgb_with_ego[ego_mask] = image_infos["pixels"][ego_mask]

                    psnr_with_ego = compute_psnr(rgb_with_ego, image_infos["pixels"])
                    ssim_score_with_ego = ssim(
                        get_numpy(rgb_with_ego),
                        get_numpy(image_infos["pixels"]),
                        data_range=1.0,
                        channel_axis=-1,
                    )
                    lpips_with_ego = trainer.lpips(
                        rgb_with_ego[None, ...].permute(0, 3, 1, 2),
                        image_infos["pixels"][None, ...].permute(0, 3, 1, 2),
                    )
                    # <<<

                    psnrs_no_ego.append(psnr_no_ego)
                    ssim_scores_no_ego.append(ssim_score_no_ego)
                    lpipss_no_ego.append(lpips_no_ego.item())
                    logger.info(
                        f"Frame {i}: PSNR (no ego) {psnr_no_ego:.4f}, SSIM (no ego) {ssim_score_no_ego:.4f}"
                    )

                    psnrs_with_ego.append(psnr_with_ego)
                    ssim_scores_with_ego.append(ssim_score_with_ego)
                    lpipss_with_ego.append(lpips_with_ego.item())
                    logger.info(
                        f"Frame {i}: PSNR (with ego) {psnr_with_ego:.4f}, SSIM (with ego) {ssim_score_with_ego:.4f}"
                    )

                if "sky_masks" in image_infos:
                    occupied_mask = ~get_numpy(image_infos["sky_masks"])
                    if occupied_mask.sum() > 0:
                        occupied_psnrs.append(
                            compute_psnr(
                                rgb[occupied_mask], image_infos["pixels"][occupied_mask]
                            )
                        )
                        occupied_ssims.append(
                            ssim(
                                get_numpy(rgb),
                                get_numpy(image_infos["pixels"]),
                                data_range=1.0,
                                channel_axis=-1,
                                full=True,
                            )[1][occupied_mask].mean()
                        )

                if "dynamic_masks" in image_infos:
                    dynamic_mask = get_numpy(image_infos["dynamic_masks"])
                    if dynamic_mask.sum() > 0:
                        masked_psnrs.append(
                            compute_psnr(
                                rgb[dynamic_mask], image_infos["pixels"][dynamic_mask]
                            )
                        )
                        masked_ssims.append(
                            ssim(
                                get_numpy(rgb),
                                get_numpy(image_infos["pixels"]),
                                data_range=1.0,
                                channel_axis=-1,
                                full=True,
                            )[1][dynamic_mask].mean()
                        )

                if "human_masks" in image_infos:
                    human_mask = get_numpy(image_infos["human_masks"])
                    if human_mask.sum() > 0:
                        human_psnrs.append(
                            compute_psnr(
                                rgb[human_mask], image_infos["pixels"][human_mask]
                            )
                        )
                        human_ssims.append(
                            ssim(
                                get_numpy(rgb),
                                get_numpy(image_infos["pixels"]),
                                data_range=1.0,
                                channel_axis=-1,
                                full=True,
                            )[1][human_mask].mean()
                        )

                if "vehicle_masks" in image_infos:
                    vehicle_mask = get_numpy(image_infos["vehicle_masks"])
                    if vehicle_mask.sum() > 0:
                        vehicle_psnrs.append(
                            compute_psnr(
                                rgb[vehicle_mask], image_infos["pixels"][vehicle_mask]
                            )
                        )
                        vehicle_ssims.append(
                            ssim(
                                get_numpy(rgb),
                                get_numpy(image_infos["pixels"]),
                                data_range=1.0,
                                channel_axis=-1,
                                full=True,
                            )[1][vehicle_mask].mean()
                        )

    # messy aggregation...
    results_dict = {}
    results_dict["psnr"] = non_zero_mean(psnrs) if compute_metrics else -1
    results_dict["ssim"] = non_zero_mean(ssim_scores) if compute_metrics else -1
    results_dict["lpips"] = non_zero_mean(lpipss) if compute_metrics else -1
    # syc >>>
    results_dict["psnr_no_ego"] = non_zero_mean(psnrs_no_ego) if compute_metrics else -1
    results_dict["ssim_no_ego"] = (
        non_zero_mean(ssim_scores_no_ego) if compute_metrics else -1
    )
    results_dict["lpips_no_ego"] = (
        non_zero_mean(lpipss_no_ego) if compute_metrics else -1
    )
    results_dict["psnr_with_ego"] = (
        non_zero_mean(psnrs_with_ego) if compute_metrics else -1
    )
    results_dict["ssim_with_ego"] = (
        non_zero_mean(ssim_scores_with_ego) if compute_metrics else -1
    )
    results_dict["lpips_with_ego"] = (
        non_zero_mean(lpipss_with_ego) if compute_metrics else -1
    )
    # <<<
    results_dict["occupied_psnr"] = (
        non_zero_mean(occupied_psnrs) if compute_metrics else -1
    )
    results_dict["occupied_ssim"] = (
        non_zero_mean(occupied_ssims) if compute_metrics else -1
    )
    results_dict["masked_psnr"] = non_zero_mean(masked_psnrs) if compute_metrics else -1
    results_dict["masked_ssim"] = non_zero_mean(masked_ssims) if compute_metrics else -1
    results_dict["human_psnr"] = non_zero_mean(human_psnrs) if compute_metrics else -1
    results_dict["human_ssim"] = non_zero_mean(human_ssims) if compute_metrics else -1
    results_dict["vehicle_psnr"] = (
        non_zero_mean(vehicle_psnrs) if compute_metrics else -1
    )
    results_dict["vehicle_ssim"] = (
        non_zero_mean(vehicle_ssims) if compute_metrics else -1
    )
    results_dict["rgbs"] = rgbs
    results_dict["depths"] = depths
    results_dict["cam_names"] = cam_names
    results_dict["cam_ids"] = cam_ids
    if len(opacities) > 0:
        results_dict["opacities"] = opacities
    if len(gt_rgbs) > 0:
        results_dict["gt_rgbs"] = gt_rgbs
    if len(gt_road_rgbs) > 0:
        results_dict["gt_road_rgbs"] = gt_road_rgbs
    if len(road_rgbs) > 0:
        results_dict["road_rgbs"] = road_rgbs
    if len(error_maps) > 0:
        results_dict["rgb_error_maps"] = error_maps
    if len(rgb_sky_blend) > 0:
        results_dict["rgb_sky_blend"] = rgb_sky_blend
    if len(rgb_sky) > 0:
        results_dict["rgb_sky"] = rgb_sky
    if len(sky_masks) > 0:
        results_dict["gt_sky_masks"] = sky_masks
    if len(lidar_on_images) > 0:
        results_dict["lidar_on_images"] = lidar_on_images
    if len(Background_rgbs) > 0:
        results_dict["Background_rgbs"] = Background_rgbs
    if len(RigidNodes_rgbs) > 0:
        results_dict["RigidNodes_rgbs"] = RigidNodes_rgbs
    if len(DeformableNodes_rgbs) > 0:
        results_dict["DeformableNodes_rgbs"] = DeformableNodes_rgbs
    if len(SMPLNodes_rgbs) > 0:
        results_dict["SMPLNodes_rgbs"] = SMPLNodes_rgbs
    if len(Dynamic_rgbs) > 0:
        results_dict["Dynamic_rgbs"] = Dynamic_rgbs
    if len(Background_depths) > 0:
        results_dict["Background_depths"] = Background_depths
    if len(RigidNodes_depths) > 0:
        results_dict["RigidNodes_depths"] = RigidNodes_depths
    if len(DeformableNodes_depths) > 0:
        results_dict["DeformableNodes_depths"] = DeformableNodes_depths
    if len(SMPLNodes_depths) > 0:
        results_dict["SMPLNodes_depths"] = SMPLNodes_depths
    if len(Dynamic_depths) > 0:
        results_dict["Dynamic_depths"] = Dynamic_depths
    if len(Background_opacities) > 0:
        results_dict["Background_opacities"] = Background_opacities
    if len(RigidNodes_opacities) > 0:
        results_dict["RigidNodes_opacities"] = RigidNodes_opacities
    if len(DeformableNodes_opacities) > 0:
        results_dict["DeformableNodes_opacities"] = DeformableNodes_opacities
    if len(SMPLNodes_opacities) > 0:
        results_dict["SMPLNodes_opacities"] = SMPLNodes_opacities
    if len(Dynamic_opacities) > 0:
        results_dict["Dynamic_opacities"] = Dynamic_opacities
    return results_dict


def save_images(
    render_results: Dict[str, List[Tensor]],
    save_pth: str,
    layout: Callable,
    timestamps: int,
    keys: List[str] = ["gt_rgbs", "rgbs", "depths"],
    verbose: bool = True,
):
    return_frame = save_seperate_images(
        render_results,
        save_pth,
        layout,
        timestamps=timestamps,
        keys=keys,
        verbose=verbose,
    )
    return return_frame


def save_seperate_images(
    render_results: Dict[str, List[Tensor]],
    save_pth: str,
    layout: Callable,
    timestamps: int,
    keys: List[str] = ["gt_rgbs", "rgbs", "depths"],
    verbose: bool = False,
):
    return_frame_dict = {}
    for key in keys:
        video_save_pth = save_pth.replace(".mp4", f"_{key}_layout.mp4")
        img_save_pth = save_pth.replace(".mp4", f"_{key}.png")
        if not os.path.exists(img_save_pth.replace(".png", "")):
            os.makedirs(img_save_pth.replace(".png", ""), exist_ok=True) 
        if not os.path.exists(video_save_pth.replace(".mp4", "")):
            os.makedirs(video_save_pth.replace(".mp4", ""), exist_ok=True) 

        if "mask" not in key:
            if key not in render_results or len(render_results[key]) == 0:
                continue

        cam_names = render_results["cam_names"]
        # skip if the key is not in render_results
        if "mask" in key:
            new_key = key.replace("mask", "opacities")
            if new_key not in render_results or len(render_results[new_key]) == 0:
                continue
            frames = render_results[new_key]
        else:
            if key not in render_results or len(render_results[key]) == 0:
                continue
            frames = render_results[key]
        # convert to rgb if necessary
        if key == "gt_sky_masks":
            frames = [np.stack([frame, frame, frame], axis=-1) for frame in frames]
        elif "mask" in key:
            frames = [np.stack([frame, frame, frame], axis=-1) for frame in frames]
        elif "depth" in key:
            try:
                opacities = render_results[key.replace("depths", "opacities")]
            except:
                if "median" in key:
                    opacities = render_results[
                        key.replace("median_depths", "opacities")
                    ]
                else:
                    continue
            frames = [
                depth_visualizer(frame, opacity)
                for frame, opacity in zip(frames, opacities)
            ]

        for j, frame in enumerate(frames):
            imageio.imwrite(
                img_save_pth.replace(".png", f"/{timestamps:06d}_{j:03d}.png"),
                to8b(frame),
            )
        tiled_img = layout(frames, cam_names)
        tiled_img = to8b(tiled_img)
        video_save_pth = video_save_pth.replace(".mp4", f"/{timestamps:06d}.png")
        imageio.imwrite(
            video_save_pth,
            tiled_img,
        )
        return_frame_dict[key] = tiled_img
    del render_results
    return return_frame_dict


def save_videos(
    render_results: Dict[str, List[Tensor]],
    save_pth: str,
    layout: Callable,
    num_timestamps: int,
    keys: List[str] = ["gt_rgbs", "rgbs", "depths"],
    num_cams: int = 3,
    save_seperate_video: bool = False,
    save_images: bool = False,
    fps: int = 10,
    verbose: bool = True,
):
    if save_seperate_video:
        print("Save videos path: ", save_pth)
        return_frame = save_seperate_videos(
            render_results,
            save_pth,
            layout,
            num_timestamps=num_timestamps,
            keys=keys,
            num_cams=num_cams,
            save_images=save_images,
            fps=fps,
            verbose=verbose,
        )
    else:
        print("Save concatenated videos")
        return_frame = save_concatenated_videos(
            render_results,
            save_pth,
            layout,
            num_timestamps=num_timestamps,
            keys=keys,
            num_cams=num_cams,
            save_images=save_images,
            fps=fps,
            verbose=verbose,
        )
    return return_frame


def render_novel_views(
    trainer: BasicTrainer,
    render_data: list,
    camera_data: CameraData,
) -> list:
    trainer.set_eval()

    need_ego_blend = hasattr(camera_data, 'egocar_mask') and camera_data.egocar_mask is not None and \
                     hasattr(camera_data, 'ego_template') and camera_data.ego_template is not None

    with torch.no_grad():
        rgbs = []
        depths = []
        opacities = []

        for frame_data in render_data:
            # Move data to GPU
            for key, value in frame_data["cam_infos"].items():
                frame_data["cam_infos"][key] = value.cuda(non_blocking=True)
            for key, value in frame_data["image_infos"].items():
                frame_data["image_infos"][key] = value.cuda(non_blocking=True)

            # Perform rendering
            outputs, _ = trainer(
                image_infos=frame_data["image_infos"],
                camera_infos=frame_data["cam_infos"],
                novel_view=True,
                render_only=True,
            )

            # ------------- clip rgb ------------- #
            for k, v in outputs.items():
                if isinstance(v, Tensor) and "rgb" in k:
                    outputs[k] = v.clamp(0.0, 1.0)

            rgb = get_numpy(outputs["rgb"])
            ### NOTE(gls): 增加ego_mask原始
            image_infos = frame_data["image_infos"]

            # 融合 ego_mask
            # NOTE(ga): novel 轨迹的 image_infos 可能不含 egocar_masks，此时跳过。
            if need_ego_blend and "egocar_masks" in image_infos:
                rgb_tensor = outputs["rgb"]  # 新轨迹渲染的RGB
                ego_mask = frame_data["image_infos"]["egocar_masks"][..., None]  # [H, W, 1]
                ego_template = camera_data.ego_template  # ego区域模板 [H, W, 3]
                
                # 混合：非ego区域=新渲染，ego区域=原始模板
                rgb_tensor = torch.where(ego_mask, ego_template, rgb_tensor)
                rgb = get_numpy(rgb_tensor)
            
            depth = get_numpy(outputs["depth"])
            ### NOTE(gls): 增加ego_mask区域，深度设置为极端颜色
            if "egocar_masks" in image_infos:
                depth_tensor = outputs["depth"]
                ego_mask = image_infos["egocar_masks"][..., None]
                depth_tensor = torch.where(ego_mask, 200.0, depth_tensor)
                depth = get_numpy(depth_tensor)
            
            opacity = get_numpy(outputs["opacity"]) if "opacity" in outputs else None
            # 恢复畸变图像
            # NOTE(syc): 是否不只针对鱼眼相机？
            if camera_data.is_fisheye:
                intrinsics = frame_data["cam_infos"]["intrinsics"].cpu().numpy()
                kb_coeffs = frame_data["cam_infos"]["kb_coeffs"].cpu().numpy().flatten()
                crop = False

                rgb = distort_image(
                    image=rgb,
                    intrinsics=intrinsics,
                    kb_coeffs=kb_coeffs,
                )
                depth = distort_image(
                    image=depth,
                    intrinsics=intrinsics,
                    kb_coeffs=kb_coeffs,
                )
                if opacity is not None:
                    opacity = distort_image(
                        image=opacity,
                        intrinsics=intrinsics,
                        kb_coeffs=kb_coeffs,
                    )

            rgbs.append(rgb)
            depths.append(depth)
            if opacity is not None:
                opacities.append(opacity)

        render_results = {}
        render_results["rgbs"] = rgbs
        render_results["depths"] = depths
        if len(opacities) > 0:
            render_results["opacities"] = opacities

    return render_results


def save_concatenated_videos(
    render_results: Dict[str, List[Tensor]],
    save_pth: str,
    layout: Callable,
    num_timestamps: int,
    keys: List[str] = ["gt_rgbs", "rgbs", "depths"],
    num_cams: int = 3,
    save_images: bool = False,
    fps: int = 10,
    verbose: bool = True,
):
    if num_timestamps == 1:  # it's an image
        writer = imageio.get_writer(save_pth, mode="I")
        return_frame_id = 0
    else:
        return_frame_id = num_timestamps // 2
        writer = imageio.get_writer(save_pth, mode="I", fps=fps)
    for i in trange(num_timestamps, desc="saving video", dynamic_ncols=True):
        merged_list = []
        cam_names = render_results["cam_names"][i * num_cams : (i + 1) * num_cams]
        for key in keys:
            # skip if the key is not in render_results
            if "mask" in key:
                new_key = key.replace("mask", "opacities")
                if new_key not in render_results or len(render_results[new_key]) == 0:
                    continue
                frames = render_results[new_key][i * num_cams : (i + 1) * num_cams]
            else:
                if key not in render_results or len(render_results[key]) == 0:
                    continue
                frames = render_results[key][i * num_cams : (i + 1) * num_cams]
            # convert to rgb if necessary
            if key == "gt_sky_masks":
                frames = [np.stack([frame, frame, frame], axis=-1) for frame in frames]
            elif "mask" in key:
                frames = [np.stack([frame, frame, frame], axis=-1) for frame in frames]
            elif "depth" in key:
                try:
                    opacities = render_results[key.replace("depths", "opacities")][
                        i * num_cams : (i + 1) * num_cams
                    ]
                except:
                    if "median" in key:
                        opacities = render_results[
                            key.replace("median_depths", "opacities")
                        ][i * num_cams : (i + 1) * num_cams]
                    else:
                        continue
                frames = [
                    depth_visualizer(frame, opacity)
                    for frame, opacity in zip(frames, opacities)
                ]
            tiled_img = layout(frames, cam_names)
            # frames = np.concatenate(frames, axis=1)
            merged_list.append(tiled_img)
        merged_frame = to8b(np.concatenate(merged_list, axis=0))
        if i == return_frame_id:
            return_frame = merged_frame
        writer.append_data(merged_frame)
    writer.close()
    if verbose:
        logger.info(f"saved video to {save_pth}")
    del render_results
    return {"concatenated_frame": return_frame}


def save_seperate_videos(
    render_results: Dict[str, List[Tensor]],
    save_pth: str,
    layout: Callable,
    num_timestamps: int,
    keys: List[str] = ["gt_rgbs", "rgbs", "depths"],
    num_cams: int = 3,
    fps: int = 10,
    verbose: bool = False,
    save_images: bool = False,
):
    return_frame_id = num_timestamps // 2
    return_frame_dict = {}
    for key in keys:
        tmp_save_pth = save_pth.replace(".mp4", f"_{key}.mp4")
        tmp_save_pth = tmp_save_pth.replace(".png", f"_{key}.png")
        if num_timestamps == 1:  # it's an image
            writer = imageio.get_writer(tmp_save_pth, mode="I")
        else:
            writer = imageio.get_writer(tmp_save_pth, mode="I", fps=fps)
        if "mask" not in key:
            if key not in render_results or len(render_results[key]) == 0:
                continue
        for i in range(num_timestamps):
            cam_names = render_results["cam_names"][i * num_cams : (i + 1) * num_cams]
            cam_ids = render_results["cam_ids"][i * num_cams : (i + 1) * num_cams]
            # skip if the key is not in render_results
            if "mask" in key:
                new_key = key.replace("mask", "opacities")
                if new_key not in render_results or len(render_results[new_key]) == 0:
                    continue
                frames = render_results[new_key][i * num_cams : (i + 1) * num_cams]
            else:
                if key not in render_results or len(render_results[key]) == 0:
                    continue
                frames = render_results[key][i * num_cams : (i + 1) * num_cams]
            # convert to rgb if necessary
            if key == "gt_sky_masks":
                frames = [np.stack([frame, frame, frame], axis=-1) for frame in frames]
            elif "mask" in key:
                frames = [np.stack([frame, frame, frame], axis=-1) for frame in frames]
            elif "depth" in key:
                try:
                    opacities = render_results[key.replace("depths", "opacities")][
                        i * num_cams : (i + 1) * num_cams
                    ]
                except:
                    if "median" in key:
                        opacities = render_results[
                            key.replace("median_depths", "opacities")
                        ][i * num_cams : (i + 1) * num_cams]
                    else:
                        continue
                frames = [
                    depth_visualizer(frame, opacity)
                    for frame, opacity in zip(frames, opacities)
                ]
            tiled_img = layout(frames, cam_names)
            if save_images:
                if i == 0:
                    os.makedirs(tmp_save_pth.replace(".mp4", ""), exist_ok=True)
                for j, frame in enumerate(frames):
                    imageio.imwrite(
                        tmp_save_pth.replace(".mp4", f"/{i:06d}_{cam_ids[j]}.png"), # NOTE:恢复id，原来是索引
                        to8b(frame),
                    )
            # frames = to8b(np.concatenate(frames, axis=1))
            frames = to8b(tiled_img)
            writer.append_data(frames)
            if i == return_frame_id:
                return_frame_dict[key] = frames
        # close the writer
        writer.close()
        del writer
        if verbose:
            logger.info(f"saved video to {tmp_save_pth}")
    del render_results
    return return_frame_dict


def save_seperate_images(
    render_results: Dict[str, List[Tensor]],
    save_pth: str,
    layout: Callable,
    timestamps: int,
    keys: List[str] = ["gt_rgbs", "rgbs", "depths"],
    verbose: bool = False,
):
    return_frame_dict = {}
    for key in keys:
        video_save_pth = save_pth.replace(".mp4", f"_{key}_layout.mp4")
        img_save_pth = save_pth.replace(".mp4", f"_{key}.png")
        if not os.path.exists(img_save_pth.replace(".png", "")):
            os.makedirs(img_save_pth.replace(".png", ""), exist_ok=True) 
        if not os.path.exists(video_save_pth.replace(".mp4", "")):
            os.makedirs(video_save_pth.replace(".mp4", ""), exist_ok=True) 

        if "mask" not in key:
            if key not in render_results or len(render_results[key]) == 0:
                continue

        cam_names = render_results["cam_names"]
        cam_ids = render_results["cam_ids"]

        # skip if the key is not in render_results
        if "mask" in key:
            new_key = key.replace("mask", "opacities")
            if new_key not in render_results or len(render_results[new_key]) == 0:
                continue
            frames = render_results[new_key]
        else:
            if key not in render_results or len(render_results[key]) == 0:
                continue
            frames = render_results[key]
        # convert to rgb if necessary
        if key == "gt_sky_masks":
            frames = [np.stack([frame, frame, frame], axis=-1) for frame in frames]
        elif "mask" in key:
            frames = [np.stack([frame, frame, frame], axis=-1) for frame in frames]
        elif "depth" in key:
            try:
                opacities = render_results[key.replace("depths", "opacities")]
            except:
                if "median" in key:
                    opacities = render_results[
                        key.replace("median_depths", "opacities")
                    ]
                else:
                    continue
            frames = [
                depth_visualizer(frame, opacity)
                for frame, opacity in zip(frames, opacities)
            ]

        for j, frame in enumerate(frames):
            imageio.imwrite(
                img_save_pth.replace(".png", f"/{timestamps:06d}_{cam_ids[j]:03d}.png"),
                to8b(frame),
            )
        tiled_img = layout(frames, cam_names)
        tiled_img = to8b(tiled_img)
        video_save_pth = video_save_pth.replace(".mp4", f"/{timestamps:06d}.png")
        imageio.imwrite(
            video_save_pth,
            tiled_img,
        )
        return_frame_dict[key] = tiled_img
    del render_results
    return return_frame_dict


def save_single_camera_video(
    render_results: Dict[str, List[Tensor]],
    cam_id: int,
    start_timestep: int,
    end_timestep: int,
    video_save_pth: str,
    image_save_dir: str,
    keys: List[str] = ["rgbs", "depths"],
    fps: int = 10,
    verbose: bool = False,
):
    for key in keys:
        if image_save_dir is not None:
            # 为每个 timestep 创建目录
            for timestep in range(start_timestep, end_timestep):
                os.makedirs(
                    os.path.join(image_save_dir, f"{timestep:06d}"), exist_ok=True
                )

        video_tmp_save_pth = video_save_pth.replace(".mp4", f"_cam{cam_id}_{key}.mp4")
        writer = imageio.get_writer(video_tmp_save_pth, mode="I", fps=fps)

        if "mask" not in key:
            if key not in render_results or len(render_results[key]) == 0:
                continue

        for frame_idx, timestep in enumerate(range(start_timestep, end_timestep)):
            # skip if the key is not in render_results
            if "mask" in key:
                new_key = key.replace("mask", "opacities")
                if new_key not in render_results or len(render_results[new_key]) == 0:
                    continue
                single_frame = render_results[new_key][frame_idx]
            else:
                if key not in render_results or len(render_results[key]) == 0:
                    continue
                single_frame = render_results[key][frame_idx]

            # convert to rgb if necessary
            if key == "gt_sky_masks":
                single_frame = [
                    np.stack([frame, frame, frame], axis=-1) for frame in single_frame
                ]
            elif "mask" in key:
                single_frame = [
                    np.stack([frame, frame, frame], axis=-1) for frame in single_frame
                ]
            elif "depth" in key:
                try:
                    opacities = render_results[key.replace("depths", "opacities")][
                        frame_idx
                    ]
                except:
                    if "median" in key:
                        opacities = render_results[
                            key.replace("median_depths", "opacities")
                        ][frame_idx]
                    else:
                        continue
                single_frame = [
                    depth_visualizer(frame, opacity)
                    for frame, opacity in zip(single_frame, opacities)
                ]

            single_frame = np.array(single_frame)
            single_frame = to8b(single_frame)

            if image_save_dir is not None:
                imageio.imwrite(
                    os.path.join(
                        image_save_dir, f"{timestep:06d}", f"{cam_id}_{key}.png"
                    ),
                    single_frame,
                )
            writer.append_data(single_frame)

        # close the writer
        writer.close()
        del writer

        if verbose:
            logger.info(f"saved video to {video_tmp_save_pth}")
