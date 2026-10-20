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

from cv2 import transform
import torch
from torch import nn
import numpy as np
from utils.graphics_utils import (
    getWorld2View2,
    getProjectionMatrix,
    getAsymmetricProjectionMatrix,
)
import math


class Camera(nn.Module):
    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        image,
        gt_alpha_mask,
        img_mask,
        image_name,
        uid,
        beam_inclinations,
        lidar_center,
        original_lidar_center,
        lidar_hfov=None,
        width=None,
        height=None,
        fov_up=None,
        fov_down=None,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx if FoVx is not None else 512
        self.FoVy = FoVy if FoVy is not None else 512
        self.image_name = image_name
        self.img_mask = None

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(
                f"[Warning] Custom device {data_device} failed, fallback to default cuda device"
            )
            self.data_device = torch.device("cuda")

        self.original_image = None
        if image is not None:
            self.original_image = torch.tensor(
                image, dtype=torch.float32, device=data_device
            )
        self.beam_inclinations = torch.tensor(
            beam_inclinations, dtype=torch.float32, device=data_device
        )
        # 水平视场角（弧度，完整 HFOV），与 beam_inclinations 同为 range-view 投影内参，
        # 由 dataloader 经 CameraInfo 传入，供 renderer 喂给 rasterizer kernel
        self.lidar_hfov = lidar_hfov
        if self.original_image is not None:
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]
        else:
            self.image_width = int(width)
            self.image_height = int(height)

        if img_mask is not None:
            self.img_mask = torch.tensor(
                img_mask, dtype=torch.float32, device=data_device
            )
        else:
            self.img_mask = torch.ones(
                (self.image_height, self.image_width),
                dtype=torch.float32,
                device=self.data_device,
            )

        if self.original_image is not None and gt_alpha_mask is not None:
            gt_alpha_mask = torch.tensor(gt_alpha_mask)
            self.original_image *= gt_alpha_mask.to(self.data_device)
        elif self.original_image is not None:
            self.original_image *= torch.ones(
                (1, self.image_height, self.image_width),
                dtype=torch.float32,
                device=self.data_device,
            )

        self.zfar = 1000.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.FoVx = FoVx if FoVx is not None else self.FoVx
        self.FoVy = FoVy if FoVy is not None else self.FoVy
        self.up_angle = fov_up if fov_up is not None else self.FoVy * 0.5
        self.down_angle = (
            -fov_down if fov_down is not None else -self.FoVy * 0.5
        )  # 约定输入为向下正值
        self.left_angle = -self.FoVx * 0.5
        self.right_angle = self.FoVx * 0.5

        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        )
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        # self.projection_matrix = getAsymmetricProjectionMatrix(znear=self.znear, zfar=self.zfar, \
        #                                                        left_angle=self.left_angle, right_angle=self.right_angle, \
        #                                                        up_angle=self.up_angle, down_angle=self.down_angle).transpose(0,1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.lidar_center = torch.tensor(
            lidar_center, dtype=torch.float32, device=data_device
        )
        self.original_lidar_center = torch.tensor(
            original_lidar_center, dtype=torch.float32, device=data_device
        )


class MiniCam:
    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
