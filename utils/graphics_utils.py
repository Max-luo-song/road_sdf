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

import torch
import math
import numpy as np
from typing import NamedTuple


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getAsymmetricProjectionMatrix(
    znear, zfar, left_angle, right_angle, up_angle, down_angle
):
    up = math.tan(up_angle) * znear
    down = math.tan(down_angle) * znear  # 注意：tan(-13°) 是负数
    right = math.tan(right_angle) * znear
    left = math.tan(left_angle) * znear  # 负数

    P = torch.zeros(4, 4)

    # x-axis
    P[0, 0] = 2 * znear / (right - left)
    P[0, 2] = (right + left) / (right - left)  # 非零！表示主点偏移

    # y-axis
    P[1, 1] = 2 * znear / (up - down)
    P[1, 2] = (up + down) / (up - down)  # 非零！

    # z-axis (OpenGL style, depth in [-1, 1] or [0, 1] depending on convention)
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = 1.0

    return P


def getProjectionMatrix(
    znear, zfar, fovX, fovY
):  # TODO 潜在问题 这里当成了对称正交投影 但是真实相机是非对称投影矩阵
    tanHalfFovY = math.tan((fovY / 2))  # cx/fx
    tanHalfFovX = math.tan((fovX / 2))  # cy/fy

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)  # = fx/cx
    P[1, 1] = 2.0 * znear / (top - bottom)  # = fy/cy
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))
