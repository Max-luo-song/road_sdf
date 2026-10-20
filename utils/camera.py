# Camera pose manipulation and trajectory generation.
import os
import torch
import numpy as np
from typing import Dict

from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R


def get_interp_novel_trajectories(
    dataset_type: str,
    ego_to_worlds: torch.Tensor,
    traj_type: str = "original_traj",
    target_frames: int = 100,
) -> torch.Tensor:
    """
    直接基于全局的 ego_to_worlds 生成新轨迹
    """
    trajectory_generators = {
        "original_traj": lambda ego: _ego_shift_trajectory_fn(0, 0, 0, 0, ego),

        "left_shift_1m": lambda ego: _ego_shift_trajectory_fn(0, 1.0, 0, 0, ego),
        "left_shift_2m": lambda ego: _ego_shift_trajectory_fn(0, 2.0, 0, 0, ego),
        "left_shift_3m": lambda ego: _ego_shift_trajectory_fn(0, 3.0, 0, 0, ego),
        "left_shift_5m": lambda ego: _ego_shift_trajectory_fn(0, 5.0, 0, 0, ego),

        "right_shift_1m": lambda ego: _ego_shift_trajectory_fn(0, -1.0, 0, 0, ego),
        "right_shift_2m": lambda ego: _ego_shift_trajectory_fn(0, -2.0, 0, 0, ego),
        "right_shift_3m": lambda ego: _ego_shift_trajectory_fn(0, -3.0, 0, 0, ego),
        "right_shift_5m": lambda ego: _ego_shift_trajectory_fn(0, -5.0, 0, 0, ego),

        "front_shift_1m": lambda ego: _ego_shift_trajectory_fn(1.0, 0, 0, 0, ego),
        "front_shift_3m": lambda ego: _ego_shift_trajectory_fn(3.0, 0, 0, 0, ego),
        "front_shift_5m": lambda ego: _ego_shift_trajectory_fn(5.0, 0, 0, 0, ego),

        "back_shift_1m": lambda ego: _ego_shift_trajectory_fn(-1.0, 0, 0, 0, ego),
        "back_shift_3m": lambda ego: _ego_shift_trajectory_fn(-3.0, 0, 0, 0, ego),
        "back_shift_5m": lambda ego: _ego_shift_trajectory_fn(-5.0, 0, 0, 0, ego),


        "up_shift_5m": lambda ego: _ego_shift_trajectory_fn(0, 0, 5, 0, ego),
    }
        
    if traj_type not in trajectory_generators:
        raise ValueError(f"Unknown trajectory type: {traj_type}")

    return trajectory_generators[traj_type](ego_to_worlds)

def _ego_shift_trajectory_fn(
    front: float,
    left: float,
    up: float,
    yaw_deg: float,
    ego_to_worlds: torch.Tensor,
) -> torch.Tensor:
    """
    基于自车(Ego)坐标系生成新轨迹
    坐标系定义：X前，Y左，Z上
    """
    device = ego_to_worlds.device

    # 构造自车坐标系下的局部变换矩阵 (Offset)
    yaw_rad = np.deg2rad(yaw_deg)
    cos_y = np.cos(yaw_rad)
    sin_y = np.sin(yaw_rad)

    offset = torch.eye(4, device=device, dtype=torch.float32)
    
    # Z轴旋转 (偏航角)
    offset[0, 0] = cos_y
    offset[0, 1] = -sin_y
    offset[1, 0] = sin_y
    offset[1, 1] = cos_y
    
    # 平移 (X前, Y左, Z上)
    offset[0, 3] = front
    offset[1, 3] = left
    offset[2, 3] = up

    adjusted_ego_poses = torch.matmul(ego_to_worlds, offset.unsqueeze(0))
    return adjusted_ego_poses