import numpy as np
import open3d as o3d


def save_pointcloud_pcd(points: np.ndarray, filename: str):
    """
    保存点云为PCD格式
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)


# NOTE(syc): 临时作为雷达仿真用，后续会改用其他方法
def unproject_depth_to_pointcloud(
    depth_map: np.ndarray, intrinsic: np.ndarray, T_cam_to_lidar: np.ndarray
) -> np.ndarray:
    """
    将深度图转换为目标坐标系的点云

    Args:
        depth_map: 深度图，形状为(H, W)或(H, W, C)
        intrinsic: 相机内参矩阵，形状为(3, 3)
        extrinsic: 外参矩阵，从相机坐标系到目标坐标系，形状为(4, 4)

    Returns:
        points_world: 世界坐标系下的点云，形状为(N, 3)
    """
    # 处理不同形状的深度图
    if len(depth_map.shape) == 2:
        height, width = depth_map.shape
        depth = depth_map
    else:  # 特殊处理Blender的深度数据(RGBA格式)
        height, width, _ = depth_map.shape
        depth = depth_map[:, :, 0]  # 假设深度信息在第一个通道

    # 从内参矩阵中提取参数
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    # 生成像素坐标网格
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)

    max_depth = 20

    # 过滤无效深度值
    valid_mask = (depth > 0) & (depth <= max_depth)
    # if not np.any(valid_mask):
    #     return np.empty((0, 3))

    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    depth_valid = depth[valid_mask]

    # 将像素坐标转换为相机坐标系
    X = (x_valid - cx) * depth_valid / fx
    Y = (y_valid - cy) * depth_valid / fy
    Z = depth_valid

    # 组合成相机坐标系下的点云
    points_camera = np.stack([X, Y, Z], axis=-1)  # (N, 3)

    # 转换为齐次坐标并变换到世界坐标系
    points_camera_homo = np.hstack(
        [points_camera, np.ones((points_camera.shape[0], 1))]
    )
    points_lidar_homo = (T_cam_to_lidar @ points_camera_homo.T).T
    return points_lidar_homo[:, :3]


def remove_ground_points(
    point_cloud: np.ndarray, ground_height: float = 0.0
) -> np.ndarray:
    """
    移除地面以下的点云

    Args:
        point_cloud: 世界坐标系下的点云，形状为(N, 3)
        ground_height: 地面高度阈值（默认为0，可根据实际情况调整）

    Returns:
        filtered_point_cloud: 过滤后的点云
    """
    # 提取Y坐标（假设Y轴向上，地面在Y=0附近）
    z_coords = point_cloud[:, 2]

    # 保留Y坐标 >= ground_height的点（即地面及以上的点）
    mask = z_coords > ground_height

    # x_coords = point_cloud[:, 0]

    # # 保留Y坐标 >= ground_height的点（即地面及以上的点）
    # mask = x_coords >= ground_height

    # y_coords = point_cloud[:, 1]

    # # 保留Y坐标 >= ground_height的点（即地面及以上的点）
    # mask = y_coords >= ground_height

    return point_cloud[mask]
