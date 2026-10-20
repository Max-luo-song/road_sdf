import numpy as np
import cv2

from scipy.spatial.transform import Rotation


# FIXME: 好像是错的
def imu2ego():
    """
    imu在后轴中心，右前上 -> 车体坐标系在后轴中心地面点，前左上
    1) 右x，前y，上z -> 前x ，左y， 上z
    2) z平移到地面，E03轮半径: 0.349m
    """
    T = np.eye(4)
    r = Rotation.from_euler("z", -90, degrees=True)
    T[:3, :3] = r.as_matrix()
    T[2, 3] = 0.349  # E03轮半径
    return T


def project_points_to_image(points3d, intrinsic) -> np.ndarray:
    # 将3D点转换为齐次坐标
    points3d_homogeneous = np.concatenate(
        [points3d, np.ones((points3d.shape[0], 1))], axis=1
    )
    # 投影到2D
    camera_intrinsic_extended = np.hstack([intrinsic, np.zeros((3, 1))])
    points2d_homogeneous = camera_intrinsic_extended @ points3d_homogeneous.T
    # 转换为非齐次坐标
    z = points2d_homogeneous[2, :]
    points2d = points2d_homogeneous[:2, :] / z
    # 添加有效性检查
    valid_mask = z > 0
    points2d = points2d[:, valid_mask].T

    # # 检查 2D 点是否在图像范围内
    # h, w = img_shape
    # valid_mask[valid_mask] &= (
    #     (points2d[:, 0] >= 0)
    #     & (points2d[:, 0] < w)
    #     & (points2d[:, 1] >= 0)
    #     & (points2d[:, 1] < h)
    # )
    # points2d = points2d[valid_mask[valid_mask]]
    return points2d


def draw_and_fill_box(img, points2d):
    # 计算最小外接矩形
    x, y, w, h = cv2.boundingRect(points2d.astype(int))
    # print("points2d:", points2d)
    # print(f"Bounding Rect: x={x}, y={y}, w={w}, h={h}")

    # 绘制并填充矩形
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
    return img


def filter_points_in_box(pointcloud, obj_center_pos, size):
    """
    筛选出以obj_center_pos为中心、尺寸为size的矩形区域内的点云

    参数:
        pointcloud: numpy数组，形状为(N, 3)，表示点云数据
        obj_center_pos: 列表或数组，表示矩形中心坐标[x, y, z]
        size: 列表或数组，表示矩形在x、y、z三个维度上的尺寸[l, w, h]

    返回:
        筛选后的点云数据
    """
    if pointcloud is None:
        return None

    # 计算矩形边界
    half_size = np.array(size) / 2
    min_bounds = np.array(obj_center_pos) - half_size
    max_bounds = np.array(obj_center_pos) + half_size

    # 筛选在矩形边界内的点
    mask = (
        (pointcloud[:, 0] >= min_bounds[0])
        & (pointcloud[:, 0] <= max_bounds[0])
        & (pointcloud[:, 1] >= min_bounds[1])
        & (pointcloud[:, 1] <= max_bounds[1])
        & (pointcloud[:, 2] >= min_bounds[2])
        & (pointcloud[:, 2] <= max_bounds[2])
    )

    return pointcloud[mask]


def preprocess_lidar_point_cloud(point_cloud: np.ndarray, lidar_id: int) -> np.ndarray:
    # 转换为 float32
    point_cloud = np.stack(
        [point_cloud[field].astype(np.float32) for field in point_cloud.dtype.names],
        axis=1,
    )

    # 末尾增加 lidar_id 列
    lidar_id_col = np.full((point_cloud.shape[0], 1), lidar_id, dtype=np.float32)
    point_cloud = np.hstack([point_cloud, lidar_id_col])

    return point_cloud
