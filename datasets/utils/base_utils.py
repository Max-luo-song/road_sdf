import numpy as np

def project_numpy(xyz, K, RT, H, W):
    xyz_cam = np.dot(xyz, RT[:3, :3].T) + RT[:3, 3:].T
    valid_depth = xyz_cam[:, 2] > 0
    xyz_pixel = np.dot(xyz_cam, K.T)
    xyz_pixel = xyz_pixel[:, :2] / xyz_pixel[:, 2:]
    valid_x = np.logical_and(xyz_pixel[:, 0] >= 0, xyz_pixel[:, 0] < W)
    valid_y = np.logical_and(xyz_pixel[:, 1] >= 0, xyz_pixel[:, 1] < H)
    valid_pixel = np.logical_and(valid_x, valid_y)
    mask = np.logical_and(valid_depth, valid_pixel)
    return xyz_pixel, mask

def project_points_to_image(xyz, K, RT, H, W):
    """投影世界点到图像平面，返回 cam_points(N,2), depth(N), valid_mask"""
    points_cam = (RT[:3, :3] @ xyz.T + RT[:3, 3:4]).T  # (num_pts, 3)
    points_img = (K @ points_cam.T).T  # (num_pts, 3)
    depth = points_img[:, 2]
    cam_points = points_img[:, :2] / (depth.unsqueeze(-1) + 1e-6)  # (num_pts, 2)
    valid_mask = (
        (cam_points[:, 0] >= 0) & (cam_points[:, 0] < W) &
        (cam_points[:, 1] >= 0) & (cam_points[:, 1] < H) &
        (depth > 0)
    )  # (num_pts, )
    return cam_points, depth, valid_mask