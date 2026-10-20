import argparse
import os
import matplotlib.cm as cm
import imageio
import numpy as np
import cv2
from qcraft_helpers import load_calibration, load_available_camera_ids


def load_lidar(data_path):
    if os.path.exists(os.path.join(data_path, "lidar_pose")):
        total_frames = len(os.listdir(os.path.join(data_path, "lidar_pose")))

    lidar_points = []
    for t in range(total_frames):
        lidar_path = os.path.join(data_path, "lidar", "bin", f"{t:06d}.bin")
        lidar_info = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
        lidar_info = lidar_info[lidar_info[:, -1] == 0]
        lidar_info = lidar_info[:, :4]  # x, y, z, intensity
        lidar_points.append(lidar_info)

    return lidar_points


def visualize_lidar(image, lidar_points, intrinsics, lidar2cam):
    points_xyz = lidar_points[:, :3]
    intensities = lidar_points[:, 3]

    # 将点云转换到相机坐标系
    points_homo = np.hstack((points_xyz, np.ones((len(points_xyz), 1))))
    points_camera = np.dot(lidar2cam, points_homo.T).T[:, :3]

    # 筛选在相机前方的点（z > 0）
    valid_mask = points_camera[:, 2] > 0
    points_camera = points_camera[valid_mask]
    valid_intensities = intensities[valid_mask]

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u = (fx * points_camera[:, 0] / points_camera[:, 2] + cx).astype(np.int32)
    v = (fy * points_camera[:, 1] / points_camera[:, 2] + cy).astype(np.int32)

    image_points = np.stack((u, v), axis=1)

    h, w = image.shape[:2]
    valid_image_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    image_points = image_points[valid_image_mask]
    valid_intensities = valid_intensities[valid_image_mask]

    intensities = np.clip(valid_intensities, 0, 1)
    colormap = cm.get_cmap('jet')  # 获取 Jet 调色板
    colors = colormap(intensities)[:, :3]  # 获取 RGB 颜色（忽略 alpha 通道）
    colors = (colors * 255).astype(np.uint8)  # 转换为 0-255 的整数值

    for (u, v), color in zip(image_points, colors):
        bgr_color = (int(color[2]), int(color[1]), int(color[0]))
        cv2.circle(image, (u, v), 1, bgr_color, -1)

    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 masks 可视化并保存为视频。")
    parser.add_argument("--data_dir", type=str, help="Path to the input data directory.")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory.")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir

    cam_ids = load_available_camera_ids(data_dir)

    save_dir = os.path.join(output_dir, "lidar_vis")
    os.makedirs(save_dir, exist_ok=True)

    lidar_points_list = load_lidar(data_path=data_dir)
    cam2lidars, intrinsics = load_calibration(data_dir, cam_ids)

    for cam_id in cam_ids:
        print(f"Visualizing lidar for cam {cam_id}...")

        video_path = os.path.join(save_dir, f"cam_{cam_id}.mp4")
        writer = imageio.get_writer(video_path, mode="I", fps=10)

        intrinsic = intrinsics[cam_id]
        lidar2cam = np.linalg.inv(cam2lidars[cam_id])

        for frame_id in range(len(lidar_points_list)):
            lidar_points = lidar_points_list[frame_id]
            rgb_image_path = os.path.join(
                data_dir, "images", f"{frame_id:06d}_{cam_id}.png"
            )
            image = cv2.imread(rgb_image_path)

            vis_image = visualize_lidar(
                image.copy(), lidar_points, intrinsic, lidar2cam
            )

            # # save image
            # image_save_path = os.path.join(save_dir, f"{frame_id:06d}_{cam_id}.jpg")
            # cv2.imwrite(image_save_path, vis_image)

            # save video
            writer.append_data(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        writer.close()
        print(f"Saved lidar visualization video for cam {cam_id} at {video_path}")