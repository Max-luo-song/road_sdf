import argparse
import os
import imageio
import numpy as np
from PIL import Image
import json
from datasets.utils.box_utils import draw_3d_box_on_img
from qcraft_helpers import (
    project_label_to_image,
    load_calibration,
    load_lidar2ego,
    load_available_camera_ids,
)

SKY_COLOR     = (135, 206, 235)
ROAD_COLOR    = (128, 64, 128)
HUMAN_COLOR   = (220, 20, 60)
VEHICLE_COLOR = (0, 0, 142)


def visualize_mask(image, mask, color):
    # 创建一个与输入图像相同大小的透明颜色层
    overlay = np.zeros_like(image, dtype=np.uint8)
    
    # 将掩码的非零区域设置为指定颜色（忽略透明度通道）
    mask_area = np.array(mask) > 0
    overlay[mask_area] = color[:3]
    
    # 计算Alpha混合
    alpha = 0.4  # 将透明度归一化到[0, 1]
    result = image.copy()
    result[mask_area] = (1 - alpha) * image[mask_area] + alpha * overlay[mask_area]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 masks 可视化并保存为视频。")
    parser.add_argument("--data_dir", type=str, help="Path to the input data directory.")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory.")
    parser.add_argument("--use_fine_dynamic_mask", action='store_true', help="Whether to use fine dynamic masks.")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir

    if args.use_fine_dynamic_mask:
        dynamic_mask_dir = "fine_dynamic_masks"
    else:
        dynamic_mask_dir = "dynamic_masks"

    if os.path.exists(os.path.join(data_dir, "lidar_pose")):
        total_frames = len(os.listdir(os.path.join(data_dir, "lidar_pose")))

    save_dir = os.path.join(output_dir, "mask_vis")
    os.makedirs(save_dir, exist_ok=True)

    frame_instance_path = os.path.join(data_dir, "instances", "frame_instances.json")
    instances_info_path = os.path.join(data_dir, "instances", "instances_info.json")
    with open(frame_instance_path, "r") as f:
        frame_instances = json.load(f)
    with open(instances_info_path, "r") as f:
        instances_info = json.load(f)
    
    cam_ids = load_available_camera_ids(data_dir)

    lidar2ego  = load_lidar2ego(data_dir)
    cam2lidars, intrinsics = load_calibration(data_dir, cam_ids)
    cam2egos = {cam_id: lidar2ego @ cam2lidar for cam_id, cam2lidar in cam2lidars.items()}

    vis_imgs = {}
    for frame_idx in range(total_frames):
        print(f"Processing frame {frame_idx}")
        for cam_id in cam_ids:
            rgb_image_path = os.path.join(data_dir, "images", f"{frame_idx:06d}_{cam_id}.png")
            vis_image = Image.open(rgb_image_path)
            vis_image = np.array(vis_image)

            filename = f"{frame_idx:06d}_{cam_id}.png"

            sky_mask_path = os.path.join(data_dir, "sky_masks", filename)
            sky_mask = Image.open(sky_mask_path).convert("L")

            road_mask_path = os.path.join(data_dir, "road_masks", filename)
            road_mask = Image.open(road_mask_path).convert("L")

            human_mask_path = os.path.join(data_dir, dynamic_mask_dir, "human", filename)
            human_mask = Image.open(human_mask_path).convert("L")

            vehicle_mask_path = os.path.join(data_dir, dynamic_mask_dir, "vehicle", filename)
            vehicle_mask = Image.open(vehicle_mask_path).convert("L")

            vis_image = visualize_mask(vis_image, road_mask, ROAD_COLOR)
            vis_image = visualize_mask(vis_image, sky_mask, SKY_COLOR)
            vis_image = visualize_mask(vis_image, human_mask, HUMAN_COLOR)
            vis_image = visualize_mask(vis_image, vehicle_mask, VEHICLE_COLOR)

            for track_id in frame_instances[str(frame_idx)]:
                frame_annotations = instances_info[str(track_id)]['frame_annotations']
                idx = frame_annotations['frame_idx'].index(frame_idx)
                obj2ego = np.array(frame_annotations['obj_to_ego'][idx]).reshape(4, 4)
                l, w, h= frame_annotations['box_size'][idx]
                vertices, valid = project_label_to_image(
                    dim=[l, w, h],
                    obj2ego=obj2ego,
                    cam2ego=cam2egos[cam_id],
                    intrinsic=intrinsics[cam_id],
                    img_shape=vis_image.shape[:2],
                )

                if valid.all():
                    vertices = vertices.reshape(2, 2, 2, 2).astype(np.int32)
                    draw_3d_box_on_img(vertices, vis_image)
            
            if cam_id not in vis_imgs.keys():
                vis_imgs[cam_id] = []
            vis_imgs[cam_id].append(vis_image)

    # save visualization
    for cam_id, imgs in vis_imgs.items():
        print(f"Saving video for cam {cam_id}...")
        video_path = os.path.join(save_dir, f"cam_{cam_id}.mp4")
        imageio.mimwrite(video_path, imgs, fps=10)

