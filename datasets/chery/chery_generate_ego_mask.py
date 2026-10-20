import os
from PIL import Image, ImageDraw, ImageChops
import yaml
import cv2
import numpy as np
import json

IMAGE_SIZE = (3840, 2160)


def load_intrinsics(load_dir, clip_name, cam_id=0):
    data_path = os.path.join(
        load_dir, f"{clip_name}/dynamic_obj/autolabel_10hz/{clip_name}.json"
    )
    with open(data_path, "r") as f:
        data = json.load(f)

    params = data["calibration"][f"camera{cam_id}"]

    # # 原始相机内参
    intrinsic = params["intrinsic"]
    fx = intrinsic[0][0]
    cx = intrinsic[0][2]
    fy = intrinsic[1][1]
    cy = intrinsic[1][2]
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # 8 个畸变参数
    distcoeff = params["distcoeff"][0]
    distcoeff = np.array(distcoeff)

    # 去畸变后新的相机内参
    intrinsic_scaled = params["intrinsic_scaled"]
    fx_scaled = intrinsic_scaled[0][0]
    cx_scaled = intrinsic_scaled[0][2]
    fy_scaled = intrinsic_scaled[1][1]
    cy_scaled = intrinsic_scaled[1][2]
    intrinsic_scaled = np.array(
        [[fx_scaled, 0, cx_scaled], [0, fy_scaled, cy_scaled], [0, 0, 1]]
    )

    return intrinsic, distcoeff, intrinsic_scaled


def draw_polygon(image, points, fill=255):
    draw = ImageDraw.Draw(image)
    draw.polygon(points, fill=fill)
    return image


def undistort_image(image, intrinsics=None, distortions=None, new_intrinsics=None):
    image = cv2.undistort(
        np.array(image),
        intrinsics,
        distortions,
        newCameraMatrix=new_intrinsics,
    )
    return Image.fromarray(image)


# # NOTE(syc): 不要用这个，官方标的 mask 没有把车前盖遮掉
# def generate_official_ego_mask(mask_data, load_dir, clip_name):
#     cam_id = 0

#     points = [(point['x'], point['y']) for point in mask_data['ego_mask'].values()]

#     image = Image.new('L', IMAGE_SIZE, 0)
#     image = draw_polygon(image, points, fill=255)

#     image.save('./data/ego_masks/chery/0_distorted.png')

#     intrinsics, distortions, new_intrinsics = load_intrinsics(load_dir, clip_name, cam_id=cam_id)
#     image = undistort_image(image, intrinsics, distortions, new_intrinsics)

#     # 全白图像
#     distortion_mask = Image.new('L', IMAGE_SIZE, 255)
#     distortion_mask = undistort_image(distortion_mask, intrinsics, distortions, new_intrinsics)

#     # 反色
#     distortion_mask = ImageChops.invert(distortion_mask)

#     # 求和
#     image = ImageChops.add(image, distortion_mask)

#     image.save('data/ego_masks/chery/0.png')

#     print("Done!")


def generate_ego_mask(load_dir, clip_name):
    cam_id = 0

    mask_lagacy_path = "data/ego_masks/chery/unprocessed/0.png"
    image = Image.open(mask_lagacy_path).convert("L")

    intrinsics, distortions, new_intrinsics = load_intrinsics(
        load_dir, clip_name, cam_id=cam_id
    )

    distortion_mask = Image.new("L", image.size, 255)
    distortion_mask = undistort_image(
        distortion_mask, intrinsics, distortions, new_intrinsics
    )
    # 反色
    distortion_mask = ImageChops.invert(distortion_mask)

    # 求和
    image = ImageChops.add(image, distortion_mask)

    image.save("data/ego_masks/chery/0.png")

    print("Done!")


if __name__ == "__main__":
    clip_name = "clip_1746752396800"
    filename = "20250429105611.yaml"

    load_dir = os.path.join("data/chery/raw")

    filepath = os.path.join(load_dir, clip_name, "ego_mask", filename)
    with open(filepath, "r") as file:
        yaml_data = file.read()

    mask_data = yaml.safe_load(yaml_data)

    generate_ego_mask(load_dir, clip_name)
