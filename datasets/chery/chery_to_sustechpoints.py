import numpy as np, argparse, os, shutil, json
from datasets.qcraft.qcraft_helpers import (
    OPENCV2DATASET,
    euler_to_transform_matrix,
)

CAMERA_PERSPECTIVE = [
    "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H110",  # 车内
    "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H60",
    "CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H30",
    "CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H15",
    # "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_LEFT_H60",
    "CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99",
    "CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H99",
    "CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H30",
    # "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_RIGHT_H60",
    "CAM_PBQ_FRONT_RIGHT_RESET_OPTICAL_H99",
    "CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H99",
    "CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H30",
    "CAM_PBQ_REAR_RESET_OPTICAL_H50",
]

def organize_meta(args, all_time_stamps):
    # create dirs
    for cam_pers in CAMERA_PERSPECTIVE:
        create_dirs(os.path.join(args.dst, "camera", cam_pers))
    create_dirs(os.path.join(args.dst, "lidar"))
    create_dirs(os.path.join(args.dst, "label"))

    # copy image and lidar into dst path
    for ts in all_time_stamps:
        for file in os.listdir(os.path.join(args.src, ts)):
            if file.endswith(".jpg"):
                assert len(file.split("-")) == 3
                cam_id = file.split("-")[1]
                src_file = os.path.join(args.src, ts, file)
                dst_path = os.path.join(args.dst, "camera", cam_id)
                if os.path.exists(dst_path):
                    shutil.copy(src_file, os.path.join(dst_path, ts + ".jpg"))
            elif file.endswith(".pcd"):
                src_file = os.path.join(args.src, ts, file)
                dst_path = os.path.join(args.dst, "lidar")
                if os.path.exists(dst_path):
                    shutil.copy(src_file, os.path.join(dst_path, ts + ".pcd"))

def organize_calibs(args):
    # create dirs
    create_dirs(os.path.join(args.dst, "calib/camera"))

    raw_cam_calib_file_pth = os.path.join(args.src, "camera_params.json")
    if not os.path.exists(raw_cam_calib_file_pth):
        raise FileNotFoundError("cam params file not find")
    with open(raw_cam_calib_file_pth, encoding="utf-8") as f:
        cam_params = json.load(f)

    # cal lidar to cam extrinsics
    for cam_id in CAMERA_PERSPECTIVE:
        curr_cam_params = cam_params.get(cam_id, None)
        if curr_cam_params is None:
            continue

        extrinsics_raw = curr_cam_params.get("camera_to_vehicle_extrinsics", False)
        cam2ego = euler_to_transform_matrix(
            extrinsics_raw["x"],
            extrinsics_raw["y"],
            extrinsics_raw["z"],
            extrinsics_raw["yaw"],
            extrinsics_raw["pitch"],
            extrinsics_raw["roll"],
        )
        cam2ego = cam2ego @ OPENCV2DATASET

        intrinsics_raw = curr_cam_params.get("intrinsics", False)
        fx = intrinsics_raw["fx"]
        fy = intrinsics_raw["fy"]
        cx = intrinsics_raw["cx"]
        cy = intrinsics_raw["cy"]
        intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        info = {
            "extrinsic": np.linalg.inv(cam2ego).flatten().tolist(),
            "intrinsic": intrinsics.flatten().tolist(),
        }

        dst_file_path = os.path.join(args.dst, "calib/camera", cam_id + ".json")
        with open(dst_file_path, "w", encoding="utf-8") as f:
            json.dump(info, f)
            print(f"save {dst_file_path}")

def create_dirs(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, mode=0o777, exist_ok=False)


def convert2sus(args):
    """
    sus标注目录:
     - camera
        - front
            - timestamp.jpg
        - left
            - timestamp.jpg
     - calib
        -camera
            - front.json
            - left.json
     - lidar
        - timestamp.pcd
     - label
        - timestamp.json
    """
    assert os.path.exists(args.src), "source clip path is not exists!"

    all_time_stamps_list = [
        f for f in os.listdir(args.src)
        if os.path.isdir(os.path.join(args.src, f))
    ]

    organize_calibs(args)
    organize_meta(args, all_time_stamps_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("format qcraft data into sustechpoints")
    parser.add_argument("--src", help="qcraft clip path", type=str)
    parser.add_argument(
        "--dst", help="path to save organized sus-format data", type=str
    )

    args = parser.parse_args()
    convert2sus(args)

"""
python datasets/chery/chery_to_sustechpoints.py --src ~/Downloads/场景重建/qcraft_0702/20250702_133223_Q2517_60_75 --dst ~/Tools/SUSTechPOINTS/data/qcraft_test_1215
"""
