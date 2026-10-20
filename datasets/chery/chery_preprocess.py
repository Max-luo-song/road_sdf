import json
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml
from typing import List, Dict, Tuple
from dataclasses import dataclass

from datasets.tools.multiprocess_utils import track_parallel_progress
from datasets.dataset_meta import DATASETS_CONFIG
from chery_tools.parse_lidar import parse_lidar_pcd_file
from .chery_utils import (
    project_points_to_image,
    draw_and_fill_box,
    filter_points_in_box,
    preprocess_lidar_point_cloud,
)

from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from scipy.spatial.transform import Rotation

CHERY_CLASSES = ["unknown", "Vehicle", "Pedestrian", "Sign", "Cyclist"]
# TODO(ziyu): consider all dynamic classes
CHERY_DYNAMIC_CLASSES = ["Vehicle", "Pedestrian", "Cyclist"]
CHERY_HUMAN_CLASSES = ["Pedestrian", "Cyclist"]
CHERY_VEHICLE_CLASSES = ["Vehicle"]


@dataclass
class ClipDataCache:
    clip_dir: str
    label_data: Dict
    sample_names: List[str]
    lidar2worlds: List[np.ndarray]
    cam2lidars: List[np.ndarray]
    intrinsics_matrix: List[np.ndarray]
    intrinsics_list: List[List[float]]


class CheryProcessor(object):
    """Process Chery dataset.

    Args:
        load_dir (str): Directory to load chery raw data.
        save_dir (str): Directory to save data in KITTI format.
        prefix (str): Prefix of filename.
        workers (int, optional): Number of workers for the parallel process.
            Defaults to 64.
            Defaults to False.
        save_cam_sync_labels (bool, optional): Whether to save cam sync labels.
            Defaults to True.
    """

    def __init__(
        self,
        load_dir,
        save_dir,
        prefix,
        process_keys=[
            "images",
            "lidar",
            "calib",
            "pose",
            "dynamic_masks",
            "objects",
            "lidar_velocities",
        ],
        process_id_list=None,
        workers=64,
    ):
        # self.filter_no_label_zone_points = True

        # # Only data collected in specific locations will be converted
        # # If set None, this filter is disabled
        # # Available options: location_sf (main dataset)
        # self.selected_chery_locations = None
        # self.save_track_id = False

        self.process_id_list = process_id_list
        self.process_keys = process_keys
        print("will process keys: ", self.process_keys)

        self.load_dir = load_dir
        self.save_dir = f"{save_dir}/{prefix}"
        self.workers = int(workers)

        self.pinhole_camera_ids = [
            cam_id
            for cam_id, cam_info in DATASETS_CONFIG["chery"].items()
            if cam_info["is_fisheye"] == False
        ]
        self.all_camera_ids = [cam_id for cam_id in DATASETS_CONFIG["chery"]]

        self.create_folder()

        self._cache: Dict[str, ClipDataCache] = {}

    def convert(self):
        """Convert action."""

        if self.process_id_list is None:
            print("No process_id_list is provided.")

        print("Start converting ...")
        track_parallel_progress(
            self.convert_one,
            self.process_id_list,
            self.workers,
        )
        print("\nFinished ...")

    def convert_one(self, clip_name):
        """Convert action for single file."""

        self._cache[clip_name] = {}

        clip_dir = os.path.join(self.load_dir, clip_name)
        label_data_path = os.path.join(
            clip_dir, f"dynamic_obj/autolabel_10hz/{clip_name}.json"
        )
        with open(label_data_path, "r") as f:
            label_data = json.load(f)

        sample_names = [item["frame_name"] for item in label_data["frames"]]

        cam2lidars = self._parse_extrinsics(clip_dir)
        intrinsics_matrix, intrinsics_list = self._parse_intrinsics(label_data)

        lidar2worlds = [
            np.array(frame_data["lidar_pose"]) for frame_data in label_data["frames"]
        ]

        self._cache[clip_name] = ClipDataCache(
            clip_dir=clip_dir,
            label_data=label_data,
            sample_names=sample_names,
            lidar2worlds=lidar2worlds,
            cam2lidars=cam2lidars,
            intrinsics_matrix=intrinsics_matrix,
            intrinsics_list=intrinsics_list,
        )

        if "images" in self.process_keys:
            self.save_image(clip_name)
            print(f"Processed images for {clip_name}")

        if "calib" in self.process_keys:
            self.save_calib(clip_name)
            print(f"Processed calib for {clip_name}")

        if "lidar_velocities" in self.process_keys:
            self.save_lidar_velocities(clip_name)
            print(f"Processed lidar velocities for {clip_name}")

        if "lidar" in self.process_keys:
            self.save_lidar(clip_name)
            print(f"Processed lidar for {clip_name}")

        if "pose" in self.process_keys:
            self.save_pose(clip_name)
            print(f"Processed lidar poses for {clip_name}")

        if "dynamic_masks" in self.process_keys:
            self.save_dynamic_mask(clip_name)
            print(f"Processed dynamic masks for scene {clip_name}")

        if "objects" in self.process_keys:
            instances_info, frame_instances = self.save_objects(clip_name)
            print(f"Processed objects for scene {clip_name}")

            # Save instances info and frame instances
            object_info_dir = f"{self.save_dir}/{str(clip_name).zfill(3)}/instances"
            with open(f"{object_info_dir}/instances_info.json", "w") as fp:
                json.dump(instances_info, fp, indent=4)
            with open(f"{object_info_dir}/frame_instances.json", "w") as fp:
                json.dump(frame_instances, fp, indent=4)

    def __len__(self):
        """Length of the filename list."""
        return len(self.process_id_list)

    def save_image(self, clip_name):
        """保存经过畸变校正的相机图像为 jpg 格式"""
        cache = self._cache[clip_name]

        img_paths = []
        save_paths = []

        for frame_idx, sample_name in enumerate(cache.sample_names):
            sample_dir = os.path.join(cache.clip_dir, sample_name)
            img_names = [
                img_name  # camera0_1746752396953826_ae8a9af423443189bab574a9b3904cee_undist.jpg
                for img_name in os.listdir(sample_dir)
                if img_name.endswith(".jpg")
            ]

            for img_name in img_names:
                # 求出相机编号
                cam_idx = int(img_name.split("_")[0][6:])  # 0-10
                assert (
                    cam_idx in self.all_camera_ids
                ), f"Invalid camera index: {cam_idx}"

                # NOTE(syc): 针孔相机使用去畸变图像，鱼眼相机使用原始图像
                if (
                    cam_idx in self.pinhole_camera_ids and "_undist" not in img_name
                ):  # 跳过畸变针孔图像
                    continue

                img_path = os.path.join(sample_dir, img_name)
                img_paths.append(img_path)

                save_name = f"{frame_idx:03}_{cam_idx}.jpg"  # 000_0, 000_1, ...
                save_path = os.path.join(self.save_dir, clip_name, "images", save_name)
                save_paths.append(save_path)

        for img_path, save_path in tqdm(
            zip(img_paths, save_paths), total=len(img_paths)
        ):
            Image.open(img_path).save(save_path)

    def save_calib(self, clip_name):
        """解析并保存相机的外参（以 Front main LiDAR 为参考系）和内参"""
        cache = self._cache[clip_name]

        for cam_idx, (cam2lidar, intrinsic) in tqdm(
            enumerate(zip(cache.cam2lidars, cache.intrinsics_list)),
            total=len(cache.cam2lidars),
        ):
            np.savetxt(
                f"{self.save_dir}/{clip_name}/extrinsics/{cam_idx}.txt",
                cam2lidar,
            )
            np.savetxt(
                f"{self.save_dir}/{clip_name}/intrinsics/{cam_idx}.txt",
                intrinsic,
            )

    def save_lidar(self, clip_name):
        """
        将雷达数据从 pcd 格式转换到 bin 格式

        二进制文件中包含 7 个 float32 字段：
            x y z intensity timestamp ring lidar_id
        """
        cache = self._cache[clip_name]

        # 检查是否有经过运动补偿的 LiDAR
        mc_lidar_paths = {}
        lidar_slam_info_path = os.path.join(
            cache.clip_dir, "static_obj/lidar_slam/lidar_slam_info.json"
        )
        if os.path.exists(lidar_slam_info_path):
            print("Found mc_pcds for lidar 0")

            with open(lidar_slam_info_path, "r") as f:
                lidar_slam_info = json.load(f)

            mc_lidar_paths = {
                info["frame_name"]: os.path.join(cache.clip_dir, info["mc_lidar"])
                for info in lidar_slam_info["lidar0"]["mc_pcds"]
            }
        else:
            print("No mc_pcds found for lidar 0")

        for frame_idx, sample_name in tqdm(
            enumerate(cache.sample_names), total=len(cache.sample_names)
        ):
            sample_dir = os.path.join(cache.clip_dir, sample_name)

            lidar_names = [
                lidar_name  # lidar0_1746752396900110_2e4bd97ab9d533658a28cfbe1581df57.pcd
                for lidar_name in os.listdir(sample_dir)
                if lidar_name.endswith(".pcd")
                and not lidar_name.startswith("lidar5")  # 排除5号雷达（m2雷达）
            ]
            lidar_names = sorted(
                lidar_names, key=lambda name: int(name.split("_")[0][5:])
            )
            lidar_ids = [int(name.split("_")[0][5:]) for name in lidar_names]
            lidar_paths = [
                os.path.join(sample_dir, lidar_name) for lidar_name in lidar_names
            ]

            point_cloud_list = [
                parse_lidar_pcd_file(lidar_path) for lidar_path in lidar_paths
            ]  # x y z intensity timestamp ring
            point_cloud_list = [
                preprocess_lidar_point_cloud(pc, lidar_id)
                for lidar_id, pc in zip(lidar_ids, point_cloud_list)
            ]  # x y z intensity timestamp ring lidar_id
            point_cloud = np.concatenate(point_cloud_list, axis=0)

            bin_path = (
                f"{self.save_dir}/{clip_name}/lidar/{str(frame_idx).zfill(3)}.bin"
            )
            point_cloud.tofile(bin_path)

            mc_lidar_path = mc_lidar_paths.get(sample_name, None)
            if mc_lidar_path is not None:
                mc_lidar_point_cloud = parse_lidar_pcd_file(mc_lidar_path)
                mc_lidar_point_cloud = preprocess_lidar_point_cloud(
                    mc_lidar_point_cloud,
                    lidar_id=0,
                )
                # 替换 LiDAR 0
                point_cloud_list[0] = mc_lidar_point_cloud
                mc_point_cloud = np.concatenate(point_cloud_list, axis=0)

                mc_bin_path = (
                    f"{self.save_dir}/{clip_name}/mclidar/{str(frame_idx).zfill(3)}.bin"
                )
                mc_point_cloud.tofile(mc_bin_path)

    def save_pose(self, clip_name):
        """保存每一帧的位姿"""
        # FIXME: 由于目前奇瑞数据无法正确计算 ego pose，因此改为保存 lidar pose，后续找时间修正

        cache = self._cache[clip_name]
        for frame_idx, lidar2world in tqdm(
            enumerate(cache.lidar2worlds), total=len(cache.lidar2worlds)
        ):
            np.savetxt(
                f"{self.save_dir}/{clip_name}/lidar_pose/{str(frame_idx).zfill(3)}.txt",
                lidar2world,
            )

    def save_lidar_velocities(self, clip_name):
        cache = self._cache[clip_name]

        lidar_velos = [
            np.array(frame_data["lidar_velo"])
            for frame_data in cache.label_data["frames"]
        ]
        for frame_idx, lidar_velo in tqdm(
            enumerate(lidar_velos), total=len(lidar_velos)
        ):
            np.savetxt(
                f"{self.save_dir}/{clip_name}/lidar_velocities/{str(frame_idx).zfill(3)}.txt",
                lidar_velo,
            )

    def _get_obj_pose_and_size(
        self, obj_data
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        返回 obj2lidar, (l, w, h)
        """
        rotation = np.array(obj_data["obj_rotation"])
        position = np.array(obj_data["obj_center_pos"])

        obj2lidar = np.eye(4, dtype=np.float64)
        obj2lidar[:3, :3] = Rotation.from_quat(rotation).as_matrix()
        obj2lidar[:3, 3] = position

        l, w, h = obj_data["size"]
        return obj2lidar, (l, w, h)

    def _generate_obj_masks(self, obj_data, masks, lidar2cams, intrinsics):
        """
        处理单个动态物体，生成掩码图像
        """
        obj2lidar, (l, w, h) = self._get_obj_pose_and_size(obj_data)

        # # 1.对任何一个物体，先对应到激光雷达点云
        # mask_pointcloud = filter_points_in_box(pointcloud, obj_center_pos, size)
        # # 2.把激光点云向7个视角均投影得到 uv
        # # 3.七个视角最终全部投影得到结果
        for cam_idx, lidar2cam in enumerate(lidar2cams):
            obj2cam = lidar2cam @ obj2lidar
            box = Box(
                center=obj2cam[:3, 3],
                size=[w, l, h],
                orientation=Quaternion(matrix=obj2cam[:3, :3]),
            )
            corners_cam = box.corners().T.astype(np.float32)
            corners_2d = project_points_to_image(
                corners_cam,
                intrinsics[cam_idx],
            )
            masks[cam_idx] = draw_and_fill_box(masks[cam_idx], corners_2d)

        return masks

    def save_dynamic_mask(self, clip_name):
        """
        将 box 投影到图像平面上，获取 2D mask，包括 all human vehicle 三种
        """
        cache = self._cache[clip_name]

        # 创建保存目录
        categories = ["all", "human", "vehicle"]
        for category in categories:
            mask_dir = f"{self.save_dir}/{clip_name}/dynamic_masks/{category}"
            if not os.path.exists(mask_dir):
                os.makedirs(mask_dir)

        img_shapes = [
            config["original_size"] for _, config in DATASETS_CONFIG["chery"].items()
        ]

        lidar2cams = [np.linalg.inv(cam2lidar) for cam2lidar in cache.cam2lidars]

        # 处理每一帧
        for frame_idx, frame_data in tqdm(
            enumerate(cache.label_data["frames"]),
            total=len(
                cache.label_data["frames"],
            ),
        ):
            # 初始化掩码图像
            masks_vehicle = [
                np.zeros((sz[0], sz[1], 3), dtype=np.uint8) for sz in img_shapes
            ]
            masks_human = [
                np.zeros((sz[0], sz[1], 3), dtype=np.uint8) for sz in img_shapes
            ]
            masks_all = [
                np.zeros((sz[0], sz[1], 3), dtype=np.uint8) for sz in img_shapes
            ]

            # 处理物体检测信息
            anns_info = frame_data["annotated_info"][
                "3d_city_object_detection_annotated_info"
            ]["annotated_info"]["3d_object_detection_info"][
                "3d_object_detection_anns_info"
            ]

            for obj_data in anns_info:
                category = obj_data["category"]

                # 类别对应关系
                # [Chery] -> [Waymo]
                # "person" -> "human"
                # 其余类别 -> "vehicle"
                if category == "person":
                    masks_human = self._generate_obj_masks(
                        obj_data,
                        masks_human,
                        lidar2cams,
                        cache.intrinsics_matrix,
                    )
                else:  # vehicle
                    masks_vehicle = self._generate_obj_masks(
                        obj_data,
                        masks_vehicle,
                        lidar2cams,
                        cache.intrinsics_matrix,
                    )

            for cam_idx in range(len(lidar2cams)):
                # 将 vehicle 和 human 掩码合并到 all 掩码中
                masks_all[cam_idx] = np.maximum(
                    masks_human[cam_idx], masks_vehicle[cam_idx]
                )

            # 保存掩码图像
            for category, masks in zip(
                categories, [masks_all, masks_human, masks_vehicle]
            ):
                for cam_idx, mask in enumerate(masks):
                    mask_gray = Image.fromarray(mask).convert("L")
                    mask_path = os.path.join(
                        f"{self.save_dir}/{clip_name}/dynamic_masks/{category}",
                        f"{str(frame_idx).zfill(3)}_{str(cam_idx)}.png",
                    )
                    mask_gray.save(mask_path)

    def save_objects(self, clip_name):
        """
        生成instances相关的json文件
        frame_instances是帧到实例的映射
        instances_info是实例到属性的映射
        """
        cache = self._cache[clip_name]

        instances_info, frame_instances = {}, {}

        """frame_instances.json"""
        for frame_idx, frame_data in tqdm(
            enumerate(cache.label_data["frames"]),
            total=len(
                cache.label_data["frames"],
            ),
        ):
            anns_info = frame_data["annotated_info"][
                "3d_city_object_detection_annotated_info"
            ]["annotated_info"]["3d_object_detection_info"][
                "3d_object_detection_anns_info"
            ]

            """
            instances_info = {
                "0": # simplified instance id
                    {
                        "id": str,
                        "class_name": str,
                        "frame_annotations": {
                            "frame_idx": List,
                            "obj_to_world": List,
                            "box_size": List,
                    },
                ...
            }
            """
            for obj_data in anns_info:
                track_id = str(obj_data["track_id"])
                is_cyclist = obj_data["is_cyclist"]

                if track_id not in instances_info:
                    category = obj_data.get("category")
                    if category == "person":
                        class_name = "Pedestrian"
                    else:
                        if is_cyclist is True:
                            class_name = "Cyclist"
                        else:
                            class_name = "Vehicle"

                    instances_info[track_id] = {
                        "class_name": class_name,
                        "id": track_id,
                        "frame_annotations": {
                            "frame_idx": [],
                            "obj_to_world": [],
                            "box_size": [],
                        },
                    }

                obj2lidar, box_size = self._get_obj_pose_and_size(obj_data)
                obj2world = cache.lidar2worlds[frame_idx] @ obj2lidar

                instances_info[track_id]["frame_annotations"]["frame_idx"].append(
                    frame_idx
                )
                instances_info[track_id]["frame_annotations"]["obj_to_world"].append(
                    obj2world.tolist()
                )
                instances_info[track_id]["frame_annotations"]["box_size"].append(
                    box_size
                )

            """
            frame_instances = {
                "0": # frame idx
                    List[int] # list of simplified instance ids
                ...
            }
            """
            frame_instances[str(frame_idx)] = [
                int(obj_data["track_id"]) for obj_data in anns_info
            ]

        return instances_info, frame_instances

    def create_folder(self):
        """Create folder for data preprocessing."""

        for clip_name in self.process_id_list:
            if "images" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(clip_name)}/images", exist_ok=True)
                os.makedirs(
                    f"{self.save_dir}/{str(clip_name)}/sky_masks", exist_ok=True
                )
            if "calib" in self.process_keys:
                os.makedirs(
                    f"{self.save_dir}/{str(clip_name)}/extrinsics", exist_ok=True
                )
                os.makedirs(
                    f"{self.save_dir}/{str(clip_name)}/intrinsics", exist_ok=True
                )
            if "pose" in self.process_keys:
                os.makedirs(
                    f"{self.save_dir}/{str(clip_name)}/lidar_pose", exist_ok=True
                )
            if "lidar_velocities" in self.process_keys:
                os.makedirs(
                    f"{self.save_dir}/{str(clip_name)}/lidar_velocities", exist_ok=True
                )
            if "lidar" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(clip_name)}/lidar", exist_ok=True)

                lidar_slam_info_path = os.path.join(
                    self.load_dir,
                    clip_name,
                    "static_obj/lidar_slam/lidar_slam_info.json",
                )
                if os.path.exists(lidar_slam_info_path):
                    os.makedirs(
                        f"{self.save_dir}/{str(clip_name)}/mclidar", exist_ok=True
                    )

            if "dynamic_masks" in self.process_keys:
                os.makedirs(
                    f"{self.save_dir}/{str(clip_name)}/dynamic_masks", exist_ok=True
                )
            if "objects" in self.process_keys:
                os.makedirs(
                    f"{self.save_dir}/{str(clip_name)}/instances", exist_ok=True
                )

    def _parse_extrinsics(self, clip_dir):
        extrinsics = []

        extrinsics_dir = os.path.join(clip_dir, "extrinsics", "lidar2camera")
        filenames = [
            # 针孔相机
            "lidar2frontwide.yaml",
            "lidar2frontmain.yaml",
            "lidar2leftfront.yaml",
            "lidar2leftrear.yaml",
            "lidar2rightfront.yaml",
            "lidar2rightrear.yaml",
            "lidar2rearmain.yaml",
            # 鱼眼相机
            "lidar2fisheyeleft.yaml",
            "lidar2fisheyerear.yaml",
            "lidar2fisheyefront.yaml",
            "lidar2fisheyeright.yaml",
        ]
        for filename in filenames:
            with open(
                os.path.join(extrinsics_dir, filename), "r", encoding="utf-8"
            ) as f:
                extrinsic = yaml.safe_load(f)

            lidar2cam = np.array(extrinsic["transform"])
            cam2lidar = np.linalg.inv(lidar2cam)
            extrinsics.append(cam2lidar)

        return extrinsics

    def _parse_intrinsics(self, label_data):
        intrinsics_matrix, intrinsics_list = [], []
        for cam_idx in self.all_camera_ids:
            cam_params = label_data["calibration"][f"camera{cam_idx}"]

            # # 原始相机内参
            # intrinsic = params["intrinsic"]
            # fx = intrinsic[0][0]
            # cx = intrinsic[0][2]
            # fy = intrinsic[1][1]
            # cy = intrinsic[1][2]
            # intrinsic = [fx, fy, cx, cy]

            # 畸变参数
            distcoeff = cam_params["distcoeff"][0]

            # 去畸变后新的相机内参
            intrinsic_scaled = cam_params["intrinsic_scaled"]
            fx_scaled = intrinsic_scaled[0][0]
            cx_scaled = intrinsic_scaled[0][2]
            fy_scaled = intrinsic_scaled[1][1]
            cy_scaled = intrinsic_scaled[1][2]
            intrinsic_scaled = [fx_scaled, fy_scaled, cx_scaled, cy_scaled]

            matrix_values = np.array(
                [
                    [fx_scaled, 0, cx_scaled],
                    [0, fy_scaled, cy_scaled],
                    [0, 0, 1],
                ]
            )
            list_values = intrinsic_scaled + distcoeff

            intrinsics_matrix.append(matrix_values)
            intrinsics_list.append(list_values)

        return intrinsics_matrix, intrinsics_list
