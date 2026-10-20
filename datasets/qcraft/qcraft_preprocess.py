import json
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Tuple, AnyStr
from dataclasses import dataclass
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box

from datasets.tools.multiprocess_utils import track_parallel_progress

from datasets.utils.pcd_utils import storePly
from datasets.utils.box_utils import bbox_to_corner3d, inbbox_points
from datasets.utils.img_utils import visualize_depth_numpy


from chery_tools.parse_lidar import parse_lidar_pcd_file
from .qcraft_config import CamSpec, LIDAR_CANDIDATES, ALL_CAM_SPECS, SIDE_WIDE_KEYS

from .qcraft_helpers import (
    OPENCV2DATASET,
    draw_and_fill_box,
    euler_to_rotation_matrix,
    euler_to_transform_matrix,
    project_label_to_mask,
    project_points_to_image,
    convert_raw_object_type_to_class_name,
)

QCRAFT_CLASSES = ["unknown", "Vehicle", "Pedestrian", "Sign", "Cyclist", "TrafficBarrier", "TrafficLight"]
# TODO(ziyu): consider all dynamic classes
QCRAFT_DYNAMIC_CLASSES = ["Vehicle", "Pedestrian", "Cyclist", "TrafficBarrier", "TrafficLight"]
QCRAFT_TRAFFIC_LIGHT_CLASSES = ["TrafficBarrier", "TrafficLight"]
QCRAFT_HUMAN_CLASSES = ["Pedestrian", "Cyclist"]
QCRAFT_VEHICLE_CLASSES = ["Vehicle"]


@dataclass(frozen=True)
class SceneData:
    clip_dir: str
    frame_timestamps: List[str]
    ego_masks: Dict[int, np.ndarray]
    images: List[Dict[int, List]]
    ego_pose_data: List[np.ndarray]
    cam2egos: Dict[int, np.ndarray]
    intrinsics_matrix: Dict[int, np.ndarray]
    intrinsics_params: Dict[int, List[float]]
    lidar2ego: np.ndarray
    camera_specs: Dict[int, CamSpec]  # cam_id -> CamSpec
    original_cam_name_to_cam_id: Dict[str, int]
    main_lidar_name: str


class QcraftProcessor(object):
    """Process Qcraft dataset.

    Args:
        load_dir (str): Directory to load Qcraft raw data.
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
        ],
        process_id_list=None,
        workers=64,
        skip_front_wide_side_cameras=False,
        label_type="pred",
    ):
        # self.filter_no_label_zone_points = True

        # # Only data collected in specific locations will be converted
        # # If set None, this filter is disabled
        # # Available options: location_sf (main dataset)
        # self.selected_qcraft_locations = None
        # self.save_track_id = False

        self.process_id_list = process_id_list
        assert self.process_id_list is not None, "No process_id_list is provided."

        self.process_keys = process_keys
        print("will process keys: ", self.process_keys)

        self.load_dir = load_dir

        self.save_dir = f"{save_dir}/{prefix}"
        self.workers = int(workers)

        self.skip_front_wide_side_cameras = skip_front_wide_side_cameras

        # 读取标签类型，决定标签文件夹名称
        self.label_dir = "label_pred" if label_type == "pred" else "label"
        print(f"Using label directory: {self.label_dir}")

    def convert(self):
        """Convert action."""
        print("Start converting ...")
        track_parallel_progress(
            self.convert_one,
            self.process_id_list,
            self.workers,
        )
        print("\nFinished ...")

    def convert_one(self, clip_name):
        """Convert action for single file."""
        scene_data = self._load_scene_data(clip_name)
        clip_save_dir = os.path.join(self.save_dir, clip_name)
        os.makedirs(clip_save_dir, exist_ok=True)

        # Sensor info
        sensor_info_path = os.path.join(clip_save_dir, "sensor_info.json")
        with open(sensor_info_path, "w") as f:
            sensor_info = {
                "camera_specs": {
                    cam_id: {
                        "key": cam_spec.key,
                        "name": cam_spec.name,
                        "width": cam_spec.width,
                        "height": cam_spec.height,
                    }
                    for cam_id, cam_spec in scene_data.camera_specs.items()
                },
                "main_lidar_name": scene_data.main_lidar_name,
            }
            json.dump(sensor_info, f, indent=4)

        if "ego_masks" in self.process_keys:
            self.save_ego_masks(scene_data, clip_save_dir)
            print(f"Processed ego masks for {clip_name}")

        if "images" in self.process_keys:
            self.save_image(scene_data, clip_save_dir)
            print(f"Processed images for {clip_name}")

        if "calib" in self.process_keys:
            self.save_calib(scene_data, clip_save_dir)
            print(f"Processed calib for {clip_name}")

        if "pose" in self.process_keys:
            self.save_pose(scene_data, clip_save_dir)
            print(f"Processed lidar poses for {clip_name}")

        if "objects" in self.process_keys:
            self.save_objects(scene_data, clip_save_dir)
            print(f"Processed objects for scene {clip_name}")

        if "dynamic_masks" in self.process_keys:
            self.save_dynamic_mask(scene_data, clip_save_dir)
            print(f"Processed dynamic masks for scene {clip_name}")

        if "lidar" in self.process_keys:
            self.save_lidar(scene_data, clip_save_dir)
            print(f"Processed lidar for {clip_name}")

        if "mix_novel_views" in self.process_keys:
            novel_view_dirs = f"{self.save_dir}/{clip_name}/novel_views"
            assert os.path.exists(novel_view_dirs), "novel_views is not exist"
            self.save_novel_view_infos(scene_data, clip_name)
            print(f"Processed novel_view for {clip_name}")

    def __len__(self):
        """Length of the filename list."""
        return len(self.process_id_list)

    def save_ego_masks(self, scene_data: SceneData, clip_save_dir: str):
        ego_mask_dst_dir = os.path.join(clip_save_dir, "ego_masks")
        os.makedirs(ego_mask_dst_dir, exist_ok=True)

        for cam_id, ego_mask in scene_data.ego_masks.items():
            dst_path = os.path.join(ego_mask_dst_dir, f"{cam_id}.png")
            Image.fromarray(ego_mask).save(dst_path)

    def save_image(self, scene_data: SceneData, clip_save_dir: str):
        """保存经过畸变校正的相机图像为 png 格式"""
        image_save_dir = os.path.join(clip_save_dir, "images")
        os.makedirs(image_save_dir, exist_ok=True)

        camera_timestamp = dict()
        for frame_idx, cam_images in tqdm(
            enumerate(scene_data.images), total=len(scene_data.images)
        ):
            for cam_idx, img in cam_images.items():
                # Save image
                save_path = os.path.join(
                    image_save_dir, f"{frame_idx:06d}_{cam_idx}.png"
                )  # 000000_0, 000000_1, ...
                img.save(save_path)

                # Timestamp
                camera_timestamp[f"{frame_idx:06d}_{cam_idx}"] = (
                    scene_data.frame_timestamps[frame_idx]
                )

        # Save timestamps
        camera_timestamp_save_path = os.path.join(image_save_dir, "timestamps.json")
        with open(camera_timestamp_save_path, "w") as f:
            json.dump(camera_timestamp, f, indent=2)

    def save_calib(self, scene_data: SceneData, clip_save_dir: str):
        """保存以LiDAR坐标系为参考系的相机参数"""
        extrinsics_save_dir = os.path.join(clip_save_dir, "extrinsics")
        intrinsics_save_dir = os.path.join(clip_save_dir, "intrinsics")
        os.makedirs(extrinsics_save_dir, exist_ok=True)
        os.makedirs(intrinsics_save_dir, exist_ok=True)

        ego2lidar = np.linalg.inv(scene_data.lidar2ego)
        for cam_id, cam2ego in tqdm(
            scene_data.cam2egos.items(), total=len(scene_data.cam2egos)
        ):
            cam2lidar = ego2lidar @ cam2ego
            intrinsic_param = scene_data.intrinsics_params[cam_id]
            np.savetxt(os.path.join(extrinsics_save_dir, f"{cam_id}.txt"), cam2lidar)
            np.savetxt(
                os.path.join(intrinsics_save_dir, f"{cam_id}.txt"), intrinsic_param
            )

    def save_pose(self, scene_data: SceneData, clip_save_dir: str):
        """保存每一帧的位姿"""
        # NOTE(syc): 位姿都以世界坐标系为参考系
        ego_pose_dir = os.path.join(clip_save_dir, "ego_pose")
        lidar_pose_dir = os.path.join(clip_save_dir, "lidar_pose")
        os.makedirs(ego_pose_dir, exist_ok=True)
        os.makedirs(lidar_pose_dir, exist_ok=True)

        timestamp = dict()
        timestamp["FRAME"] = dict()
        for _, cam in scene_data.camera_specs.items():
            timestamp[cam.name] = dict()

        for frame_idx, ego_pose_data in tqdm(
            enumerate(scene_data.ego_pose_data), total=len(scene_data.ego_pose_data)
        ):
            ego_pose = ego_pose_data["FRAME"]["pose"]
            lidar_pose = ego_pose @ scene_data.lidar2ego

            filename = f"{frame_idx:06d}.txt"
            np.savetxt(os.path.join(ego_pose_dir, filename), ego_pose)
            np.savetxt(os.path.join(lidar_pose_dir, filename), lidar_pose)

            timestamp["FRAME"][f"{frame_idx:06d}"] = ego_pose_data["FRAME"]["timestamp"]

            for cam_id, cam in scene_data.camera_specs.items():
                timestamp[cam.name][f"{frame_idx:06d}"] = ego_pose_data[cam.name][
                    "timestamp"
                ]
                np.savetxt(
                    os.path.join(ego_pose_dir, f"{frame_idx:06d}_{cam_id}.txt"),
                    ego_pose_data[cam.name]["pose"],
                )

        timestamp_save_path = os.path.join(clip_save_dir, "timestamps.json")
        with open(timestamp_save_path, "w") as f:
            json.dump(timestamp, f, indent=4)

    def save_objects(self, scene_data: SceneData, clip_save_dir: str):
        """
        生成instances相关的json文件
        frame_instances是帧到实例的映射
        instances_info是实例到属性的映射
        """
        object_info_save_dir = os.path.join(clip_save_dir, "instances")
        os.makedirs(object_info_save_dir, exist_ok=True)

        instances_info = {}
        for frame_idx, frame_timestamp in tqdm(
            enumerate(scene_data.frame_timestamps),
            total=len(scene_data.frame_timestamps),
        ):
            labels_path = os.path.join(scene_data.clip_dir, self.label_dir, f"{frame_timestamp}.json")
            with open(labels_path, "r") as f:
                obj_data_list = json.load(f)

            """
            instances_info = {
                "0": # simplified instance id
                    {
                        "id": str,
                        "class_name": str,
                        "frame_annotations": {
                            "frame_idx": List,
                            "obj_to_world": List,
                            "obj_to_ego": List,
                            "box_size": List,
                    },
                ...
            }
            """
            ego_pose = scene_data.ego_pose_data[frame_idx]["FRAME"]["pose"]

            for obj_data in obj_data_list:
                obj_id: str = obj_data["obj_id"]

                obj2ego, box_size = self._get_obj_pose_and_size(obj_data)
                obj2world: np.ndarray = ego_pose @ obj2ego

                if obj_id not in instances_info:
                    class_name = convert_raw_object_type_to_class_name(
                        obj_data["obj_type"]
                    )
                    instances_info[obj_id] = {
                        "class_name": class_name,
                        "id": obj_id,
                        "frame_annotations": {
                            "frame_idx": [],
                            "obj_to_world": [],
                            "obj_to_ego": [],
                            "box_size": [],
                        },
                    }
                frame_annotations = instances_info[obj_id]["frame_annotations"]
                frame_annotations["frame_idx"].append(frame_idx)
                frame_annotations["obj_to_ego"].append(obj2ego.tolist())
                frame_annotations["obj_to_world"].append(obj2world.tolist())
                frame_annotations["box_size"].append(box_size)

        """
        frame_instances = {
            "0": # frame idx
                List[int] # list of simplified instance ids
            ...
        }
        """
        frame_instances = {}
        for frame_idx, frame_timestamp in tqdm(
            enumerate(scene_data.frame_timestamps),
            total=len(scene_data.frame_timestamps),
        ):
            frame_instances[frame_idx] = []
            for k, v in instances_info.items():
                if frame_idx in v["frame_annotations"]["frame_idx"]:
                    frame_instances[frame_idx].append(v["id"])

        # correct id
        id_map = {}
        for i, (obj_id, info) in enumerate(instances_info.items()):
            id_map[info["id"]] = i

        # update keys in instances_info
        new_instances_info = {}
        for obj_id, info in instances_info.items():
            new_instances_info[id_map[info["id"]]] = info

        # update keys in frame_instances
        new_frame_instances = {}
        for obj_id, info in frame_instances.items():
            new_frame_instances[obj_id] = [id_map[i] for i in info]

        # Save instances info and frame instances
        with open(f"{object_info_save_dir}/instances_info.json", "w") as fp:
            json.dump(new_instances_info, fp, indent=4)

        with open(f"{object_info_save_dir}/frame_instances.json", "w") as fp:
            json.dump(new_frame_instances, fp, indent=4)

    def save_dynamic_mask(self, scene_data: SceneData, clip_save_dir: str):
        """
        将 box 投影到图像平面上，获取 2D mask，包括 all human vehicle 三种
        """
        dynamic_mask_dir = os.path.join(clip_save_dir, "dynamic_masks")

        # 创建保存目录
        categories = ["all", "human", "vehicle"]
        for category in categories:
            mask_dir = os.path.join(dynamic_mask_dir, category)
            os.makedirs(mask_dir, exist_ok=True)

        frame_instance_path = os.path.join(
            clip_save_dir, "instances", "frame_instances.json"
        )
        instances_info_path = os.path.join(
            clip_save_dir, "instances", "instances_info.json"
        )
        with open(frame_instance_path, "r") as f:
            frame_instances = json.load(f)
        with open(instances_info_path, "r") as f:
            instances_info = json.load(f)

        # 处理每一帧
        for frame_idx, frame_timestamp in tqdm(
            enumerate(scene_data.frame_timestamps),
            total=len(scene_data.frame_timestamps),
        ):
            visible_objects = frame_instances[str(frame_idx)]
            for cam_id, cam in scene_data.camera_specs.items():
                mask_vehicle = np.zeros((cam.height, cam.width), dtype=np.bool_)
                mask_human = np.zeros((cam.height, cam.width), dtype=np.bool_)

                for track_id in visible_objects:
                    info = instances_info[str(track_id)]
                    class_name = info["class_name"]
                    frame_annotations = info["frame_annotations"]
                    idx = frame_annotations["frame_idx"].index(frame_idx)
                    obj2ego = np.array(frame_annotations["obj_to_ego"][idx]).reshape(
                        (4, 4)
                    )
                    l, w, h = frame_annotations["box_size"][idx]

                    box_mask = project_label_to_mask(
                        dim=[l, w, h],
                        obj2ego=obj2ego,
                        cam2ego=scene_data.cam2egos[cam_id],
                        intrinsic=scene_data.intrinsics_matrix[cam_id],
                        img_shape=(cam.height, cam.width),
                    )
                    if class_name == "Pedestrian":
                        mask_human = np.logical_or(mask_human, box_mask)
                    else:
                        mask_vehicle = np.logical_or(mask_vehicle, box_mask)

                mask_all = np.logical_or(mask_human, mask_vehicle)

                # 保存掩码图像
                for category, mask in zip(
                    categories, [mask_all, mask_human, mask_vehicle]
                ):
                    mask_gray = Image.fromarray(mask.astype(np.uint8) * 255).convert(
                        "L"
                    )
                    mask_path = os.path.join(
                        dynamic_mask_dir, category, f"{frame_idx:06d}_{cam_id}.png"
                    )
                    mask_gray.save(mask_path)

    def save_lidar(self, scene_data: SceneData, clip_save_dir: str):
        """
        将雷达数据从 pcd 格式转换到 bin 格式

        二进制文件中包含 5 个 float32 字段：
            x y z intensity lidar_id
        """
        lidar_dir = os.path.join(clip_save_dir, "lidar")
        lidar_dir_bin = os.path.join(lidar_dir, "bin")
        lidar_dir_background = os.path.join(lidar_dir, "background")
        lidar_dir_actor = os.path.join(lidar_dir, "actor")
        lidar_dir_depth = os.path.join(lidar_dir, "depth")
        os.makedirs(lidar_dir_bin, exist_ok=True)
        os.makedirs(lidar_dir_background, exist_ok=True)
        os.makedirs(lidar_dir_actor, exist_ok=True)
        os.makedirs(lidar_dir_depth, exist_ok=True)

        frame_instance_path = os.path.join(
            clip_save_dir, "instances", "frame_instances.json"
        )
        instances_info_path = os.path.join(
            clip_save_dir, "instances", "instances_info.json"
        )
        with open(frame_instance_path, "r") as f:
            frame_instances = json.load(f)
        with open(instances_info_path, "r") as f:
            instances_info = json.load(f)

        # pointcloud_actor = dict()
        # for track_id, traj in trajectory.items():
        #     dynamic = not traj["stationary"]
        #     if dynamic and traj["label"] != "sign":
        #         actor_dir = os.path.join(lidar_dir_actor, track_id)
        #         os.makedirs(actor_dir, exist_ok=True)
        #         pointcloud_actor[track_id] = dict()
        #         pointcloud_actor[track_id]["xyz"] = []
        #         pointcloud_actor[track_id]["rgb"] = []
        #         pointcloud_actor[track_id]["mask"] = []
    
        # 自车区域
        half_l = 2.9
        half_w = 1.5

        for frame_idx, frame_timestamp in tqdm(
            enumerate(scene_data.frame_timestamps),
            total=len(scene_data.frame_timestamps),
        ):
            sample_dir = os.path.join(scene_data.clip_dir, frame_timestamp)

            # NOTE(syc): 目前只使用一个LiDAR，但保留了支持多个LiDAR的形式
            lidar_names = [
                lidar_name  # 20250702_133223_Q2517-LDR_FRONT-1751434420.2514-ego.pcd
                for lidar_name in os.listdir(sample_dir)
                if lidar_name.endswith(".pcd")
                and f"-{scene_data.main_lidar_name}-" in lidar_name
                and "LEFT" not in lidar_name
                and "RIGHT" not in lidar_name
            ]
            lidar_paths = [
                os.path.join(sample_dir, lidar_name) for lidar_name in lidar_names
            ]

            pc_ego_list = []
            pc_lidar_list = []
            for lidar_id, lidar_path in enumerate(lidar_paths):
                pc_ego = parse_lidar_pcd_file(lidar_path)  # x y z intensity
                pc_ego = np.stack(
                    [pc_ego[field].astype(np.float32) for field in pc_ego.dtype.names],
                    axis=1,
                )  # x y z intensity

                # 增加 lidar_id 列
                lidar_id_col = np.full((pc_ego.shape[0], 1), lidar_id, dtype=np.float32)  # [N, 1]
                pc_ego = np.hstack([pc_ego, lidar_id_col])  # x y z intensity lidar_id

                # 转换到 LiDAR 坐标系
                xyz_ego = pc_ego[:, :3]  # [N, 3]
                xyz_ego_homo = np.hstack([xyz_ego, np.ones((xyz_ego.shape[0], 1))])  # [N, 4]
                xyz_lidar_homo = (np.linalg.inv(scene_data.lidar2ego) @ xyz_ego_homo.T).T  # [N, 4]
                xyz_lidar = xyz_lidar_homo[:, :3]  # [N, 3]

                intensity = pc_ego[:, 3:4]
                pc_lidar = np.hstack([xyz_lidar, intensity, lidar_id_col])  # [N, 5]

                # 车身点
                egocar_mask = (np.abs(xyz_lidar[:, 0]) <= half_l) & (np.abs(xyz_lidar[:, 1]) <= half_w)
                # 地面以下的点
                below_ground_mask = xyz_ego[:, 2] < -3.0
                pointcloud_mask = ~egocar_mask & ~below_ground_mask

                pc_lidar = pc_lidar[pointcloud_mask]
                pc_lidar_list.append(pc_lidar)

                pc_ego = pc_ego[pointcloud_mask]
                pc_ego_list.append(pc_ego)
                # print(f"Filtered {egocar_mask.sum()} ego car points and {below_ground_mask.sum()} below ground points.")

            # 保存 LiDAR 二进制文件
            pc_lidar = np.concatenate(pc_lidar_list, axis=0)  # x y z intensity lidar_id
            pc_lidar = pc_lidar.astype(np.float32)
            bin_path = os.path.join(lidar_dir_bin, f"{frame_idx:06d}.bin")
            pc_lidar.tofile(bin_path)
            
            pc_ego = np.concatenate(pc_ego_list, axis=0)  # x y z intensity lidar_id
            xyzs_ego = pc_ego[:, :3]

            # 滤除更多无效点，生成lidar点云
            valid_global_mask = np.zeros(xyzs_ego.shape[0], dtype=bool)

            pcd_color = np.zeros((xyzs_ego.shape[0], 3), dtype=np.uint8)
            pcd_mask = np.zeros(xyzs_ego.shape[0], dtype=np.bool_)  # 是否已上色

            # 遍历每个相机进行投影、上色、深度
            for cam_id, _ in scene_data.camera_specs.items():
                image = np.array(scene_data.images[frame_idx][cam_id], dtype=np.uint8)
                h, w = image.shape[:2]

                # 相机参数
                intrinsics = scene_data.intrinsics_matrix[cam_id]  # 3x3
                c2w = scene_data.cam2egos[cam_id]
                w2c = np.linalg.inv(c2w)

                # 投影
                xyzs_ego_homo = np.concatenate(
                    [xyzs_ego, np.ones((xyzs_ego.shape[0], 1))], axis=1
                )
                xyzs_cam = xyzs_ego_homo @ w2c.T
                points3d_camera = xyzs_cam[:, :3]
                depth_cam = xyzs_cam[:, 2]

                points2d = (intrinsics @ points3d_camera.T).T
                points2d = points2d[:, :2] / np.clip(points2d[:, 2:3], 1e-5, None)
                u = points2d[:, 0]
                v = points2d[:, 1]

                # 过滤（边界 + 深度 + ego_mask）
                v_int = np.clip(v.astype(np.int32), 0, h - 1)
                u_int = np.clip(u.astype(np.int32), 0, w - 1)

                ego_mask_flat = scene_data.ego_masks[cam_id].ravel()
                ego_mask_proj = ego_mask_flat[v_int * w + u_int]
                inliner_mask = (
                    (u >= 0)
                    & (u < w)
                    & (v >= 0)
                    & (v < h)
                    & (depth_cam > 0)
                    & (~ego_mask_proj)
                )
                valid_global_mask |= inliner_mask

                #######################################################
                points2d_camera = points2d[inliner_mask]
                points3d_camera = points3d_camera[inliner_mask]
                inliner_indices_arr = np.where(inliner_mask)[0]
                #######################################################

                # 上色（仅未上色点）
                u_depth, v_depth = points2d_camera[:, 0], points2d_camera[:, 1]
                u_depth = np.clip(u_depth, 0, w - 1).astype(np.int32)
                v_depth = np.clip(v_depth, 0, h - 1).astype(np.int32)
                color_value = image[v_depth, u_depth]

                paint_mask = ~pcd_mask[inliner_indices_arr]
                paint_inliner_indices_arr = inliner_indices_arr[paint_mask]
                pcd_color[paint_inliner_indices_arr] = color_value[paint_mask]
                pcd_mask[paint_inliner_indices_arr] = True

                # 深度图
                depth_value = points3d_camera[:, 2]
                depth = np.full((h, w), np.finfo(np.float32).max).reshape(-1)
                indices = v_depth * w + u_depth
                np.minimum.at(depth, indices, depth_value)
                depth[depth >= np.finfo(np.float32).max - 1e-5] = 0

                valid_depth_pixel = depth > 0
                valid_depth_value = depth[valid_depth_pixel].astype(np.float32)
                valid_depth_pixel = valid_depth_pixel.reshape(h, w).astype(np.bool_)

                depth_filename = os.path.join(
                    lidar_dir_depth, f"{frame_idx:06d}_{cam_id}.npz"
                )
                np.savez_compressed(
                    depth_filename, mask=valid_depth_pixel, value=valid_depth_value
                )

                # 可视化（cam_id == 0）
                if cam_id == 0:
                    depth_vis_filename = os.path.join(
                        lidar_dir_depth, f"{frame_idx:06d}_{cam_id}.png"
                    )
                    depth = depth.reshape(h, w).astype(np.float32)
                    depth_vis, _ = visualize_depth_numpy(depth)
                    depth_on_img = image
                    depth_on_img[depth > 0] = depth_vis[depth > 0]
                    Image.fromarray(depth_on_img).save(depth_vis_filename)

            # 保存 actor 点云
            pcd_instance_mask = np.zeros(xyzs_ego.shape[0], dtype=np.bool_)
            for track_id in frame_instances[str(frame_idx)]:
                actor_dir = os.path.join(lidar_dir_actor, f"{track_id}")
                os.makedirs(actor_dir, exist_ok=True)

                frame_annotations = instances_info[str(track_id)]["frame_annotations"]
                idx = frame_annotations["frame_idx"].index(frame_idx)
                obj_to_ego = frame_annotations["obj_to_ego"][idx]
                length, width, height = frame_annotations["box_size"][idx]

                # 转换到物体坐标系
                xyzs_ego_homo = np.concatenate(
                    [xyzs_ego, np.ones((xyzs_ego.shape[0], 1))], axis=1
                )
                xyzs_actor = xyzs_ego_homo @ np.linalg.inv(obj_to_ego).T
                xyzs_actor = xyzs_actor[..., :3]

                bbox = (
                    np.array([[-length, -width, -height], [length, width, height]])
                    * 0.5
                )
                corners3d = bbox_to_corner3d(bbox)
                inbbox_mask = inbbox_points(xyzs_actor, corners3d)

                pcd_instance_mask |= inbbox_mask

                if inbbox_mask.sum() > 0:
                    # pointcloud_actor[track_id]["xyz"].append(xyzs_actor[inbbox_mask])
                    # pointcloud_actor[track_id]["rgb"].append(pcd_color[inbbox_mask])
                    # pointcloud_actor[track_id]["mask"].append(
                    #     pcd_mask[inbbox_mask][:, None]
                    # )

                    ply_actor_path = os.path.join(actor_dir, f"{frame_idx:06d}.ply")
                    storePly(
                        ply_actor_path,
                        xyzs_actor[inbbox_mask],
                        pcd_color[inbbox_mask],
                        pcd_mask[inbbox_mask][:, None],
                    )

            # 保存 background 点云
            valid_background_mask = ~pcd_instance_mask & pcd_mask
            ply_background_path = os.path.join(
                lidar_dir_background, f"{frame_idx:06d}.ply"
            )
            storePly(
                ply_background_path,
                xyzs_ego[valid_background_mask],
                pcd_color[valid_background_mask],
                pcd_mask[valid_background_mask][:, None],
            )

        # FIXME(syc): 这个会报错，但是不是必要的数据
        # # 合并 actor full.ply
        # for track_id, pointcloud in pointcloud_actor.items():
        #     xyzs = np.concatenate(pointcloud["xyz"], axis=0)
        #     rgbs = np.concatenate(pointcloud["rgb"], axis=0)
        #     masks = np.concatenate(pointcloud["mask"], axis=0)
        #     ply_actor_path_full = os.path.join(lidar_dir_actor, track_id, "full.ply")

        #     try:
        #         storePly(ply_actor_path_full, xyzs, rgbs, masks)
        #     except:
        #         pass  # No pcd

    def _load_scene_data(self, clip_name) -> SceneData:
        clip_dir = os.path.join(self.load_dir, clip_name)

        frame_timestamps = self._read_frame_timestamps(clip_dir)

        # Get camera specifications
        camera_specs = dict()
        for cam_id, cam in enumerate(ALL_CAM_SPECS):
            if self.skip_front_wide_side_cameras and cam.key in SIDE_WIDE_KEYS:
                continue
            camera_specs[cam_id] = cam

        original_cam_name_to_cam_id = {
            cam.key: cam_id for cam_id, cam in camera_specs.items()
        }

        # Find main lidar name
        main_lidar_name = None
        sample_dir = os.path.join(clip_dir, frame_timestamps[0])
        pcd_files = [f for f in os.listdir(sample_dir) if f.endswith(".pcd")]
        for lidar_name in LIDAR_CANDIDATES:
            if any(f"-{lidar_name}-" in pcd_file for pcd_file in pcd_files):
                main_lidar_name = lidar_name
                print(f"INFO: Found main LiDAR: {main_lidar_name}")
                break
        else:
            raise RuntimeError("No valid LiDAR found in the sample directory.")

        ego_masks = self._load_ego_masks(clip_dir, camera_specs)
        images = self._load_images(clip_dir, frame_timestamps, camera_specs)
        ego_pose_data = self._read_ego_pose_data(
            clip_dir, frame_timestamps, camera_specs, original_cam_name_to_cam_id
        )

        camera_params = self._read_camera_params(clip_dir)
        cam2egos = self._parse_extrinsics(camera_params, camera_specs)
        intrinsics_matrix, intrinsics_params = self._parse_intrinsics(
            camera_params, camera_specs
        )

        lidar2ego = self._read_lidar2ego(clip_dir, main_lidar_name)

        return SceneData(
            clip_dir=clip_dir,
            frame_timestamps=frame_timestamps,
            ego_masks=ego_masks,
            images=images,
            ego_pose_data=ego_pose_data,
            cam2egos=cam2egos,
            intrinsics_matrix=intrinsics_matrix,
            intrinsics_params=intrinsics_params,
            lidar2ego=lidar2ego,
            camera_specs=camera_specs,
            original_cam_name_to_cam_id=original_cam_name_to_cam_id,
            main_lidar_name=main_lidar_name,
        )

    def _load_ego_masks(self, clip_dir, camera_specs: Dict[int, CamSpec]):
        ego_masks = dict()
        for cam_id, cam in camera_specs.items():
            ego_mask_path = os.path.join(clip_dir, "ego_masks", f"{cam.key}.png")
            if os.path.exists(ego_mask_path):
                print(f"Found ego mask for '{cam.key}'.")
                ego_mask = Image.open(ego_mask_path).convert("L")
                ego_mask = np.array(ego_mask, dtype=np.bool_)
            else:
                print(f"No ego mask found for '{cam.key}'.")
                ego_mask = np.zeros((cam.height, cam.width), dtype=np.bool_)
            ego_masks[cam_id] = ego_mask

        return ego_masks

    def _load_images(
        self, clip_dir, frame_timestamps: List, camera_specs: Dict[int, CamSpec]
    ):
        images = []
        for frame_timestamp in frame_timestamps:
            sample_dir = os.path.join(clip_dir, frame_timestamp)
            frame_imgs = {}
            for img_name in os.listdir(sample_dir):
                if not img_name.endswith(".jpg"):
                    continue

                # 求出相机编号
                for cam_id, cam in camera_specs.items():
                    if cam.key in img_name:
                        img_path = os.path.join(sample_dir, img_name)
                        img = Image.open(img_path)
                        frame_imgs[cam_id] = img
                        break
                else:
                    continue

            frame_imgs = dict(sorted(frame_imgs.items()))
            images.append(frame_imgs)
        return images

    def _read_lidar2ego(self, clip_dir, main_lidar_name):
        data_frame_car_info_path = os.path.join(clip_dir, "data_frame_car_info.json")
        with open(data_frame_car_info_path, "r") as f:
            data_frame_car_info = json.load(f)

        lidar_params = data_frame_car_info["lidar_params"]
        lidar2ego_raw = None
        for param in lidar_params:
            if param["installation"]["lidar_id"] == main_lidar_name:
                lidar2ego_raw = param["installation"]["extrinsics"]
                break

        lidar2ego = euler_to_transform_matrix(
            lidar2ego_raw["x"],
            lidar2ego_raw["y"],
            lidar2ego_raw["z"],
            lidar2ego_raw["yaw"],
            lidar2ego_raw["pitch"],
            lidar2ego_raw["roll"],
        )
        return lidar2ego

    def _read_ego_pose_data_from_json(
        self,
        frame_data_json_path: str,
        camera_specs: Dict[int, 'CamSpec'],
        original_cam_name_to_cam_id: Dict[str, int],
    ) -> Dict:
        """
        从JSON文件读取ego_pose数据
        
        Args:
            frame_data_path: JSON文件路径
            camera_specs: 相机规格字典
            original_cam_name_to_cam_id: 原始相机名到相机ID的映射
        
        Returns:
            ego_pose_data字典，包含FRAME和各相机的位姿信息
        """
        # 读取JSON文件
        with open(frame_data_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 获取main_timestamp
        main_timestamp = data.get("main_timestamp")
        if main_timestamp is None:
            raise ValueError("JSON中缺少main_timestamp字段")
        
        # 获取image_infos列表
        image_infos = data.get("image_infos", [])
        if not image_infos:
            raise ValueError("JSON中缺少image_infos字段或为空")
        
        # 获取所有需要的相机key
        cam_keys = [c.key for _, c in camera_specs.items()]
        
        ego_pose_data = {}
        
        # 查找FRAME数据（timestamp等于main_timestamp的项）
        for info in image_infos:
            timestamp = info.get("timestamp")
            if timestamp == main_timestamp:
                pose = info.get("vehicle_pose")
                if pose:
                    pose_matrix = euler_to_transform_matrix(
                        pose["x"],
                        pose["y"],
                        pose["z"],
                        pose["yaw"],
                        pose["pitch"],
                        pose["roll"],
                    )
                    ego_pose_data["FRAME"] = {
                        "timestamp": timestamp,
                        "pose": pose_matrix,
                    }
                    break
        
        # 如果没有找到FRAME数据，使用第一个image_info
        if "FRAME" not in ego_pose_data and image_infos:
            first_info = image_infos[0]
            timestamp = first_info.get("timestamp")
            pose = first_info.get("vehicle_pose")
            if pose:
                pose_matrix = euler_to_transform_matrix(
                    pose["x"],
                    pose["y"],
                    pose["z"],
                    pose["yaw"],
                    pose["pitch"],
                    pose["roll"],
                )
                ego_pose_data["FRAME"] = {
                    "timestamp": timestamp,
                    "pose": pose_matrix,
                }
        
        # 处理所有相机的数据
        for info in image_infos:
            camera_id = info.get("camera_id")
            timestamp = info.get("timestamp")
            pose = info.get("vehicle_pose")
            
            # 检查camera_id是否在映射中
            if camera_id not in original_cam_name_to_cam_id:
                continue
                
            # 检查是否是我们需要的相机
            cam_id = original_cam_name_to_cam_id[camera_id]
            if cam_id not in camera_specs:
                continue
                
            camera_name = camera_specs[cam_id].name
            
            if pose:
                pose_matrix = euler_to_transform_matrix(
                    pose["x"],
                    pose["y"],
                    pose["z"],
                    pose["yaw"],
                    pose["pitch"],
                    pose["roll"],
                )
                ego_pose_data[camera_name] = {
                    "timestamp": timestamp,
                    "pose": pose_matrix,
                }
        
        # 验证是否找到了数据
        if not ego_pose_data:
            raise ValueError("未找到任何有效的ego_pose数据")
        
        return ego_pose_data

    def _read_ego_pose_data_from_pbtxt(
        self,
        frame_data_path,
        camera_specs: Dict[int, CamSpec],
        original_cam_name_to_cam_id: Dict[str, int],
    ):
        main_timestamp = None

        image_infos = []
        current_image_info = None

        cam_keys = [c.key for _, c in camera_specs.items()]

        with open(frame_data_path, "r", encoding="utf-8") as f:
            frame_data = f.read()

        frame_data = frame_data.strip().split("\n")

        for line in frame_data:
            line = line.strip()
            if not line:
                continue

            if line.startswith("main_timestamp") and main_timestamp is None:
                main_timestamp = float(line.split(":")[1].strip())
            elif line.startswith("image_infos"):
                current_image_info = {
                    "camera_id": None,
                    "timestamp": None,
                    "vehicle_pose": None,
                }
            elif current_image_info is not None:
                if line.startswith("camera_id:"):
                    current_image_info["camera_id"] = line.split(":")[1].strip()  # type: ignore
                elif line.startswith("timestamp:"):
                    current_image_info["timestamp"] = float(line.split(":")[1].strip())  # type: ignore
                elif line.startswith("vehicle_pose"):
                    current_image_info["vehicle_pose"] = {}  # type: ignore
                elif current_image_info["vehicle_pose"] is not None:
                    if line.startswith("}"):
                        if current_image_info["camera_id"] in cam_keys:
                            image_infos.append(current_image_info)
                        current_image_info = None
                    else:
                        key, value = line.split(":")
                        key = key.strip()
                        value = float(value.strip())
                        current_image_info["vehicle_pose"][key] = value

        ego_pose_data = dict()
        for info in image_infos:
            timestamp = info["timestamp"]
            if timestamp == main_timestamp:
                pose = info["vehicle_pose"]
                pose_matrix = euler_to_transform_matrix(
                    pose["x"],
                    pose["y"],
                    pose["z"],
                    pose["yaw"],
                    pose["pitch"],
                    pose["roll"],
                )
                ego_pose_data["FRAME"] = {
                    "timestamp": timestamp,
                    "pose": pose_matrix,
                }
                break

        for info in image_infos:
            cam_id = original_cam_name_to_cam_id[info["camera_id"]]
            camera_name = camera_specs[cam_id].name
            timestamp = info["timestamp"]
            pose = info["vehicle_pose"]
            pose_matrix = euler_to_transform_matrix(
                pose["x"],
                pose["y"],
                pose["z"],
                pose["yaw"],
                pose["pitch"],
                pose["roll"],
            )
            ego_pose_data[camera_name] = {
                "timestamp": timestamp,
                "pose": pose_matrix,
            }
        return ego_pose_data

    def _read_ego_pose_data(
        self,
        clip_dir,
        frame_timestamps,
        camera_specs,
        original_cam_name_to_cam_id: Dict[str, int],
    ):
        ego_pose_data_list = []
        for frame_timestamp in frame_timestamps:
            sample_dir = os.path.join(clip_dir, frame_timestamp)

            frame_data_path = os.path.join(sample_dir, "data_frame.pb.txt")
            frame_data_json_path = os.path.join(sample_dir, "data_frame.json")
            
            if os.path.exists(frame_data_path):
                ego_pose_data = self._read_ego_pose_data_from_pbtxt(
                    frame_data_path, camera_specs, original_cam_name_to_cam_id
                )
                
            elif os.path.exists(frame_data_json_path):
                ego_pose_data = self._read_ego_pose_data_from_json(
                    frame_data_json_path, camera_specs, original_cam_name_to_cam_id
                )
            
            ego_pose_data_list.append(ego_pose_data)

        return ego_pose_data_list

    def _read_frame_timestamps(self, clip_dir):
        data_frame_seq_path = os.path.join(clip_dir, "data_frame_seq.json")
        with open(data_frame_seq_path, "r") as f:
            data_frame_seq = json.load(f)
        frame_timestamps = [
            item["data_frame_path"] for item in data_frame_seq["data_frame_seq_items"]
        ]
        return frame_timestamps

    def _read_camera_params(self, clip_dir):
        camera_params_path = os.path.join(clip_dir, "camera_params.json")
        with open(camera_params_path, "r") as f:
            camera_params = json.load(f)
        return camera_params

    def _parse_extrinsics(self, camera_params, camera_specs: Dict[int, CamSpec]):
        extrinsics = dict()
        for cam_id, cam in camera_specs.items():
            cam2ego = camera_params[cam.key]["camera_to_vehicle_extrinsics"]
            cam2ego = euler_to_transform_matrix(
                cam2ego["x"],
                cam2ego["y"],
                cam2ego["z"],
                cam2ego["yaw"],
                cam2ego["pitch"],
                cam2ego["roll"],
            )
            cam2ego = cam2ego @ OPENCV2DATASET
            extrinsics[cam_id] = cam2ego

        return extrinsics

    def _parse_intrinsics(self, camera_params, camera_specs: Dict[int, CamSpec]):
        intrinsics_matrix = dict()
        intrinsics_params = dict()

        for cam_id, cam in camera_specs.items():
            intrinsic_raw = camera_params[cam.key]["intrinsics"]

            fx = intrinsic_raw["fx"]
            fy = intrinsic_raw["fy"]
            cx = intrinsic_raw["cx"]
            cy = intrinsic_raw["cy"]

            # 当前数据未提供畸变参数，按无畸变处理
            k1 = k2 = p1 = p2 = k3 = k4 = k5 = k6 = 0.0

            matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            params = [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6]

            intrinsics_matrix[cam_id] = matrix
            intrinsics_params[cam_id] = params

        return intrinsics_matrix, intrinsics_params

    def _get_obj_pose_and_size(
        self, obj_data
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        返回 obj2lidar, (l, w, h)
        """
        rotation = obj_data["psr"]["rotation"]
        rotation = euler_to_rotation_matrix(rotation["z"], rotation["y"], rotation["x"])

        position = obj_data["psr"]["position"]
        position = np.array([position["x"], position["y"], position["z"]])

        obj2ego = np.eye(4, dtype=np.float64)
        obj2ego[:3, :3] = rotation
        obj2ego[:3, 3] = position

        scale = obj_data["psr"]["scale"]
        l, w, h = scale["x"], scale["y"], scale["z"]

        return obj2ego, (l, w, h)

    def save_novel_view_infos(
        self, scene_data: SceneData, novel_view_dirs: AnyStr, clip_name: AnyStr
    ):
        # Note:
        # Dir Tree
        # novel_views_infos dir (onmire processed)
        # -- novel_views
        #   -- shift_2
        #       -- images
        #           000000_0_shift_2.00.png
        #       -- novel_caminfos
        #           000000_0shift_2.00.json
        # -- dynamic_mask
        # -- instances

        cache = self._cache[clip_name]
        categories = ["all", "human", "vehicle"]

        def create_dynamic_mask_dirs(novel_cam):
            for obj_type in categories:
                mask_dir = f"{novel_view_dirs}/{novel_cam}/dynamic_masks/{obj_type}"
                if not os.path.exists(mask_dir):
                    os.makedirs(mask_dir)

        def process_novel_cam_obj_mask(novel_cam):
            caminfo_path = f"{novel_view_dirs}/{novel_cam}/novel_caminfos"
            caminfos = [
                json.load(open(f"{caminfo_path}/{cam}"))
                for cam in os.listdir(caminfo_path)
            ]

            assert len(caminfos) == len(
                cache.sample_names
            ), "Novel View image length do not equal with cache.sample_names"

            create_dynamic_mask_dirs(novel_cam)

            # fetch origin image info
            from_cam_id = (
                caminfos[0]
                .get("guidance_rgb_path", False)
                .split("/")[-1]
                .split("_")[-1]
                .split(".")[0]
            )

            this_cam_specs = scene_data.camera_specs.get(int(from_cam_id), False)
            img_shapes = (
                (this_cam_specs.get("width"), this_cam_specs.get("height"))
                if this_cam_specs
                else (-1, -1)
            )

            # 初始化掩码图像
            masks_vehicle = [
                np.zeros((img_shapes[0], img_shapes[1], 3), dtype=np.uint8)
                for _ in caminfos
            ]
            masks_human = [
                np.zeros((img_shapes[0], img_shapes[1], 3), dtype=np.uint8)
                for _ in caminfos
            ]
            masks_all = [
                np.zeros((img_shapes[0], img_shapes[1], 3), dtype=np.uint8)
                for _ in caminfos
            ]

            center_point = np.mean(np.array(cache.ego_poses)[:, :3, 3], axis=0)

            for cam_t in caminfos:
                (frame_idx_s, _) = (
                    cam_t.get("guidance_rgb_path", False).split("/")[-1].split("_")
                )

                cur_sample_label = cache.sample_names[int(frame_idx_s)]
                label_path = os.path.join(
                    cache.clip_dir, "label", f"{cur_sample_label}.json"
                )
                with open(label_path, "r") as f:
                    labels = json.load(f)

                # crafter存world2cam
                R = np.array(cam_t.get("R", False))
                T = np.array(cam_t.get("T", False))
                K = np.array(cam_t.get("K", False))

                ego_pose = np.array(cache.ego_poses[int(frame_idx_s)])
                ego_pose[:3, 3] -= center_point

                for obj_data in labels:
                    # filter obj not in novel view
                    obj2ego, (l, w, h) = self._get_obj_pose_and_size(obj_data)

                    world2cam = np.eye(4)
                    world2cam[:3, :3] = R.T
                    world2cam[:3, 3] = T

                    ego2cam = world2cam @ ego_pose
                    obj2cam = ego2cam @ obj2ego

                    depth = obj2cam[2, -1]
                    if depth < 1.5 and depth > -1.5:
                        # skip this obj
                        continue

                    box = Box(
                        center=obj2cam[:3, 3],
                        size=[w, l, h],
                        orientation=Quaternion(matrix=obj2cam[:3, :3]),
                    )
                    corners_cam = box.corners().T.astype(np.float32)
                    corners_2d = project_points_to_image(
                        corners_cam,
                        K,
                    )

                    obj_type = obj_data["obj_type"]
                    if obj_type == "Person":
                        masks_human[int(frame_idx_s)] = draw_and_fill_box(
                            masks_human[int(frame_idx_s)], corners_2d
                        )

                    else:
                        masks_vehicle[int(frame_idx_s)] = draw_and_fill_box(
                            masks_vehicle[int(frame_idx_s)], corners_2d
                        )

            for idx in range(len(caminfos)):
                masks_all[idx] = np.maximum(masks_all[idx], masks_vehicle[idx])

            # 保存掩码图像
            for obj_type, masks in zip(
                categories, [masks_all, masks_human, masks_vehicle]
            ):
                for frame_idx, mask in enumerate(masks):
                    mask_gray = Image.fromarray(mask).convert("L")
                    mask_path = os.path.join(
                        f"{novel_view_dirs}/{novel_cam}/dynamic_masks/{obj_type}",
                        f"{str(frame_idx).zfill(6)}_{novel_cam}.png",
                    )
                    mask_gray.save(mask_path)

        for novel_view in os.listdir(novel_view_dirs):
            print(f"process novel view {novel_view}")
            process_novel_cam_obj_mask(novel_view)
