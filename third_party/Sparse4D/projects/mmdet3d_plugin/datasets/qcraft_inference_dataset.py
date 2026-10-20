import os
import json
import copy
from typing import List, Dict
from dataclasses import dataclass
import cv2
import imageio

import mmcv
import numpy as np
from torch.utils.data import Dataset
from pyquaternion import Quaternion
from PIL import Image
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.eval.detection.config import config_factory as det_configs

from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose

from projects.mmdet3d_plugin.datasets.utils import (
    draw_lidar_bbox3d_on_bev,
    draw_lidar_bbox3d_on_img,
)

OPENCV2DATASET = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

def euler_to_rotation_matrix(yaw, pitch, roll):
    # Z 轴旋转（yaw）
    R_z = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    # Y 轴旋转（pitch）
    R_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    # X 轴旋转（roll）
    R_x = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )

    R = R_z @ R_y @ R_x
    return R

def euler_to_transform_matrix(x, y, z, yaw, pitch, roll):
    R = euler_to_rotation_matrix(yaw, pitch, roll)
    T = np.array([x, y, z])
    T_matrix = np.eye(4)
    T_matrix[:3, :3] = R
    T_matrix[:3, 3] = T
    return T_matrix

@dataclass(frozen=True)
class CamSpec:
    key: str  # 原始传感器名称
    name: str
    width: int
    height: int
    comment: str = ""

@dataclass(frozen=True)
class SceneData:
    frame_timestamps: List[str]
    image_paths: List[Dict[str, str]]
    lidar_paths: List[str]
    ego_poses: List[np.ndarray]
    cam2egos: Dict[str, np.ndarray]
    intrinsics_matrix: Dict[str, np.ndarray]
    lidar2ego: np.ndarray

@DATASETS.register_module()
class QCraftInferenceDataset(Dataset):
    CLASSES = (
        "car",
        "truck",
        "trailer",
        "bus",
        "construction_vehicle",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "barrier",
    )
    NUSC_TO_SUSTECH_MAP = {
        "car": "Car",
        "truck": "Truck",
        "trailer": "LongVehicle",
        "bus": "Bus",
        "construction_vehicle": "ConstructionCart",
        "bicycle": "Bicycle",
        "motorcycle": "Motorcycle",
        "pedestrian": "Pedestrian",
        "traffic_cone": "Cone",
        "barrier": "TrafficBarrier",
    }
    ID_COLOR_MAP = [
        (59, 59, 238),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 255),
        (0, 127, 255),
        (71, 130, 255),
        (127, 127, 0),
    ]

    CAM_SPECS: List[CamSpec] = [
        # CamSpec(
        #     key="CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H110",
        #     name="front_wide_110",
        #     width=1024,
        #     height=512,
        #     comment="广角前视 FOV110",
        # ),
        CamSpec(
            key="CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H60",
            name="front_wide_60",
            width=1024,
            height=512,
            comment="广角前视 FOV60",
        ),
        # CamSpec(
        #     key="CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H30",
        #     name="front_tele_30",
        #     width=1024,
        #     height=512,
        #     comment="长焦前视 FOV30",
        # ),
        CamSpec(
            key="CAM_PBQ_FRONT_RIGHT_RESET_OPTICAL_H99",
            name="front_right_99",
            width=1024,
            height=512,
            comment="右前 FOV99",
        ),
        CamSpec(
            key="CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99",
            name="front_left_99",
            width=1024,
            height=512,
            comment="左前 FOV99",
        ),
        CamSpec(
            key="CAM_PBQ_REAR_RESET_OPTICAL_H50",
            name="rear_50",
            width=1024,
            height=512,
            comment="后视 FOV50",
        ),
        CamSpec(
            key="CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H99",
            name="rear_left_99",
            width=1024,
            height=512,
            comment="左后 FOV99",
        ),
        CamSpec(
            key="CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H99",
            name="rear_right_99",
            width=1024,
            height=512,
            comment="右后 FOV99",
        ),
    ]

    def __init__(
        self,
        scene_id,
        pipeline=None,
        data_root=None,
        classes=None,
        # with_velocity=False,  # TODO: remove
        modality=None,
        test_mode=False,
        vis_score_threshold=0.25,
        data_aug_conf=None,
        tracking=False,
        # tracking_threshold=0.2,
    ):
        super().__init__()

        self.data_root = data_root
        self.test_mode = test_mode
        self.modality = modality

        if classes is not None:
            self.CLASSES = classes
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.data_infos = self.load_annotations(scene_id)

        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        # self.with_velocity = with_velocity
        self.det3d_eval_configs = det_configs("detection_cvpr_2019")
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )
        self.vis_score_threshold = vis_score_threshold

        self.data_aug_conf = data_aug_conf
        self.tracking = tracking
        # self.tracking_threshold = tracking_threshold

    def __len__(self):
        return len(self.data_infos)

    def get_augmentation(self):
        if self.data_aug_conf is None:
            return None
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]

        resize = max(fH / H, fW / W)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = (
            int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH)
            - fH
        )
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate = 0
        rotate_3d = 0

        aug_config = {
            "resize": resize,
            "resize_dims": resize_dims,
            "crop": crop,
            "flip": flip,
            "rotate": rotate,
            "rotate_3d": rotate_3d,
        }
        return aug_config

    def __getitem__(self, idx):
        aug_config = self.get_augmentation()

        data = self.get_data_info(idx)
        data["aug_config"] = aug_config
        data = self.pipeline(data)
        return data

    def load_annotations(self, scene_id, max_sweeps=10) -> list:
        scene_dir = os.path.join(self.data_root, scene_id)
        raw_scene_data = self._load_raw_scene_data(scene_dir)
        self.raw_scene_data = raw_scene_data

        data_infos = []

        for frame_idx, timestamp in enumerate(raw_scene_data.frame_timestamps):
            lidar_path = raw_scene_data.lidar_paths[frame_idx]
            mmcv.check_file_exist(lidar_path)

            # lidar2ego
            lidar2ego_raw = raw_scene_data.lidar2ego
            lidar2ego_translation = lidar2ego_raw[:3, 3].tolist()
            lidar2ego_rotation = lidar2ego_raw[:3, :3]
            lidar2ego_rotation = Quaternion(matrix=lidar2ego_rotation).elements.tolist()

            # ego2global
            ego2global_raw = raw_scene_data.ego_poses[frame_idx]
            ego2global_translation = ego2global_raw[:3, 3].tolist()
            ego2global_rotation = ego2global_raw[:3, :3]
            ego2global_rotation = Quaternion(matrix=ego2global_rotation).elements.tolist()

            info = {
                "lidar_path": lidar_path,
                "token": frame_idx,
                "sweeps": [],
                "cams": dict(),
                "lidar2ego_translation": lidar2ego_translation,  # [3]
                "lidar2ego_rotation": lidar2ego_rotation,  # [4]
                "ego2global_translation": ego2global_translation,  # [3]
                "ego2global_rotation": ego2global_rotation,  # [4]
                "timestamp": float(timestamp) * 1e6,
            }

            l2e_r = info["lidar2ego_rotation"]
            l2e_t = info["lidar2ego_translation"]
            e2g_r = info["ego2global_rotation"]
            e2g_t = info["ego2global_translation"]
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix

            # obtain 6 image's information per frame
            camera_types = [c.name for c in self.CAM_SPECS]
            for cam in camera_types:
                cam_info = obtain_sensor2top(
                    raw_scene_data,
                    frame_idx,
                    l2e_t,
                    l2e_r_mat,
                    e2g_t,
                    e2g_r_mat,
                    cam
                )
                cam_info.update(cam_intrinsic=raw_scene_data.intrinsics_matrix[cam])
                info["cams"].update({cam: cam_info})

            # obtain sweeps for a single key-frame
            sweeps = []
            for sweep_cnt in range(max_sweeps):
                prev_idx = frame_idx - sweep_cnt - 1
                if prev_idx < 0:
                    break

                sweep = obtain_sensor2top(
                    raw_scene_data,
                    prev_idx,
                    l2e_t,
                    l2e_r_mat,
                    e2g_t,
                    e2g_r_mat,
                    "lidar",
                )
                sweeps.append(sweep)

            info["sweeps"] = sweeps
            data_infos.append(info)
        return data_infos


    def get_data_info(self, index):
        info = self.data_infos[index]
        # standard protocol modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info["token"],
            pts_filename=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"] / 1e6,
            lidar2ego_translation=info["lidar2ego_translation"],
            lidar2ego_rotation=info["lidar2ego_rotation"],
            ego2global_translation=info["ego2global_translation"],
            ego2global_rotation=info["ego2global_rotation"],
        )
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(
            info["lidar2ego_rotation"]
        ).rotation_matrix
        lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"])
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(
            info["ego2global_rotation"]
        ).rotation_matrix
        ego2global[:3, 3] = np.array(info["ego2global_translation"])
        input_dict["lidar2global"] = ego2global @ lidar2ego

        if self.modality["use_camera"]:
            image_paths = []
            lidar2img_rts = []
            cam_intrinsic = []
            for cam_type, cam_info in info["cams"].items():
                image_paths.append(cam_info["data_path"])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
                lidar2cam_t = (
                    cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
                )
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = copy.deepcopy(cam_info["cam_intrinsic"])
                cam_intrinsic.append(intrinsic)
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                lidar2img_rt = viewpad @ lidar2cam_rt.T
                lidar2img_rts.append(lidar2img_rt)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsic,
                )
            )

        if not self.test_mode:
            pass
            # annos = self.get_ann_info(index)
            # input_dict.update(annos)

        return input_dict

    def visualize_scene(self, batch_data, pred_results, output_dir, cfg, gt_results=None):
        videoWriter = None

        for frame_id, data in enumerate(batch_data):
            img_norm_mean = np.array(cfg.img_norm_cfg["mean"])
            img_norm_std = np.array(cfg.img_norm_cfg["std"])
            raw_imgs = data["img"][0].permute(0, 2, 3, 1).cpu().numpy()
            raw_imgs = raw_imgs * img_norm_std + img_norm_mean
            raw_imgs = [
                cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
                for img in raw_imgs
            ]

            lidar2img = data["projection_mat"][0]

            results = {}
            pred_result = pred_results[frame_id]
            results["pred"] = pred_result
            if gt_results is not None:
                gt_result = gt_results[frame_id]
                results["gt"] = gt_result

            final_images = []
            for key, result in results.items():
                bboxes_3d = result["boxes_3d"]

                if "instance_ids" in result and self.tracking:
                    color = []
                    for id in result["instance_ids"].cpu().numpy().tolist():
                        color.append(self.ID_COLOR_MAP[int(id % len(self.ID_COLOR_MAP))])
                elif "labels_3d" in result:
                    color = []
                    for id in result["labels_3d"].cpu().numpy().tolist():
                        color.append(self.ID_COLOR_MAP[id])
                else:
                    color = (255, 0, 0)

                # ===== draw boxes_3d to images =====
                imgs = []
                for j, img_origin in enumerate(raw_imgs):
                    img = img_origin.copy()
                    img = draw_lidar_bbox3d_on_img(
                        bboxes_3d,
                        img,
                        lidar2img[j],
                        img_metas=None,
                        color=color,
                        thickness=2,
                    )
                    imgs.append(img)

                # ===== draw boxes_3d to BEV =====
                bev = draw_lidar_bbox3d_on_bev(
                    bboxes_3d,
                    bev_size=img.shape[0] * 2,
                    color=color,
                    thickness=2,
                )

                image = np.concatenate(
                    [
                        np.concatenate([imgs[2], imgs[0], imgs[1]], axis=1),
                        np.concatenate([imgs[5], imgs[3], imgs[4]], axis=1),
                    ],
                    axis=0,
                )
                image = np.concatenate([image, bev], axis=1)
                final_images.append(image)

            # ===== save video =====
            if videoWriter is None:
                videoWriter = imageio.get_writer(
                    os.path.join(output_dir, "label.mp4"),
                    fps=10,
                    codec='libx264',
                    quality=4
                )

            final_image = np.concatenate(final_images, axis=0)
            image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

            videoWriter.append_data(image_rgb)
        videoWriter.close()


    def _load_raw_scene_data(self, scene_dir) -> SceneData:
        frame_timestamps = self._read_frame_timestamps(scene_dir)

        image_paths = self._get_image_paths(scene_dir, frame_timestamps)
        lidar_paths = self._get_lidar_paths(scene_dir, frame_timestamps)
        ego_poses = self._read_ego_poses(scene_dir, frame_timestamps)

        camera_params_path = os.path.join(scene_dir, "camera_params.json")
        with open(camera_params_path, "r") as f:
            camera_params = json.load(f)

        cam2egos = self._parse_extrinsics(camera_params)
        intrinsics_matrix = self._parse_intrinsics(camera_params)

        lidar2ego = self._read_lidar2ego(scene_dir)

        return SceneData(
            frame_timestamps=frame_timestamps,
            image_paths=image_paths,
            lidar_paths=lidar_paths,
            ego_poses=ego_poses,
            cam2egos=cam2egos,
            intrinsics_matrix=intrinsics_matrix,
            lidar2ego=lidar2ego,
        )
    
    def _read_frame_timestamps(self, clip_dir):
        data_frame_seq_path = os.path.join(clip_dir, "data_frame_seq.json")
        with open(data_frame_seq_path, "r") as f:
            data_frame_seq = json.load(f)
        frame_timestamp = [
            item["data_frame_path"] for item in data_frame_seq["data_frame_seq_items"]
        ]
        return frame_timestamp
    
    def _get_image_paths(self, scene_dir, frame_timestamps):
        image_paths = []
        for frame_timestamp in frame_timestamps:
            current_image_paths = dict()
            sample_dir = os.path.join(scene_dir, frame_timestamp)
            for img_name in os.listdir(sample_dir):
                for cam in self.CAM_SPECS:
                    if cam.key in img_name:
                        img_path = os.path.join(sample_dir, img_name)
                        current_image_paths[cam.name] = img_path
                        break
                else:
                    continue
            image_paths.append(current_image_paths)
        return image_paths

    def _get_lidar_paths(self, scene_dir, frame_timestamps):
        lidar_paths = []
        for frame_timestamp in frame_timestamps:
            sample_dir = os.path.join(scene_dir, frame_timestamp)

            # NOTE(syc): 目前只使用一个LiDAR，但保留了支持多个LiDAR的形式
            lidar_names = [
                lidar_name  # 20250702_133223_Q2517-LDR_FRONT-1751434420.2514-ego.pcd
                for lidar_name in os.listdir(sample_dir)
                if lidar_name.endswith(".pcd")
                and f"-LDR_FRONT-" in lidar_name
            ]
            lidar_path = os.path.join(sample_dir, lidar_names[0])
            lidar_paths.append(lidar_path)
        return lidar_paths
    
    def _read_ego_pose_from_json(self, frame_data_json_path: str) -> np.ndarray:
        """
        从JSON文件读取ego_pose数据
        
        Args:
            frame_data_path: JSON文件路径
        
        Returns:
            ego_pose的变换矩阵 (4x4)
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
        
        # 查找timestamp等于main_timestamp的项
        ego_pose = None
        for info in image_infos:
            timestamp = info.get("timestamp")
            if timestamp == main_timestamp:
                pose = info.get("vehicle_pose")
                if pose:
                    ego_pose = euler_to_transform_matrix(
                        pose["x"],
                        pose["y"],
                        pose["z"],
                        pose["yaw"],
                        pose["pitch"],
                        pose["roll"],
                    )
                    break
        
        # 如果没有找到，使用第一个image_info
        if ego_pose is None and image_infos:
            first_info = image_infos[0]
            pose = first_info.get("vehicle_pose")
            if pose:
                ego_pose = euler_to_transform_matrix(
                    pose["x"],
                    pose["y"],
                    pose["z"],
                    pose["yaw"],
                    pose["pitch"],
                    pose["roll"],
                )
        
        assert ego_pose is not None, f"No matching ego pose found for main timestamp {main_timestamp}"
        return ego_pose

    def _read_ego_pose_from_pbtxt(self, frame_data_path):
        main_timestamp = None

        image_infos = []
        current_image_info = None

        cam_keys = [c.key for c in self.CAM_SPECS]

        with open(frame_data_path, "r", encoding="utf-8") as f:
            frame_data = f.read()

        frame_data = frame_data.strip().split("\n")

        for line in frame_data:
            line = line.strip()
            if not line:
                continue

            if line.startswith("main_timestamp") and main_timestamp is None:
                main_timestamp = float(line.split(":")[1].strip())
            elif line.startswith("image_infos {"):
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
                        # if current_image_info["camera_id"] in cam_keys:
                            # image_infos.append(current_image_info)
                        image_infos.append(current_image_info)
                        current_image_info = None
                    else:
                        key, value = line.split(":")
                        key = key.strip()
                        value = float(value.strip())
                        current_image_info["vehicle_pose"][key] = value

        ego_pose = None
        for info in image_infos:
            timestamp = info["timestamp"]
            if timestamp == main_timestamp:
                ego_pose = info["vehicle_pose"]
                ego_pose = euler_to_transform_matrix(
                    ego_pose["x"],
                    ego_pose["y"],
                    ego_pose["z"],
                    ego_pose["yaw"],
                    ego_pose["pitch"],
                    ego_pose["roll"],
                )
                break

        assert ego_pose is not None, f"No matching ego pose found for main timestamp {main_timestamp}"
        return ego_pose

    def _read_ego_poses(self, scene_dir, frame_timestamps):
        ego_poses = []
        for frame_timestamp in frame_timestamps:
            sample_dir = os.path.join(scene_dir, frame_timestamp)
            
            frame_data_path = os.path.join(sample_dir, "data_frame.pb.txt")
            frame_data_json_path = os.path.join(sample_dir, "data_frame.json")
            
            if os.path.exists(frame_data_path):
                ego_pose_data = self._read_ego_pose_from_pbtxt(frame_data_path)
                
            elif os.path.exists(frame_data_json_path):
                ego_pose_data = self._read_ego_pose_from_json(frame_data_json_path)
            
            ego_poses.append(ego_pose_data)

        return ego_poses
    
    def _parse_extrinsics(self, camera_params):
        extrinsics = dict()
        for cam in self.CAM_SPECS:
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
            extrinsics[cam.name] = cam2ego
        return extrinsics

    def _parse_intrinsics(self, camera_params):
        intrinsics_matrix = dict()
        for cam in self.CAM_SPECS:
            intrinsic_raw = camera_params[cam.key]["intrinsics"]

            fx = intrinsic_raw["fx"]
            fy = intrinsic_raw["fy"]
            cx = intrinsic_raw["cx"]
            cy = intrinsic_raw["cy"]

            intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            intrinsics_matrix[cam.name] = intrinsics
        return intrinsics_matrix
    
    def _read_lidar2ego(self, scene_dir):
        data_frame_car_info_path = os.path.join(scene_dir, "data_frame_car_info.json")
        with open(data_frame_car_info_path, "r") as f:
            data_frame_car_info = json.load(f)

        lidar_params = data_frame_car_info["lidar_params"]
        lidar2ego = None
        for param in lidar_params:
            if param["installation"]["lidar_id"] == "LDR_FRONT":
                lidar2ego = param["installation"]["extrinsics"]
                lidar2ego = euler_to_transform_matrix(
                    lidar2ego["x"],
                    lidar2ego["y"],
                    lidar2ego["z"],
                    lidar2ego["yaw"],
                    lidar2ego["pitch"],
                    lidar2ego["roll"],
                )
                break
        return lidar2ego


def output_to_nusc_box(detection, threshold=None):
    box3d = detection["boxes_3d"]
    scores = detection["scores_3d"].numpy()
    labels = detection["labels_3d"].numpy()
    if "instance_ids" in detection:
        ids = detection["instance_ids"]  # .numpy()
    if threshold is not None:
        if "cls_scores" in detection:
            mask = detection["cls_scores"].numpy() >= threshold
        else:
            mask = scores >= threshold
        box3d = box3d[mask]
        scores = scores[mask]
        labels = labels[mask]
        ids = ids[mask]

    if hasattr(box3d, "gravity_center"):
        box_gravity_center = box3d.gravity_center.numpy()
        box_dims = box3d.dims.numpy()
        nus_box_dims = box_dims[:, [1, 0, 2]]
        box_yaw = box3d.yaw.numpy()
    else:
        box3d = box3d.numpy()
        box_gravity_center = box3d[..., :3].copy()
        box_dims = box3d[..., 3:6].copy()
        nus_box_dims = box_dims[..., [1, 0, 2]]
        box_yaw = box3d[..., 6].copy()

    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    # box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        if hasattr(box3d, "gravity_center"):
            velocity = (*box3d.tensor[i, 7:9], 0.0)
        else:
            velocity = (*box3d[i, 7:9], 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            nus_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity,
        )
        if "instance_ids" in detection:
            box.token = ids[i]
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(
    info,
    boxes,
    classes,
    cls_range_map,
):
    box_list = []
    for i, box in enumerate(boxes):
        # Move box to ego vehicle coord system
        box.rotate(Quaternion(info["lidar2ego_rotation"]))
        box.translate(np.array(info["lidar2ego_translation"]))
        # # filter det in ego.
        # radius = np.linalg.norm(box.center[:2], 2)
        # det_range = cls_range_map[classes[box.label]]
        # if radius > det_range:
        #     continue
        # Move box to global coord system
        box.rotate(Quaternion(info["ego2global_rotation"]))
        box.translate(np.array(info["ego2global_translation"]))
        box_list.append(box)
    return box_list

def lidar_nusc_box_to_ego(
    info,
    boxes,
    classes,
    cls_range_map,
):
    box_list = []
    for i, box in enumerate(boxes):
        # Move box to ego vehicle coord system
        box.rotate(Quaternion(info["lidar2ego_rotation"]))
        box.translate(np.array(info["lidar2ego_translation"]))
        # # filter det in ego.
        # radius = np.linalg.norm(box.center[:2], 2)
        # det_range = cls_range_map[classes[box.label]]
        # if radius > det_range:
        #     continue
        box_list.append(box)
    return box_list


def obtain_sensor2top(
    raw_scene_data: SceneData, frame_idx, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type="lidar"
):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str, optional): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    if sensor_type == "lidar":
        data_path = raw_scene_data.lidar_paths[frame_idx]
        sensor2ego_raw = raw_scene_data.lidar2ego
    else:
        data_path = raw_scene_data.image_paths[frame_idx][sensor_type]
        sensor2ego_raw = raw_scene_data.cam2egos[sensor_type]

    sensor2ego_translation = sensor2ego_raw[:3, 3].tolist()
    sensor2ego_rotation = sensor2ego_raw[:3, :3]
    sensor2ego_rotation = Quaternion(matrix=sensor2ego_rotation).elements.tolist()

    # ego2global
    ego2global_raw = raw_scene_data.ego_poses[frame_idx]
    ego2global_translation = ego2global_raw[:3, 3].tolist()
    ego2global_rotation = ego2global_raw[:3, :3]
    ego2global_rotation = Quaternion(matrix=ego2global_rotation).elements.tolist()

    sweep = {
        "data_path": data_path,  # 相对路径
        "type": sensor_type,
        "sample_data_token": frame_idx,
        "sensor2ego_translation": sensor2ego_translation,
        "sensor2ego_rotation": sensor2ego_rotation,
        "ego2global_translation": ego2global_translation,
        "ego2global_rotation": ego2global_rotation,
        "timestamp": float(raw_scene_data.frame_timestamps[frame_idx]) * 1e6,
    }
    l2e_r_s = sweep["sensor2ego_rotation"]
    l2e_t_s = sweep["sensor2ego_translation"]
    e2g_r_s = sweep["ego2global_rotation"]
    e2g_t_s = sweep["ego2global_translation"]

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    )
    sweep["sensor2lidar_rotation"] = R.T  # points @ R.T + T
    sweep["sensor2lidar_translation"] = T
    return sweep
