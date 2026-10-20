from typing import Dict
import logging
import os
import json
import joblib
import numpy as np
from tqdm import trange, tqdm
from omegaconf import OmegaConf

import torch
from torch import Tensor

from pytorch3d.transforms import matrix_to_quaternion
from datasets.base.scene_dataset import ModelType
from datasets.base.lidar_source import SceneLidarSource
from datasets.base.pixel_source import ScenePixelSource, CameraData

logger = logging.getLogger()

# define each class's node type
OBJECT_CLASS_NODE_MAPPING = {
    "Vehicle": ModelType.RigidNodes,
    "Pedestrian": ModelType.SMPLNodes,
    "Cyclist": ModelType.DeformableNodes,
}
SMPLNODE_CLASSES = ["Pedestrian"]

# OpenCV to Dataset coordinate transformation
# opencv coordinate system: x right, y down, z front
OPENCV2DATASET = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

# Chery Camera List:
# 0 : "front_wide",
# 1 : "front_main",
# 2 : "left_front",
# 3 : "left_rear",
# 4 : "right_front",
# 5 : "right_rear",
# 6 : "rear_main",
# 7 : "fisheye_left",
# 8 : "fisheye_rear",
# 9 : "fisheye_front",
# 10 : "fisheye_right"
AVAILABLE_CAM_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class CheryCameraData(CameraData):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_calibrations(self):
        """
        Load the camera intrinsics, extrinsics, timestamps, etc.
        Compute the camera-to-world matrices, lidar-to-world matrices, etc.
        """
        intrinsic = np.loadtxt(
            os.path.join(self.data_path, "intrinsics", f"{self.cam_id}.txt")
        )
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        distortions = intrinsic[4:]

        # scale intrinsics w.r.t. load size
        fx, fy = (
            fx * self.load_size[1] / self.original_size[1],
            fy * self.load_size[0] / self.original_size[0],
        )
        cx, cy = (
            cx * self.load_size[1] / self.original_size[1],
            cy * self.load_size[0] / self.original_size[0],
        )

        _intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        _distortions = np.array(distortions)

        # load camera extrinsics
        cam_to_main_lidar = np.loadtxt(
            os.path.join(self.data_path, "extrinsics", f"{self.cam_id}.txt")
        )
        # covnert rays from opencv coordinate system to chery coordinate system.
        cam_to_main_lidar = cam_to_main_lidar @ OPENCV2DATASET

        # compute per-image poses and intrinsics
        cam_to_worlds, lidar_to_worlds = [], []
        intrinsics, distortions = [], []

        # we tranform the camera poses w.r.t. the first timestep to make the translation vector of
        # the first lidar pose as the origin of the world coordinate system.
        lidar_to_world_start = np.loadtxt(
            os.path.join(self.data_path, "lidar_pose", f"{self.start_timestep:03d}.txt")
        )
        for t in range(self.start_timestep, self.end_timestep):
            lidar_to_world_current = np.loadtxt(
                os.path.join(self.data_path, "lidar_pose", f"{t:03d}.txt")
            )
            # compute lidar_to_world transformation
            lidar_to_world = (
                np.linalg.inv(lidar_to_world_start) @ lidar_to_world_current
            )
            lidar_to_worlds.append(lidar_to_world)
            # transformation:
            #   (opencv_cam -> chery_cam -> chery_lidar) -> current_world
            cam2world = lidar_to_world @ cam_to_main_lidar
            cam_to_worlds.append(cam2world)
            intrinsics.append(_intrinsics)
            distortions.append(_distortions)

        self.intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0)).float()
        self.distortions = torch.from_numpy(np.stack(distortions, axis=0)).float()
        self.cam_to_worlds = torch.from_numpy(np.stack(cam_to_worlds, axis=0)).float()

        self.cam_to_main_lidar = cam_to_main_lidar  # syc

    @classmethod
    def get_camera2worlds(
        cls, data_path: str, cam_id: str, start_timestep: int, end_timestep: int
    ) -> torch.Tensor:
        """
        Returns camera-to-world matrices for the specified camera and time range.

        Args:
            data_path (str): Path to the dataset.
            cam_id (str): Camera ID.
            start_timestep (int): Start timestep.
            end_timestep (int): End timestep.

        Returns:
            torch.Tensor: Camera-to-world matrices of shape (num_frames, 4, 4).
        """
        # Load camera extrinsics
        cam_to_lidar = np.loadtxt(
            os.path.join(data_path, "extrinsics", f"{cam_id}.txt")
        )
        cam_to_lidar = cam_to_lidar @ OPENCV2DATASET

        # Load lidar poses and compute camera-to-world matrices
        cam_to_worlds = []
        lidar_to_world_start = np.loadtxt(
            os.path.join(data_path, "lidar_pose", f"{start_timestep:03d}.txt")
        )

        for t in range(start_timestep, end_timestep):
            lidar_to_world_current = np.loadtxt(
                os.path.join(data_path, "lidar_pose", f"{t:03d}.txt")
            )
            lidar_to_world = (
                np.linalg.inv(lidar_to_world_start) @ lidar_to_world_current
            )
            cam2world = lidar_to_world @ cam_to_lidar
            cam_to_worlds.append(cam2world)

        return torch.from_numpy(np.stack(cam_to_worlds, axis=0)).float()


class CheryPixelSource(ScenePixelSource):
    def __init__(
        self,
        dataset_name: str,
        pixel_data_config: OmegaConf,
        data_path: str,
        start_timestep: int,
        end_timestep: int,
        device: torch.device = torch.device("cpu"),
        novel_view_mode: bool = False,
    ):
        super().__init__(dataset_name, pixel_data_config, device=device)
        self.data_path = data_path
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.novel_view_mode = novel_view_mode  # syc
        self.load_data()

    def load_cameras(self):
        self._timesteps = torch.arange(
            self.start_timestep, self.end_timestep
        )  ### 生成等差数列，目的是获取序列时间，对每一个时间都加载相机
        self.register_normalized_timestamps()  ### 归一化时间戳

        for idx, cam_id in enumerate(
            self.camera_list
        ):  ### camera_list就是config中相机列表[0,1,2,3,4]
            logger.info(f"Loading camera {cam_id}")
            camera = CheryCameraData(
                dataset_name=self.dataset_name,
                data_path=self.data_path,
                cam_id=cam_id,
                start_timestep=self.start_timestep,
                end_timestep=self.end_timestep,
                load_dynamic_mask=self.data_cfg.load_dynamic_mask,
                load_sky_mask=self.data_cfg.load_sky_mask,
                downscale_when_loading=self.data_cfg.downscale_when_loading[idx],
                undistort=self.data_cfg.undistort,
                buffer_downscale=self.buffer_downscale,
                device=self.device,
                novel_view_mode=self.novel_view_mode,
            )
            camera.load_time(self.normalized_time)
            unique_img_idx = (
                torch.arange(len(camera), device=self.device) * len(self.camera_list)
                + idx
            )
            camera.set_unique_ids(unique_cam_idx=idx, unique_img_idx=unique_img_idx)
            logger.info(f"Camera {camera.cam_name} loaded.")
            self.camera_data[cam_id] = camera
    
    # syc
    def load_specified_cameras(self, cam_ids, downscale_when_loading) -> Dict[int, CameraData]:
        camera_data = {}
        for idx, cam_id in enumerate(cam_ids):
            camera = CheryCameraData(
                dataset_name=self.dataset_name,
                data_path=self.data_path,
                cam_id=cam_id,
                start_timestep=self.start_timestep,
                end_timestep=self.end_timestep,
                downscale_when_loading=downscale_when_loading[idx],
                undistort=False,
                buffer_downscale=self.buffer_downscale,
                device=self.device,
                novel_view_mode=True,
            )
            camera.load_time(self.normalized_time)
            unique_img_idx = (
                torch.arange(len(camera), device=self.device) * len(cam_ids)
                + idx
            )
            camera.set_unique_ids(unique_cam_idx=idx, unique_img_idx=unique_img_idx)
            logger.info(f"Specified camera {camera.cam_name} loaded.")

            camera_data[cam_id] = camera
        
        return camera_data


    def load_objects(self):
        """
        get ground truth bounding boxes of the dynamic objects

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
        frame_instances = {
            "0": # frame idx
                List[int] # list of simplified instance ids
            ...
        }
        """

        instances_info_path = os.path.join(
            self.data_path, "instances", "instances_info.json"
        )
        frame_instances_path = os.path.join(
            self.data_path, "instances", "frame_instances.json"
        )
        with open(instances_info_path, "r") as f:
            instances_info = json.load(f)
        with open(frame_instances_path, "r") as f:
            frame_instances = json.load(f)
        # get pose of each instance at each frame
        # shape (num_frames, num_instances, 4, 4)
        num_instances = len(instances_info)
        num_full_frames = len(frame_instances)
        instances_pose = np.zeros((num_full_frames, num_instances, 4, 4))
        instances_size = np.zeros((num_full_frames, num_instances, 3))
        instances_true_id = np.arange(num_instances)
        instances_model_types = np.ones(num_instances) * -1

        lidar_to_world_start = np.loadtxt(
            os.path.join(self.data_path, "lidar_pose", f"{self.start_timestep:03d}.txt")
        )
        ### k最大有500+, 但instances_info只有119
        for index, (k, v) in enumerate(instances_info.items()):
            instances_model_types[int(index)] = OBJECT_CLASS_NODE_MAPPING[
                v["class_name"]
            ]
            for frame_idx, obj_to_world, box_size in zip(
                v["frame_annotations"]["frame_idx"],
                v["frame_annotations"]["obj_to_world"],
                v["frame_annotations"]["box_size"],
            ):
                # the first lidar pose as the origin of the world coordinate system.
                obj_to_world = np.array(obj_to_world).reshape(4, 4)
                obj_to_world = np.linalg.inv(lidar_to_world_start) @ obj_to_world
                instances_pose[frame_idx, int(index)] = np.array(obj_to_world)
                instances_size[frame_idx, int(index)] = np.array(box_size)

        per_frame_instance_mask = np.zeros(
            (num_full_frames, num_instances)
        )  ### 实例个数

        ## TODO：把valid_instances里面的编号换成索引
        for frame_idx, valid_instances in frame_instances.items():
            temp_result = []
            for i in range(len(valid_instances)):
                for index, (k, v) in enumerate(instances_info.items()):
                    if k == str(valid_instances[i]):
                        temp_result.append(index)
                        break
            valid_instances = temp_result
            # valid_instances = change_index(valid_instances, instances_info)
            per_frame_instance_mask[int(frame_idx), valid_instances] = 1

        # # waymo origin
        # for index, (k, v) in enumerate(instances_info.items()):
        #     instances_model_types[int(k)] = OBJECT_CLASS_NODE_MAPPING[v["class_name"]]
        #     for frame_idx, obj_to_world, box_size in zip(v["frame_annotations"]["frame_idx"], v["frame_annotations"]["obj_to_world"], v["frame_annotations"]["box_size"]):
        #         # the first lidar pose as the origin of the world coordinate system.
        #         obj_to_world = np.array(obj_to_world).reshape(4, 4)
        #         obj_to_world = np.linalg.inv(lidar_to_world_start) @ obj_to_world
        #         instances_pose[frame_idx, int(k)] = np.array(obj_to_world)
        #         instances_size[frame_idx, int(k)] = np.array(box_size)

        # # get frame valid instances
        # # shape (num_frames, num_instances)
        # per_frame_instance_mask = np.zeros((num_full_frames, num_instances)) ### 实例个数
        # for frame_idx, valid_instances in frame_instances.items():
        #     per_frame_instance_mask[int(frame_idx), valid_instances] = 1

        # select the frames that are in the range of start_timestep and end_timestep
        instances_pose = torch.from_numpy(
            instances_pose[self.start_timestep : self.end_timestep]
        ).float()
        instances_size = torch.from_numpy(
            instances_size[self.start_timestep : self.end_timestep]
        ).float()
        instances_true_id = torch.from_numpy(instances_true_id).long()
        instances_model_types = torch.from_numpy(instances_model_types).long()
        per_frame_instance_mask = torch.from_numpy(
            per_frame_instance_mask[self.start_timestep : self.end_timestep]
        ).bool()

        # filter out the instances that are not visible in selected frames
        ins_frame_cnt = per_frame_instance_mask.sum(dim=0)
        instances_pose = instances_pose[:, ins_frame_cnt > 0]
        instances_size = instances_size[:, ins_frame_cnt > 0]
        instances_true_id = instances_true_id[ins_frame_cnt > 0]
        instances_model_types = instances_model_types[ins_frame_cnt > 0]
        per_frame_instance_mask = per_frame_instance_mask[:, ins_frame_cnt > 0]

        # assign to the class
        # (num_frames, num_instances, 4, 4)
        self.instances_pose = instances_pose
        # (num_instances, 3)
        self.instances_size = instances_size.sum(0) / per_frame_instance_mask.sum(
            0
        ).unsqueeze(-1)
        # (num_frames, num_instances)
        self.per_frame_instance_mask = per_frame_instance_mask
        # (num_instances)
        self.instances_true_id = instances_true_id
        # (num_instances)
        self.instances_model_types = instances_model_types

        if self.data_cfg.load_smpl:
            # Collect camera-to-world matrices for all available cameras
            cam_to_worlds = {}
            for cam_id in AVAILABLE_CAM_LIST:
                cam_to_worlds[cam_id] = CheryCameraData.get_camera2worlds(
                    self.data_path, str(cam_id), self.start_timestep, self.end_timestep
                )

            # load SMPL parameters
            smpl_dict = joblib.load(
                os.path.join(self.data_path, "humanpose", "smpl.pkl")
            )
            frame_num = self.end_timestep - self.start_timestep

            smpl_human_all = {}
            for fi in tqdm(
                range(self.start_timestep, self.end_timestep), desc="Loading SMPL"
            ):
                for instance_id, ins_smpl in smpl_dict.items():
                    if instance_id not in smpl_human_all:
                        smpl_human_all[instance_id] = {
                            "smpl_quats": torch.zeros(
                                (frame_num, 24, 4), dtype=torch.float32
                            ),
                            "smpl_trans": torch.zeros(
                                (frame_num, 3), dtype=torch.float32
                            ),
                            "smpl_betas": torch.zeros(
                                (frame_num, 10), dtype=torch.float32
                            ),
                            "frame_valid": torch.zeros((frame_num), dtype=torch.bool),
                        }
                        smpl_human_all[instance_id]["smpl_quats"][:, :, 0] = 1.0
                    if ins_smpl["valid_mask"][fi]:
                        betas = ins_smpl["smpl"]["betas"][fi]
                        smpl_human_all[instance_id]["smpl_betas"][
                            fi - self.start_timestep
                        ] = betas

                        body_pose = ins_smpl["smpl"]["body_pose"][fi]
                        smpl_orient = ins_smpl["smpl"]["global_orient"][fi]
                        cam_depend = ins_smpl["selected_cam_idx"][fi].item()

                        c2w = cam_to_worlds[cam_depend][fi - self.start_timestep]
                        world_orient = (
                            c2w[:3, :3].to(smpl_orient.device) @ smpl_orient.squeeze()
                        )
                        smpl_quats = matrix_to_quaternion(
                            torch.cat([world_orient[None, ...], body_pose], dim=0)
                        )

                        ii = instances_info[str(instance_id)]["frame_annotations"][
                            "frame_idx"
                        ].index(fi)
                        o2w = np.array(
                            instances_info[str(instance_id)]["frame_annotations"][
                                "obj_to_world"
                            ][ii]
                        )
                        o2w = torch.from_numpy(
                            np.linalg.inv(lidar_to_world_start) @ o2w
                        )
                        # box_size = instances_info[str(instance_id)]['frame_annotations']["box_size"][ii]

                        smpl_human_all[instance_id]["smpl_quats"][
                            fi - self.start_timestep
                        ] = smpl_quats
                        smpl_human_all[instance_id]["smpl_trans"][
                            fi - self.start_timestep
                        ] = o2w[:3, 3]
                        smpl_human_all[instance_id]["frame_valid"][
                            fi - self.start_timestep
                        ] = True

            self.smpl_human_all = smpl_human_all


class CheryLiDARSource(SceneLidarSource):
    def __init__(
        self,
        lidar_data_config: OmegaConf,
        data_path: str,
        start_timestep: int,
        end_timestep: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(lidar_data_config, device=device)
        self.data_path = data_path
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.create_all_filelist()
        self.load_data()

    def create_all_filelist(self):
        """
        Create a list of all the files in the dataset.
        e.g., a list of all the lidar scans in the dataset.
        """
        lidar_type = self.data_cfg.lidar_type
        assert lidar_type in ["lidar", "mclidar"]
        logger.info(f"Using '{lidar_type}' for lidar data.")

        lidar_filepaths = []
        # lidar_pose_filepaths = []
        for t in range(self.start_timestep, self.end_timestep):
            lidar_filepaths.append(
                os.path.join(self.data_path, lidar_type, f"{t:03d}.bin")
            )
            # lidar_pose_filepaths.append(
            #     os.path.join(self.data_path, "lidar_pose", f"{t:03d}.txt")
            # )
        self.lidar_filepaths = np.array(lidar_filepaths)
        # self.lidar_pose_filepaths = np.array(lidar_pose_filepaths)

    def load_calibrations(self):
        """
        Load the calibration files of the dataset.
        e.g., lidar to world transformation matrices.
        """
        lidar_to_worlds = []

        # we tranform the poses w.r.t. the first timestep to make the origin of the
        # first lidar pose as the origin of the world coordinate system.
        ### 将第一个时间步的 LiDAR 坐标作为世界坐标系的原点
        lidar_to_world_start = np.loadtxt(
            os.path.join(self.data_path, "lidar_pose", f"{self.start_timestep:03d}.txt")
        )
        for t in range(self.start_timestep, self.end_timestep):
            lidar_to_world_current = np.loadtxt(
                os.path.join(self.data_path, "lidar_pose", f"{t:03d}.txt")
            )
            # compute lidar_to_world transformation
            lidar_to_world = (
                np.linalg.inv(lidar_to_world_start) @ lidar_to_world_current
            )
            lidar_to_worlds.append(lidar_to_world)

        self.lidar_to_worlds = torch.from_numpy(
            np.stack(lidar_to_worlds, axis=0)
        ).float()

    def load_lidar(self):
        """
        Load the lidar data of the dataset from the filelist.
        """
        origins, directions, ranges, laser_ids = [], [], [], []
        timesteps = []

        accumulated_num_original_rays = 0
        accumulated_num_rays = 0
        for t in trange(
            0, len(self.lidar_filepaths), desc="Loading lidar", dynamic_ncols=True
        ):
            lidar_info = np.fromfile(self.lidar_filepaths[t], dtype=np.float32).reshape(
                -1, 7
            )

            # select lidar points based on the laser id
            if self.data_cfg.only_use_top_lidar:
                # laser_ids: 按照奇瑞的雷达排布 0-4
                lidar_info = lidar_info[lidar_info[:, 6] == 0]

            original_length = len(lidar_info)
            accumulated_num_original_rays += original_length

            lidar_points = torch.from_numpy(lidar_info[:, :3]).float()
            lidar_origins = torch.zeros_like(lidar_points)

            lidar_origins = (
                self.lidar_to_worlds[t][:3, :3] @ lidar_origins.T
                + self.lidar_to_worlds[t][:3, 3:4]
            ).T
            lidar_points = (
                self.lidar_to_worlds[t][:3, :3] @ lidar_points.T
                + self.lidar_to_worlds[t][:3, 3:4]
            ).T
            lidar_ids = torch.from_numpy(lidar_info[:, 6]).float()  # 13全选出来为0的
            lidar_directions = lidar_points - lidar_origins
            lidar_ranges = torch.norm(lidar_directions, dim=-1, keepdim=True)
            lidar_directions = lidar_directions / lidar_ranges
            lidar_timestamp = torch.ones_like(lidar_ranges).squeeze(-1) * t
            accumulated_num_rays += len(lidar_ranges)

            origins.append(lidar_origins)
            directions.append(lidar_directions)
            ranges.append(lidar_ranges)
            timesteps.append(lidar_timestamp)
            laser_ids.append(lidar_ids)

        logger.info(
            f"Number of lidar rays: {accumulated_num_rays} "
            f"({accumulated_num_rays / accumulated_num_original_rays * 100:.2f}% of "
            f"{accumulated_num_original_rays} original rays)"
        )

        self.origins = torch.cat(origins, dim=0)
        self.directions = torch.cat(directions, dim=0)
        self.ranges = torch.cat(ranges, dim=0)
        self.visible_masks = torch.zeros_like(self.ranges).squeeze().bool()
        self.colors = torch.ones_like(self.directions)
        self.laser_ids = torch.cat(laser_ids, dim=0)

        self._timesteps = torch.cat(timesteps, dim=0)
        self.register_normalized_timestamps()

    def to(self, device: torch.device):
        super().to(device)

    def get_lidar_rays(self, time_idx: int) -> Dict[str, Tensor]:
        """
        Get the of rays for rendering at the given timestep.
        Args:
            time_idx: the index of the lidar scan to render.
        Returns:
            a dict of the sampled rays.
        """
        origins = self.origins[self.timesteps == time_idx]
        directions = self.directions[self.timesteps == time_idx]
        ranges = self.ranges[self.timesteps == time_idx]
        normalized_time = self.normalized_time[self.timesteps == time_idx]
        return {
            "lidar_origins": origins,
            "lidar_viewdirs": directions,
            "lidar_ranges": ranges,
            "lidar_normed_time": normalized_time,
            "lidar_mask": self.timesteps == time_idx,
        }

    def delete_invisible_pts(self) -> None:
        if self.visible_masks is not None:
            num_bf = self.origins.shape[0]
            self.origins = self.origins[self.visible_masks]
            self.directions = self.directions[self.visible_masks]
            self.ranges = self.ranges[self.visible_masks]
            self._timesteps = self._timesteps[self.visible_masks]
            self._normalized_time = self._normalized_time[self.visible_masks]
            self.colors = self.colors[self.visible_masks]
            logger.info(
                f"[Lidar] {num_bf - self.visible_masks.sum()} out of {num_bf} points are cleared. {self.visible_masks.sum()} points left."
            )
            self.visible_masks = None
        else:
            logger.info("[Lidar] No unvisible points to clear.")
