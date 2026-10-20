from typing import Dict, Union, Literal
import logging
import os
import numpy as np
from omegaconf import OmegaConf

import torch

from models.gaussians.basics import *
from datasets.base.scene_dataset import SceneDataset
from datasets.base.pixel_source import CameraData
from utils.camera import get_interp_novel_trajectories
from utils.misc import export_points_to_ply, import_str

logger = logging.getLogger()

DEBUG_PCD = False
if DEBUG_PCD:
    DEBUG_OUTPUT_DIR = "debug"
    os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)


class DrivingDatasetNovelView(SceneDataset):
    def __init__(
        self,
        data_cfg: OmegaConf,
    ) -> None:
        super().__init__(data_cfg)

        # AVAILABLE DATASETS:
        #   Chery:    11 cameras
        #   Qcraft:   13 cameras
        #   Waymo:    5 Cameras
        #   KITTI:    2 Cameras
        #   NuScenes: 6 Cameras
        #   ArgoVerse:7 Cameras
        #   PandaSet: 6 Cameras
        #   NuPlan:   8 Cameras
        self.type = self.data_cfg.dataset
        try:  # For Waymo, NuScenes, ArgoVerse, PandaSet
            self.data_path = os.path.join(
                self.data_cfg.data_root, f"{int(self.scene_idx):03d}"
            )
        except:  # For Chery, Qcraft, KITTI, NuPlan
            self.data_path = os.path.join(self.data_cfg.data_root, self.scene_idx)

        assert os.path.exists(self.data_path), f"{self.data_path} does not exist"
        if os.path.exists(os.path.join(self.data_path, "lidar_pose")):
            total_frames = len(os.listdir(os.path.join(self.data_path, "lidar_pose")))
        elif os.path.exists(os.path.join(self.data_path, "ego_pose")):
            total_frames = len(os.listdir(os.path.join(self.data_path, "ego_pose")))
        else:
            raise ValueError(
                "Unable to determine the total number of frames. Neither 'ego_pose' nor 'lidar_pose' directories found."
            )

        # ---- find the number of synchronized frames ---- #
        if self.data_cfg.end_timestep == -1:
            end_timestep = total_frames - 1
        else:
            end_timestep = self.data_cfg.end_timestep
        # to make sure the last timestep is included
        self.end_timestep = end_timestep + 1
        self.start_timestep = self.data_cfg.start_timestep

        # ---- create data source ---- #
        self.pixel_source, self.lidar_source = self.build_data_source()

        self.aabb = self.get_aabb()

        # ---- define train and test indices ---- #
        # note that the timestamps of the pixel source and the lidar source are the same in waymo dataset
        (
            self.train_timesteps,
            self.test_timesteps,
            self.train_indices,
            self.test_indices,
        ) = self.split_train_test()

        # # ---- create split wrappers ---- #
        # image_sets = self.build_split_wrapper()
        # self.train_image_set, self.test_image_set, self.full_image_set = image_sets

    @property
    def instance_num(self):
        return len(self.pixel_source.instances_pose[0])

    @property
    def frame_num(self):
        return self.pixel_source.num_frames

    def build_split_wrapper(self):
        #     train_image_set = SplitWrapper(
        #         datasource=self.pixel_source,
        #         # train_indices are img indices, so the length is num_cams * num_train_timesteps
        #         split_indices=self.train_indices,
        #         split="train",
        #     )
        #     full_image_set = SplitWrapper(
        #         datasource=self.pixel_source,
        #         # cover all the images
        #         split_indices=np.arange(self.pixel_source.num_imgs).tolist(),
        #         split="full",
        #     )
        #     test_image_set = None
        #     if len(self.test_indices) > 0:
        #         test_image_set = SplitWrapper(
        #             datasource=self.pixel_source,
        #             # test_indices are img indices, so the length is num_cams * num_test_timesteps
        #             split_indices=self.test_indices,
        #             split="test",
        #         )
        #     image_sets = (train_image_set, test_image_set, full_image_set)
        #     return image_sets
        pass

    def build_data_source(self):
        """
        Create the data source for the dataset.
        """
        # ---- create pixel source ---- #
        pixel_source = import_str(self.data_cfg.pixel_source.type)(
            self.data_cfg.dataset,
            self.data_cfg.pixel_source,
            self.data_path,
            self.start_timestep,
            self.end_timestep,
            device=self.device,
            render_only=True,
        )
        pixel_source.to(self.device)

        # ---- create lidar source ---- #
        lidar_source = None
        # if self.data_cfg.lidar_source.load_lidar:
        #     lidar_source = import_str(self.data_cfg.lidar_source.type)(
        #         self.data_cfg.lidar_source,
        #         self.data_path,
        #         self.start_timestep,
        #         self.end_timestep,
        #         device=self.device,
        #     )
        #     lidar_source.to(self.device)
        #     assert (
        #         pixel_source._unique_normalized_timestamps
        #         - lidar_source._unique_normalized_timestamps
        #     ).abs().sum().item() == 0.0, "The timestamps of the pixel source and the lidar source are not synchronized"
        return pixel_source, lidar_source

    def split_train_test(self):
        if self.data_cfg.pixel_source.test_image_stride != 0:
            test_timesteps = np.arange(
                # it makes no sense to have test timesteps before the start timestep
                self.data_cfg.pixel_source.test_image_stride,
                self.num_img_timesteps,
                self.data_cfg.pixel_source.test_image_stride,
            )
        else:
            test_timesteps = []
        train_timesteps = np.array(
            [i for i in range(self.num_img_timesteps) if i not in test_timesteps]
        )

        # propagate the train and test timesteps to the train and test indices
        train_indices, test_indices = [], []
        for t in range(self.num_img_timesteps):
            if t in train_timesteps:
                for cam in range(self.pixel_source.num_cams):
                    train_indices.append(t * self.pixel_source.num_cams + cam)
            elif t in test_timesteps:
                for cam in range(self.pixel_source.num_cams):
                    test_indices.append(t * self.pixel_source.num_cams + cam)

        # Again, training and testing indices are indices into the full dataset
        # train_indices are img indices, so the length is num_cams * num_train_timesteps
        # but train_timesteps are timesteps, so the length is num_train_timesteps (len(unique_train_timestamps))
        return train_timesteps, test_timesteps, train_indices, test_indices

    def get_novel_render_traj(
        self,
        traj_type: str,
        target_frames: int = 100,
        traj_path: str = None,
    ) -> torch.Tensor:
        """
        Get multiple novel trajectories of the scene for rendering.

        Args:
            traj_type: str
                The trajectory type to generate
            target_frames: int
                The total number of frames for each novel trajectory

        Returns:
            Dict[str, torch.Tensor]: A dictionary where keys are trajectory types and values
            are the generated novel trajectories, each of shape (target_frames, 4, 4)
        """
        if traj_type == "custom":
            # TODO: 支持自定义轨迹
            # novel_trajs[traj_type] = load_custom_trajectory(traj_path)
            pass
        else:
            novel_traj = get_interp_novel_trajectories(
                dataset_type=self.type,
                ego_to_worlds=self.pixel_source.ego_to_worlds,
                traj_type=traj_type,
                target_frames=target_frames,
            )
        return novel_traj

    def prepare_novel_view_render_data(
        self,
        traj: torch.Tensor,
        target_cam_data: CameraData,
        is_original_traj: bool = False,
    ) -> list:
        return self.pixel_source.prepare_novel_view_render_data(
            dataset_type=self.type,
            ego_novel_traj=traj,
            target_cam_data=target_cam_data,
            is_original_traj=is_original_traj,
        )

    def prepare_online_render_data(
        self,
        traj: torch.Tensor,
        target_cam_data: CameraData,
        frame_id: int,
    ) -> list:
        return self.pixel_source.prepare_online_render_data(
            dataset_type=self.type,
            ego_novel_traj=traj,
            target_cam_data=target_cam_data,
            frame_id=frame_id,
        )
    
    def load_specified_cameras(
        self, cam_ids: List[int], downscales: List[float]
    ) -> Dict[int, CameraData]:
        camera_data = self.pixel_source.load_specified_cameras(cam_ids, downscales)
        return camera_data
