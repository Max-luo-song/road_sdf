from typing import Dict, Union, Literal
import logging
import os
import cv2
import numpy as np
from tqdm import trange, tqdm
from omegaconf import OmegaConf
from enum import Enum

import torch
from torch import Tensor

from models.gaussians.basics import *
from datasets.base.scene_dataset import ModelType
from datasets.base.scene_dataset import SceneDataset
from datasets.base.split_wrapper import SplitWrapper
from datasets.utils.base_utils import project_points_to_image
from utils.visualization import get_layout
from utils.geometry import transform_points
from utils.misc import export_points_to_ply, import_str
from utils.camera import get_interp_novel_trajectories
import open3d as o3d
logger = logging.getLogger()

DEBUG_PCD = False
if DEBUG_PCD:
    DEBUG_OUTPUT_DIR = "debug"
    os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)

NAME_TO_NODE = {
    "RigidNodes": ModelType.RigidNodes,
    "SMPLNodes": ModelType.SMPLNodes,
    "DeformableNodes": ModelType.DeformableNodes,
}

class DepthMode(Enum):
    SINGLE_FRAME_RAW: str = "sf_raw"
    SINGLE_FRAME_STATIC: str = "sf_static"
    MULTI_FRAME_STATIC: str = "mf_static"

def parse_depth_mode(value: str, default: DepthMode = DepthMode.SINGLE_FRAME_STATIC) -> DepthMode:
    try:
        return DepthMode(value)
    except ValueError:
        logger.warn(f"Invalid depth_mode: '{value}'. Using default: {default.value}")
        return default

class DrivingDataset(SceneDataset):
    def __init__(
        self,
        data_cfg: OmegaConf,
        weather_type: str = None,
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
        self.weather_type = weather_type
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

        # ---- create layout for visualization ---- #
        self.layout = get_layout(self.type)

        # ---- create data source ---- #
        self.pixel_source, self.lidar_source = self.build_data_source()
        # assert self.pixel_source is not None and self.lidar_source is not None, \
        #     "Must have both pixel source and lidar source"

        self.depth_mode = None
        if self.lidar_source is not None:  # gls
            depth_mode = self.lidar_source.data_cfg.get("depth_mode", None)
            self.depth_mode = parse_depth_mode(depth_mode)
            logger.info(f"Projecting LiDAR points onto images with depth mode: {self.depth_mode.value}")

            self.project_lidar_pts_on_images(delete_out_of_view_points=True)

        self.aabb = self.get_aabb()

        # ---- define train and test indices ---- #
        # note that the timestamps of the pixel source and the lidar source are the same in waymo dataset
        (
            self.train_timesteps,
            self.test_timesteps,
            self.train_indices,
            self.test_indices,
        ) = self.split_train_test()

        # ---- create split wrappers ---- #
        image_sets = self.build_split_wrapper()
        self.train_image_set, self.test_image_set, self.full_image_set = image_sets

        # debug use
        # self.seg_dynamic_instances_in_lidar_frame(-1, frame_idx=0)
        # self.get_init_objects()

    @property
    def instance_num(self):
        return len(self.pixel_source.instances_pose[0])

    @property
    def frame_num(self):
        return self.pixel_source.num_frames

    def get_instance_infos(self):
        return (
            self.pixel_source.instances_pose.clone(),
            self.pixel_source.instances_size.clone(),
            self.pixel_source.instances_model_types.clone(),
            self.pixel_source.per_frame_instance_mask.clone(),
        )

    def build_split_wrapper(self):
        train_image_set = SplitWrapper(
            datasource=self.pixel_source,
            # train_indices are img indices, so the length is num_cams * num_train_timesteps
            split_indices=self.train_indices,
            split="train",
        )
        full_image_set = SplitWrapper(
            datasource=self.pixel_source,
            # cover all the images
            split_indices=np.arange(self.pixel_source.num_imgs).tolist(),
            split="full",
        )
        test_image_set = None
        if len(self.test_indices) > 0:
            test_image_set = SplitWrapper(
                datasource=self.pixel_source,
                # test_indices are img indices, so the length is num_cams * num_test_timesteps
                split_indices=self.test_indices,
                split="test",
            )
        image_sets = (train_image_set, test_image_set, full_image_set)
        return image_sets

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
            weather_type=self.weather_type,
            device=self.device,
        )
        pixel_source.to(self.device)

        # ---- create lidar source ---- #
        lidar_source = None
        if self.data_cfg.lidar_source.load_lidar:
            lidar_source = import_str(self.data_cfg.lidar_source.type)(
                self.data_cfg.lidar_source,
                self.data_path,
                self.start_timestep,
                self.end_timestep,
                device=self.device,
            )
            lidar_source.to(self.device)
            assert (
                pixel_source._unique_normalized_timestamps
                - lidar_source._unique_normalized_timestamps
            ).abs().sum().item() == 0.0, "The timestamps of the pixel source and the lidar source are not synchronized"
        return pixel_source, lidar_source

    ### 其实相当于从雷达数据中采样，设定一个采样值，采样值通常应该小于雷达点数，如果大于雷达点数采样值就是雷达点数
    def get_lidar_samples(
        self,
        num_samples: float = None,
        downsample_factor: float = None,
        return_color=False,
        return_normalized_time=False,
        device: torch.device = torch.device("cpu"),
    ) -> Tensor:
        assert (
            self.lidar_source is not None
        ), "Must have lidar source if you want to get init pcd"
        assert (num_samples is None) != (
            downsample_factor is None
        ), "Must provide either num_samples or downsample_factor, but not both"  # 不能同时提供两者
        if downsample_factor is not None:
            num_samples = int(len(self.lidar_source.pts_xyz) / downsample_factor)
        if num_samples > len(self.lidar_source.pts_xyz):
            logger.warning(
                f"num_samples {num_samples} is larger than the number of points {len(self.lidar_source.pts_xyz)}"
            )
            num_samples = len(self.lidar_source.pts_xyz)

        # randomly sample points  随机采样
        sampled_idx = torch.randperm(len(self.lidar_source.pts_xyz))[:num_samples]
        sampled_pts = self.lidar_source.pts_xyz[sampled_idx].to(device)

        # get color if needed
        sampled_color = None
        if return_color:
            sampled_color = self.lidar_source.colors[sampled_idx].to(device)

        sampled_time = None
        if return_normalized_time:
            sampled_time = self.lidar_source._normalized_time[sampled_idx].to(device)
            sampled_time = sampled_time[..., None]

        return sampled_pts, sampled_color, sampled_time

    def seg_dynamic_instances_in_lidar_frame(
        self, instance_ids: Union[int, list], frame_idx: int
    ):
        if isinstance(instance_ids, int):
            instance_num = len(self.pixel_source.instances_pose[frame_idx])
            assert (
                instance_ids < instance_num
            ), f"instance_id {instance_ids} is larger than the number of instances {instance_num}"
            if instance_ids == -1:
                instance_ids = list(range(instance_num))
            else:
                instance_ids = [instance_ids]
        elif isinstance(instance_ids, list):
            instance_ids = instance_ids

        # get the lidar points
        lidar_dict = self.lidar_source.get_lidar_rays(frame_idx)
        lidar_pts = (
            lidar_dict["lidar_origins"]
            + lidar_dict["lidar_viewdirs"] * lidar_dict["lidar_ranges"]
        )
        valid_mask = torch.zeros_like(lidar_pts[:, 0], dtype=torch.bool)
        for instance_id in instance_ids:
            is_valid_instance = self.pixel_source.per_frame_instance_mask[
                frame_idx, instance_id
            ]
            if not is_valid_instance:
                continue
            # get the pose of the instance at the given frame
            o2w = self.pixel_source.instances_pose[frame_idx, instance_id]
            o_size = self.pixel_source.instances_size[instance_id]

            # transform the lidar points to the instance's coordinate system
            # instance_pose [4, 4], pts [N, 3]
            w2o = torch.inverse(o2w)
            o_pts = transform_points(lidar_pts, w2o)
            # get the mask of the points that are inside the instance's bounding box
            mask = (
                (o_pts[:, 0] > -o_size[0] / 2)
                & (o_pts[:, 0] < o_size[0] / 2)
                & (o_pts[:, 1] > -o_size[1] / 2)
                & (o_pts[:, 1] < o_size[1] / 2)
                & (o_pts[:, 2] > -o_size[2] / 2)
                & (o_pts[:, 2] < o_size[2] / 2)
            )
            valid_mask = valid_mask | mask

        valid_points = lidar_pts[valid_mask]
        valid_colors = self.lidar_source.colors[lidar_dict["lidar_mask"]][valid_mask]

        if DEBUG_PCD:
            export_points_to_ply(
                valid_points,
                valid_colors,
                save_path=os.path.join(DEBUG_OUTPUT_DIR, "vehicle_lidar_pts.ply"),
            )
            export_points_to_ply(
                lidar_pts,
                self.lidar_source.colors[lidar_dict["lidar_mask"]],
                save_path=os.path.join(DEBUG_OUTPUT_DIR, "lidar_pts.ply"),
            )

    def get_init_objects(
        self,
        cur_node_type: Literal["RigidNodes", "DeformableNodes", "TrafficLightNodes"],
        instance_max_pts: int = 5000,
        only_moving: bool = True,
        traj_length_thres: float = 0.5,
        exclude_smpl: bool = False,
    ):
        """
        return:
            instances_dict: Dict[int, Dict[str, Tensor]]
                keys: instance_id
                values: Dict[str, Tensor]
                    keys: "pts", "colors", "num_pts", "flows"(Optional)
                    values: Tensor

        NOTE: pts are in object coordinate system
        """
        if self.type == "KITTI":
            traj_length_thres = 5.0
            logger.info(
                f"For KITTI dataset, the trajectory length threshold is set \
                to {traj_length_thres} to filter out noisy short trajectories of static objects"
            )

        instance_dict = {}
        for fi in range(self.frame_num):
            if self.lidar_source != None:
                lidar_dict = self.lidar_source.get_lidar_rays(fi)
                lidar_pts = (
                    lidar_dict["lidar_origins"]
                    + lidar_dict["lidar_viewdirs"] * lidar_dict["lidar_ranges"]
                )
            else:
                lidar_dict = None
                lidar_pts = np.random.rand(20000, 3).astype(np.float64)
                lidar_pts = torch.from_numpy(lidar_pts).to(self.device)

            for ins_id in range(self.instance_num):
                instance_active = self.pixel_source.per_frame_instance_mask[fi, ins_id]
                o_type = self.pixel_source.instances_model_types[ins_id].item()

                if not instance_active:
                    continue

                if cur_node_type == "DeformableNodes":
                    if not (
                        o_type == ModelType.DeformableNodes
                        or o_type == ModelType.SMPLNodes
                    ):
                        continue
                elif cur_node_type == "RigidNodes":
                    if not o_type == ModelType.RigidNodes:
                        continue
                elif cur_node_type == "TrafficLightNodes":
                    if not o_type == ModelType.TrafficLightNodes:
                        continue

                if exclude_smpl:
                    # objects with smpl pose will be modeled by SMPLNodes
                    assert (
                        cur_node_type == "DeformableNodes"
                    ), "Only exclude SMPL for DeformableNodes"
                    true_id = self.pixel_source.instances_true_id[ins_id].item()
                    if hasattr(self.pixel_source, 'smpl_human_all') and self.pixel_source.smpl_human_all is not None:
                        if true_id in self.pixel_source.smpl_human_all.keys():
                            continue

                if ins_id not in instance_dict:
                    instance_dict[ins_id] = {
                        "node_type": cur_node_type,
                        "pts": [],
                        "colors": [],
                        "flows": [],
                    }
                # get the pose of the instance at the given frame
                o2w = self.pixel_source.instances_pose[fi, ins_id]
                # print("fi:{}, ins_id:{}".format(fi, ins_id))
                o_size = self.pixel_source.instances_size[ins_id]
                # convert the lidar points to the instance's coordinate system
                # w2o = torch.inverse(o2w)
                w2o = torch.linalg.pinv(o2w, rcond=1e-6)
                w2o = w2o.to(lidar_pts.dtype)  # 将变换矩阵转换为与点云相同的类型
                o_pts = transform_points(lidar_pts, w2o)
                # get the mask of the points that are inside the instance's bounding box
                mask = (
                    (o_pts[:, 0] > -o_size[0] / 2)
                    & (o_pts[:, 0] < o_size[0] / 2)
                    & (o_pts[:, 1] > -o_size[1] / 2)
                    & (o_pts[:, 1] < o_size[1] / 2)
                    & (o_pts[:, 2] > -o_size[2] / 2)
                    & (o_pts[:, 2] < o_size[2] / 2)
                )
                valid_pts = o_pts[mask]
                if self.lidar_source != None:
                    valid_colors = self.lidar_source.colors[lidar_dict["lidar_mask"]][
                        mask
                    ]
                    # valid_flows = lidar_dict["lidar_flows"][mask]
                else:
                    valid_colors = torch.from_numpy(
                        np.random.rand(valid_pts.shape[0], 3).astype(np.float64)
                    ).to(self.device)
                    # valid_flows = torch.from_numpy(np.random.rand(valid_pts.shape[0], 3).astype(np.float64)).to(self.device)
                instance_dict[ins_id]["pts"].append(valid_pts)
                instance_dict[ins_id]["colors"].append(valid_colors)
                # instance_dict[ins_id]["flows"].append(valid_flows)

        logger.info(f"Aggregating lidar points across {self.frame_num} frames")
        min_points_threshold = 100  # 如果点云数量少于这个值，使用 bounding box 生成点云
        for ins_id in instance_dict:
            instance_dict[ins_id]["pts"] = torch.cat(
                instance_dict[ins_id]["pts"], dim=0
            )
            instance_dict[ins_id]["colors"] = torch.cat(
                instance_dict[ins_id]["colors"], dim=0
            )
            # instance_dict[ins_id]["flows"] = torch.cat(instance_dict[ins_id]["flows"], dim=0)
            instance_dict[ins_id]["num_pts"] = instance_dict[ins_id]["pts"].shape[0]
            
            # 如果点云数量太少（可能因为只有前视 LiDAR，后视车辆没有点云），使用 bounding box 生成点云
            if instance_dict[ins_id]["num_pts"] < min_points_threshold:
                logger.warning(
                    f"Instance {ins_id} has only {instance_dict[ins_id]['num_pts']} lidar points, "
                    f"generating points from bounding box"
                )
                # 获取该实例的 bounding box 尺寸
                o_size = self.pixel_source.instances_size[ins_id]
                # 在 bounding box 内均匀采样点云
                num_generated_pts = max(instance_max_pts // 2, min_points_threshold)
                generated_pts = torch.rand(num_generated_pts, 3, device=self.device) - 0.5
                generated_pts = generated_pts * o_size.unsqueeze(0)  # 缩放到 bounding box 尺寸
                # 使用随机颜色或平均颜色
                if instance_dict[ins_id]["num_pts"] > 0:
                    avg_color = instance_dict[ins_id]["colors"].mean(dim=0)
                    generated_colors = avg_color.unsqueeze(0).repeat(num_generated_pts, 1)
                else:
                    generated_colors = torch.rand(num_generated_pts, 3, device=self.device)
                
                # 合并原始点云和生成的点云
                if instance_dict[ins_id]["num_pts"] > 0:
                    instance_dict[ins_id]["pts"] = torch.cat([
                        instance_dict[ins_id]["pts"], generated_pts
                    ], dim=0)
                    instance_dict[ins_id]["colors"] = torch.cat([
                        instance_dict[ins_id]["colors"], generated_colors
                    ], dim=0)
                else:
                    instance_dict[ins_id]["pts"] = generated_pts
                    instance_dict[ins_id]["colors"] = generated_colors
                instance_dict[ins_id]["num_pts"] = instance_dict[ins_id]["pts"].shape[0]
                logger.info(
                    f"Instance {ins_id} now has {instance_dict[ins_id]['num_pts']} points "
                    f"({instance_dict[ins_id]['num_pts'] - num_generated_pts} from lidar, "
                    f"{num_generated_pts} generated from bbox)"
                )
            
            if instance_dict[ins_id]["num_pts"] > instance_max_pts:
                # randomly sample points
                sampled_idx = torch.randperm(instance_dict[ins_id]["num_pts"])[
                    :instance_max_pts
                ]
                instance_dict[ins_id]["pts"] = instance_dict[ins_id]["pts"][sampled_idx]
                instance_dict[ins_id]["colors"] = instance_dict[ins_id]["colors"][
                    sampled_idx
                ]
                # instance_dict[ins_id]["flows"] = instance_dict[ins_id]["flows"][sampled_idx]
                instance_dict[ins_id]["num_pts"] = instance_max_pts
            logger.info(
                f"Instance {ins_id} has {instance_dict[ins_id]['num_pts']} lidar sample points"
            )

        if only_moving:
            # consider only the instances with non-zero flows
            logger.info(f"Filtering out the instances with non-moving trajectories")
            new_instance_dict = {}
            for k, v in instance_dict.items():
                if v["num_pts"] > 0:
                    # flows = v["flows"]
                    # if flows.norm(dim=-1).mean() > moving_thres:
                    #     v.pop("flows")
                    #     new_instance_dict[k] = v
                    #     logger.info(f"Instance {k} has {v['num_pts']} lidar sample points")
                    frame_info = self.pixel_source.per_frame_instance_mask[:, k]
                    instances_pose = self.pixel_source.instances_pose[:, k]
                    instances_trans = instances_pose[:, :3, 3]
                    valid_trans = instances_trans[frame_info]
                    traj_length = valid_trans[1:] - valid_trans[:-1]
                    traj_length = torch.norm(traj_length, dim=-1).sum()
                    if traj_length > traj_length_thres:
                        new_instance_dict[k] = v
                        logger.info(
                            f"Instance {k} has {v['num_pts']} lidar sample points"
                        )
            instance_dict = new_instance_dict

        # get instance info
        for ins_id in instance_dict:
            instance_dict[ins_id]["poses"] = self.pixel_source.instances_pose[:, ins_id]
            instance_dict[ins_id]["size"] = self.pixel_source.instances_size[ins_id]
            instance_dict[ins_id]["frame_info"] = (
                self.pixel_source.per_frame_instance_mask[:, ins_id]
            )

        if DEBUG_PCD:
            output_dir = os.path.join(DEBUG_OUTPUT_DIR, "aggregated_instance_lidar_pts")
            os.makedirs(output_dir, exist_ok=True)
            for ins_id in instance_dict:
                export_points_to_ply(
                    instance_dict[ins_id]["pts"],
                    instance_dict[ins_id]["colors"],
                    save_path=os.path.join(output_dir, f"ID={ins_id}.ply"),
                )
        return instance_dict

    def get_init_smpl_objects(
        self, only_moving: bool = False, traj_length_thres: float = 0.5
    ):
        instance_dict = {}
        """
        instance_dict = {
            ins_id: {
                "node_type": str, 
                "pts": Tensor, [frame_num, num_pts, 3]
                "colors": Tensor, [frame_num, num_pts, 3]
                "quats": Tensor, [frame_num, 4]
                "trans": Tensor, [frame_num, 3]
                "size": Tensor, [3]
                "frame_info": Tensor, [frame_num]
        }
        """

        for ins_id in range(self.instance_num):
            true_id = self.pixel_source.instances_true_id[ins_id].item()
            if not hasattr(self.pixel_source, 'smpl_human_all') or self.pixel_source.smpl_human_all is None:
                continue
            if true_id in self.pixel_source.smpl_human_all.keys():
                if self.pixel_source.smpl_human_all[true_id]["frame_valid"].sum() == 0:
                    continue
                smpl_trans = self.pixel_source.smpl_human_all[true_id]["smpl_trans"]
                frame_info = self.pixel_source.smpl_human_all[true_id]["frame_valid"]
                if only_moving and traj_length_thres > 0:
                    # compute the distance between two consecutive frames
                    traj_length = (
                        smpl_trans[frame_info][1:] - smpl_trans[frame_info][:-1]
                    )
                    traj_length = torch.norm(traj_length, dim=-1).sum()
                    if traj_length < traj_length_thres:
                        continue
                smpl_quats = self.pixel_source.smpl_human_all[true_id]["smpl_quats"]
                smpl_betas = self.pixel_source.smpl_human_all[true_id]["smpl_betas"]
                size = self.pixel_source.instances_size[ins_id]
                # NOTE: set the first frame's betas as the betas of the instance
                first_frame_betas = smpl_betas[frame_info][0]

                collected_lidar_pts = []
                collected_lidar_colors = []
                for fi in range(self.frame_num):
                    lidar_dict = self.lidar_source.get_lidar_rays(fi)
                    lidar_pts = (
                        lidar_dict["lidar_origins"]
                        + lidar_dict["lidar_viewdirs"] * lidar_dict["lidar_ranges"]
                    )
                    instance_active = self.pixel_source.per_frame_instance_mask[
                        fi, ins_id
                    ]
                    if not instance_active:
                        continue

                    # get the pose of the instance at the given frame
                    o2w = self.pixel_source.instances_pose[fi, ins_id]
                    o_size = self.pixel_source.instances_size[ins_id]
                    # convert the lidar points to the instance's coordinate system
                    w2o = torch.inverse(o2w)
                    o_pts = transform_points(lidar_pts, w2o)
                    # get the mask of the points that are inside the instance's bounding box
                    mask = (
                        (o_pts[:, 0] > -o_size[0] / 2)
                        & (o_pts[:, 0] < o_size[0] / 2)
                        & (o_pts[:, 1] > -o_size[1] / 2)
                        & (o_pts[:, 1] < o_size[1] / 2)
                        & (o_pts[:, 2] > -o_size[2] / 2)
                        & (o_pts[:, 2] < o_size[2] / 2)
                    )
                    valid_pts = o_pts[mask]
                    valid_colors = self.lidar_source.colors[lidar_dict["lidar_mask"]][
                        mask
                    ]
                    # valid_flows = lidar_dict["lidar_flows"][mask]
                    collected_lidar_pts.append(valid_pts)
                    collected_lidar_colors.append(valid_colors)

                instance_dict[ins_id] = {
                    "node_type": "SMPLNodes",
                    "smpl_quats": smpl_quats,  # [frame_num, 24, 4]
                    "smpl_trans": smpl_trans,  # [frame_num, 3]
                    "smpl_betas": first_frame_betas,  # [10]
                    "size": size,  # [3]
                    "frame_info": frame_info,  # [frame_num]
                    "pts": torch.cat(collected_lidar_pts, dim=0),
                    "colors": torch.cat(collected_lidar_colors, dim=0),
                }

        return instance_dict

    def filter_pts_in_boxes(
        self,
        seed_pts: Tensor,
        valid_instances_dict: Dict[int, Dict[str, Tensor]],
        seed_colors: Tensor = None,
        seed_time: Tensor = None,
    ):
        """
        This function is used to filter out the points that are inside the bounding boxes of the instances
        """
        if DEBUG_PCD:
            os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
            export_points_to_ply(
                seed_pts,
                seed_colors,
                save_path=os.path.join(DEBUG_OUTPUT_DIR, "original_seed_pts.ply"),
            )
        valid_instance_keys = valid_instances_dict.keys()

        inside_mask = torch.zeros_like(seed_pts[:, 0], dtype=torch.bool)
        for fi in range(self.frame_num):
            for ins_id in valid_instance_keys:
                instance_active = self.pixel_source.per_frame_instance_mask[fi, ins_id]
                if not instance_active:
                    continue
                # get the pose of the instance at the given frame
                o2w = self.pixel_source.instances_pose[fi, ins_id].to(seed_pts.device)
                o_size = self.pixel_source.instances_size[ins_id].to(seed_pts.device)
                # convert the lidar points to the instance's coordinate system
                # w2o = torch.inverse(o2w)
                w2o = torch.linalg.pinv(o2w, rcond=1e-6)
                o_pts = transform_points(seed_pts, w2o)
                # get the mask of the points that are inside the instance's bounding box
                mask = (
                    (o_pts[:, 0] > -o_size[0] / 2)
                    & (o_pts[:, 0] < o_size[0] / 2)
                    & (o_pts[:, 1] > -o_size[1] / 2)
                    & (o_pts[:, 1] < o_size[1] / 2)
                    & (o_pts[:, 2] > -o_size[2] / 2)
                    & (o_pts[:, 2] < o_size[2] / 2)
                )
                inside_mask = inside_mask | mask

        # filter out the points that are inside the bounding boxes
        seed_pts = seed_pts[~inside_mask]
        if seed_colors is not None:
            seed_colors = seed_colors[~inside_mask]
        if seed_time is not None:
            seed_time = seed_time[~inside_mask]

        if DEBUG_PCD:
            export_points_to_ply(
                seed_pts,
                seed_colors,
                save_path=os.path.join(DEBUG_OUTPUT_DIR, "filtered_seed_pts.ply"),
            )

            for fi in range(self.frame_num):
                if fi % 10 != 0:
                    continue
                frame_save_dir = os.path.join(DEBUG_OUTPUT_DIR, f"frame_{fi}")
                os.makedirs(frame_save_dir, exist_ok=True)
                for ins_id in valid_instances_dict:
                    # print number of points
                    # print(f"Frame {fi}, Instance {ins_id} has {valid_instances_dict[ins_id]['pts'].shape[0]} points")
                    o2w = self.pixel_source.instances_pose[fi, ins_id]
                    pts_in_obj = valid_instances_dict[ins_id]["pts"]
                    # rotate the points back to the world coordinate system
                    pts_in_world = transform_points(pts_in_obj, o2w)
                    export_points_to_ply(
                        pts_in_world,
                        valid_instances_dict[ins_id]["colors"],
                        save_path=os.path.join(frame_save_dir, f"ID={ins_id}.ply"),
                    )

        return {"pts": seed_pts, "colors": seed_colors, "time": seed_time}

    ### NOTE(gls):提取去掉动态点云的背景点云（根据2d dynamic mask）——根据实验效果优于3d box版本
    def project_aggregated_lidar_pts_wo_box(self, delete_out_of_view_points=True):
        """
        Project the lidar points on the images and attribute the color of the nearest pixel to the lidar point.

        Args:
            delete_out_of_view_points: bool
                If True, the lidar points that are not visible from the camera will be removed.
        """
        aggregated_lidar_points_world = []
        aggregated_lidar_points_colors = []

        for idx, cam in enumerate(self.pixel_source.camera_data.values()):
            for frame_idx in tqdm(
                range(len(cam)),
                desc="Projecting lidar pts on images for camera {}".format(cam.cam_name),
                dynamic_ncols=True,
            ):
                normed_time = self.pixel_source.normalized_time[frame_idx]

                # get lidar depth on image plane
                closest_lidar_idx = self.lidar_source.find_closest_timestep(normed_time)

                lidar_infos = self.lidar_source.get_lidar_rays(closest_lidar_idx)
                lidar_points_world = ( # 单帧雷达点
                    lidar_infos["lidar_origins"]
                    + lidar_infos["lidar_viewdirs"] * lidar_infos["lidar_ranges"]
                )

                # project lidar points to the image plane
                if cam.undistort:
                    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                        cam.intrinsics[frame_idx].cpu().numpy(),
                        cam.distortions[frame_idx].cpu().numpy(),
                        (cam.WIDTH, cam.HEIGHT),
                        alpha=1,
                    )
                    intrinsic_4x4 = torch.nn.functional.pad(
                        torch.from_numpy(new_camera_matrix), (0, 1, 0, 1)
                    ).to(self.device)
                else:
                    intrinsic_4x4 = torch.nn.functional.pad(
                        cam.intrinsics[frame_idx], (0, 1, 0, 1)
                    )
                intrinsic_4x4[3, 3] = 1.0
                lidar2img = intrinsic_4x4 @ cam.cam_to_worlds[frame_idx].inverse()
                lidar_points_img = (lidar2img[:3, :3] @ lidar_points_world.T + lidar2img[:3, 3:4]).T  # (num_pts, 3)
                depth = lidar_points_img[:, 2]
                cam_points = lidar_points_img[:, :2] / (depth.unsqueeze(-1) + 1e-6)  # (num_pts, 2)
                valid_mask = (
                    (cam_points[:, 0] >= 0)
                    & (cam_points[:, 0] < cam.WIDTH)
                    & (cam_points[:, 1] >= 0)
                    & (cam_points[:, 1] < cam.HEIGHT)
                    & (depth > 0)
                )  # (num_pts, )
                
                # syc: 更改逻辑 - 使用 dynamic_mask 过滤动态点云
                # 2. 获取动态掩码 (H, W)
                # 确保 dynamic_mask 是 bool 类型，并且设备一致
                dynamic_mask = cam.dynamic_masks[frame_idx]
                if dynamic_mask.device != self.device:
                    dynamic_mask = dynamic_mask.to(self.device)
                # dynamic_mask = dynamic_mask.bool()
                # 3. 初始化最终保留的掩码，默认为 False
                keep_mask = torch.zeros(lidar_points_world.shape[0], device=self.device, dtype=torch.bool)
                
                # 4. 找到在有效视口内的点
                valid_indices = torch.where(valid_mask)[0]
                if valid_indices.shape[0] > 0:
                    # 获取视口内点的像素坐标 (整数)
                    pixel_y = cam_points[valid_indices, 1].long()
                    pixel_x = cam_points[valid_indices, 0].long()
                    
                    # 检查这些坐标在 dynamic_mask 中的值
                    is_dynamic_at_pixel = dynamic_mask[pixel_y, pixel_x]
                    
                    # 我们只保留 **静态** 的点 (dynamic_mask 为 False)
                    static_indices_in_valid = valid_indices[~is_dynamic_at_pixel]
                    
                    # 标记这些静态点为保留
                    keep_mask[static_indices_in_valid] = True
                
                # 5. 处理颜色信息
                # 初始化全0颜色
                current_frame_colors = torch.zeros(lidar_points_world.shape[0], 3, device=self.device, dtype=cam.images[frame_idx].dtype)
                
                if valid_indices.shape[0] > 0:
                    # 获取所有视口内点的坐标 (用于取色)
                    all_pixel_y = cam_points[valid_indices, 1].long()
                    all_pixel_x = cam_points[valid_indices, 0].long()
                    
                    # 取色
                    colors_in_view = cam.images[frame_idx][all_pixel_y, all_pixel_x]
                    
                    # 将取到的颜色赋给原始数组的对应位置
                    current_frame_colors[valid_indices] = colors_in_view

                # 6. 根据掩码筛选点云和颜色
                static_lidar_points_world = lidar_points_world[keep_mask]
                static_lidar_points_colors = current_frame_colors[keep_mask]
                
                aggregated_lidar_points_world.append(static_lidar_points_world)
                aggregated_lidar_points_colors.append(static_lidar_points_colors)

        # syc: aggregate lidar points from all frames to get dense depth maps
        aggregated_lidar_points_world = torch.cat(aggregated_lidar_points_world, dim=0)
        aggregated_lidar_points_colors = torch.cat(aggregated_lidar_points_colors, dim=0)
        return {"pts": aggregated_lidar_points_world, "colors": aggregated_lidar_points_colors}

    ### NOTE(gls): exclude all the road pointclouds
    def filter_pts_in_road(
        self,
        seed_pts: Tensor,
        seed_colors: Tensor = None,
        TYPE: str = None,
        road_only: bool = False,
        seed_time: Tensor = None,
    ):
        if TYPE == "IMAGE":
            # 使用一个布尔张量来记录一个点是否在任意一帧的任意相机中被识别为路面点
            on_road_mask = torch.zeros_like(seed_pts[:, 0], dtype=torch.bool)
            
            # 遍历所有有效相机
            for cam_id in self.pixel_source.camera_list:
                camera_info = self.pixel_source.camera_data[cam_id]
                intrinsics = camera_info.intrinsics.to(seed_pts.device)  # (3, 3)
                cam_to_worlds = camera_info.cam_to_worlds.to(seed_pts.device) # (F, 4, 4)
                road_masks = camera_info.road_masks.to(seed_pts.device)     # (F, H, W)
                ego_masks = camera_info.egocar_mask.to(seed_pts.device) # (F, H, W)

                num_frames = cam_to_worlds.shape[0]
                img_h, img_w = road_masks.shape[1], road_masks.shape[2]
                
                # --- 核心改动：逐帧处理 ---
                for frame_idx in range(num_frames):
                    # 1. 获取当前帧的变换矩阵和 road mask
                    cam_to_world = cam_to_worlds[frame_idx] # (4, 4)
                    world_to_cam = torch.linalg.inv(cam_to_world) # (4, 4)
                    current_road_mask = road_masks[frame_idx] # (H, W)
                    current_ego_mask = ego_masks # (H, W) # 为什么是(1024)? egocar_mask 对于同一个相机来说是固定不变的，它不随 frame_idx 变化

                    # 2. 将所有点云变换到当前帧的相机坐标系
                    #    pts_cam: (N, 3)
                    pts_cam = transform_points(seed_pts, world_to_cam)

                    # 3. 筛选出在相机前方的点
                    in_front_of_cam_mask = pts_cam[:, 2] > 0  # (N,)

                    fx, fy = intrinsics[frame_idx][0, 0], intrinsics[frame_idx][1, 1]
                    cx, cy = intrinsics[frame_idx][0, 2], intrinsics[frame_idx][1, 2]
        
                    u = (fx * pts_cam[:, 0] / pts_cam[:, 2] + cx).long()
                    v = (fy * pts_cam[:, 1] / pts_cam[:, 2] + cy).long()

                    # 4. 筛选在图像范围内的点（坐标在0w 0h之间）
                    h, w = img_h, img_w
                    in_image_bounds_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
                    
                    # 6. 合并当前帧的所有有效条件
                    final_valid_mask = in_front_of_cam_mask & in_image_bounds_mask # (N,) # 所有点里在相机前方且投影在相机平面上的

                    ### 对pts_cam_valid
                    # 7. 对满足条件的点，检查它们是否在路面 mask 上
                    if final_valid_mask.any():
                        # 获取满足条件的点的像素坐标
                        valid_u = u[final_valid_mask]
                        valid_v = v[final_valid_mask]
                        valid_pixels = torch.stack([valid_u, valid_v], dim=1)
                        
                        # 找到这些有效点在原始点云中的索引
                        point_indices = torch.where(final_valid_mask)[0]
                        
                        # 在road_mask的点去掉ego_mask部分
                        is_on_road_for_frame = current_road_mask[valid_pixels[:, 1], valid_pixels[:, 0]]
                        current_ego_mask = ~current_ego_mask
                        ego_mask_at_valid_pixels = current_ego_mask[valid_pixels[:, 1], valid_pixels[:, 0]]
                        is_on_road_for_frame_wo_ego = is_on_road_for_frame * ego_mask_at_valid_pixels
                            
                        # 更新全局 on_road_mask
                        # 找到在本帧中被识别为路面点的原始点云索引
                        road_point_indices = point_indices[is_on_road_for_frame_wo_ego]
                        on_road_mask[road_point_indices] = True
            ### 逻辑：对于每个相机每一帧，找到相机视线+路面mask+去除ego_mask下所有点（存在冗余：路面分割错误识别

            final_road_mask = on_road_mask
            # --- 新增逻辑：根据 Z 轴将路面点分为“真路面”和“高处环境” ---
            
            # 1. 提取视觉识别出的所有路面点
            visual_road_pts = seed_pts[final_road_mask]
            
            # 2. 判断 Z 轴高度：保留小于 0.1 的
            # 注意：这里假设 Z 轴是世界坐标系下的高度
            z_threshold = -1.3  # 可以根据需要调整阈值 可视化点云测试1.3消除栅栏
            is_low_road = visual_road_pts[:, 2] < z_threshold
            
            # 3. 构建最终的“纯路面”掩码 (用于返回路面点云)
            # 我们需要找到 visual_road_pts 中满足 z < 0.1 的点在原始 seed_pts 中的索引
            # 首先获取 visual_road_pts 在原始数组中的索引
            visual_road_indices = torch.where(final_road_mask)[0]
            
            # 筛选出低处的索引
            real_road_indices = visual_road_indices[is_low_road]
            
            # 高处路面点索引 (Z >= 0.1) -> 这里显式使用了该变量
            high_lying_indices = visual_road_indices[~is_low_road]

            # 4. 准备环境点云
            # 原始环境点索引 (不在视觉路面上的点)
            non_road_indices = torch.where(~on_road_mask)[0]

            # 【关键步骤】合并索引：环境点 = 原始环境点 + 高处路面误检点
            final_env_indices = torch.cat([non_road_indices, high_lying_indices])

            # --- 处理环境点云 (去除地下噪音) ---
            # 先提取坐标用于判断
            env_pts_candidate = seed_pts[final_env_indices]

            if not road_only:
                # 去除地下点 (Z < 0)
                z_filter_mask = env_pts_candidate[:, 2] > 0.0
                num_removed = (~z_filter_mask).sum().item()
                if num_removed > 0:
                    print(f"Removed {num_removed} underground noise points.")
                
                # 获取最终有效的环境点索引
                valid_env_indices = final_env_indices[z_filter_mask]
            else:
                # 如果 road_only=True，暂不额外过滤
                valid_env_indices = final_env_indices

            # 赋值环境点数据
            filtered_pts = seed_pts[valid_env_indices]
            filtered_colors = seed_colors[valid_env_indices] if seed_colors is not None else None
            filtered_time = seed_time[valid_env_indices] if seed_time is not None else None

            # --- 处理路面点云 (采样) ---
            filtered_road_pts = seed_pts[real_road_indices]
            filtered_road_colors = seed_colors[real_road_indices] if seed_colors is not None else None
            filtered_road_time = seed_time[real_road_indices] if seed_time is not None else None 

            # 对路面点云进行随机采样
            num_samples = 100000 # 随机采样值 

            if num_samples > filtered_road_pts.shape[0]:
                num_samples = filtered_road_pts.shape[0]
            sampled_idx = torch.randperm(filtered_road_pts.shape[0])[:num_samples]

            filtered_road_pts = filtered_road_pts[sampled_idx]
            filtered_road_colors = filtered_road_colors[sampled_idx] if filtered_road_colors is not None else None
            filtered_road_time = filtered_road_time[sampled_idx] if filtered_road_time is not None else None   

        elif TYPE == "LIDAR":
            filtered_pts, filtered_colors, filtered_time, filtered_road_pts, filtered_road_colors, filtered_road_time = self.filter_pts_in_road_lidar(seed_pts, seed_colors, road_only, seed_time)

        elif TYPE == "LIDAR_AND_IMAGE":
            # 结合 LIDAR 和 IMAGE 两种方法
            # 1. 先用 LIDAR PMF 算法提取地面点
            print("Using LIDAR_AND_IMAGE mode: combining PMF ground segmentation with image road masks")
            filtered_pts, filtered_colors, filtered_time, lidar_road_pts, lidar_road_colors, lidar_road_time = self.filter_pts_in_road_lidar(
                seed_pts, seed_colors, road_only, seed_time
            )
            
            # 2. 对 LIDAR 提取的地面点应用 IMAGE 的 road_mask 过滤
            print(f"Filtering {lidar_road_pts.shape[0]} LIDAR ground points with image road masks...")
            
            # 使用一个布尔张量来记录 LIDAR 地面点中哪些在 road_mask 中
            on_road_mask = torch.zeros_like(lidar_road_pts[:, 0], dtype=torch.bool)
            
            # 遍历所有有效相机
            for cam_id in self.pixel_source.camera_list:
                camera_info = self.pixel_source.camera_data[cam_id]
                intrinsics = camera_info.intrinsics.to(lidar_road_pts.device)
                cam_to_worlds = camera_info.cam_to_worlds.to(lidar_road_pts.device)
                road_masks = camera_info.road_masks.to(lidar_road_pts.device)
                ego_masks = camera_info.egocar_mask.to(lidar_road_pts.device)
                
                num_frames = cam_to_worlds.shape[0]
                img_h, img_w = road_masks.shape[1], road_masks.shape[2]
                
                # 逐帧处理
                for frame_idx in range(num_frames):
                    cam_to_world = cam_to_worlds[frame_idx]
                    world_to_cam = torch.linalg.inv(cam_to_world)
                    current_road_mask = road_masks[frame_idx]
                    current_ego_mask = ego_masks
                    
                    # 将 LIDAR 地面点变换到当前帧的相机坐标系
                    from utils.geometry import transform_points
                    pts_cam = transform_points(lidar_road_pts, world_to_cam)
                    
                    # 筛选出在相机前方的点
                    in_front_of_cam_mask = pts_cam[:, 2] > 0
                    
                    fx, fy = intrinsics[frame_idx][0, 0], intrinsics[frame_idx][1, 1]
                    cx, cy = intrinsics[frame_idx][0, 2], intrinsics[frame_idx][1, 2]
                    
                    u = (fx * pts_cam[:, 0] / pts_cam[:, 2] + cx).long()
                    v = (fy * pts_cam[:, 1] / pts_cam[:, 2] + cy).long()
                    
                    # 筛选在图像范围内的点
                    in_image_bounds_mask = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
                    
                    # 合并当前帧的所有有效条件
                    final_valid_mask = in_front_of_cam_mask & in_image_bounds_mask
                    
                    # 对满足条件的点，检查它们是否在路面 mask 上
                    if final_valid_mask.any():
                        valid_u = u[final_valid_mask]
                        valid_v = v[final_valid_mask]
                        valid_pixels = torch.stack([valid_u, valid_v], dim=1)
                        
                        point_indices = torch.where(final_valid_mask)[0]
                        
                        # 检查是否在 road_mask 上（去除 ego_mask）
                        is_on_road_for_frame = current_road_mask[valid_pixels[:, 1], valid_pixels[:, 0]]
                        current_ego_mask = ~current_ego_mask
                        ego_mask_at_valid_pixels = current_ego_mask[valid_pixels[:, 1], valid_pixels[:, 0]]
                        is_on_road_for_frame_wo_ego = is_on_road_for_frame * ego_mask_at_valid_pixels
                        
                        # 更新全局 on_road_mask
                        road_point_indices = point_indices[is_on_road_for_frame_wo_ego]
                        on_road_mask[road_point_indices] = True
            
            # 3. 筛选出同时满足 LIDAR 地面 + IMAGE road_mask 的点
            final_road_indices = torch.where(on_road_mask)[0]
            filtered_road_pts = lidar_road_pts[final_road_indices]
            filtered_road_colors = lidar_road_colors[final_road_indices] if lidar_road_colors is not None else None
            filtered_road_time = lidar_road_time[final_road_indices] if lidar_road_time is not None else None
            
            print(f"LIDAR ground points: {lidar_road_pts.shape[0]}, After road mask filtering: {filtered_road_pts.shape[0]}")
            
            # 采样到目标数量
            num_samples = 100000
            if num_samples > filtered_road_pts.shape[0]:
                num_samples = filtered_road_pts.shape[0]
            
            if num_samples > 0:
                sampled_idx = torch.randperm(filtered_road_pts.shape[0])[:num_samples]
                filtered_road_pts = filtered_road_pts[sampled_idx]
                filtered_road_colors = filtered_road_colors[sampled_idx] if filtered_road_colors is not None else None
                filtered_road_time = filtered_road_time[sampled_idx] if filtered_road_time is not None else None

        torch.cuda.empty_cache()

        return {"pts": filtered_pts, "colors": filtered_colors, "time": filtered_time}, {"pts": filtered_road_pts, "colors": filtered_road_colors, "time": filtered_road_time}

    def filter_pts_in_road_lidar(
        self,
        seed_pts: Tensor,
        seed_colors: Tensor = None,
        road_only: bool = False,
        seed_time: Tensor = None,
    ):
        """
        使用 Progressive Morphological Filter (PMF) 算法分离路面点云和环境点云
        
        
            seed_pts: 输入点云 (N, 3)
            seed_colors: 点云颜色 (N, 3)
            road_only: 是否只关注地面点
            seed_time: 点云时间戳 (N,)
        
        返回:
            env_dict: 环境点云字典 {"pts": ..., "colors": ..., "time": ...}
            road_dict: 路面点云字典 {"pts": ..., "colors": ..., "time": ...}
        """
        
        @torch.no_grad()
        def robust_elevation_init(points: Tensor, linear_indices: Tensor, num_grid: int, cell_size: float):
            """
            稳健的高程图初始化：处理地下射线噪点
            """
            device = points.device
            
            # --- 1. 基于全局分布的粗过滤 ---
            # 统计 Z 轴分布，剔除极端的底部离群点（假设噪点占极少数）
            # 这里的 0.05 分位数表示我们认为最底部的 5% 极大可能是噪点
            z_values = points[:, 2]
            # --- 修改后 (稳健采样版) ---
            if z_values.numel() > 1000000:
                # 随机采样 100w 个点来估算分位数
                indices = torch.randint(0, z_values.numel(), (1000000,), device=z_values.device)
                sample_z = z_values[indices]
                global_low_bound = torch.quantile(sample_z, 0.05)
            else:
                global_low_bound = torch.quantile(z_values, 0.05)
            
            # 初步掩码：只保留高于“极端底部”一定距离的点
            # 如果路面水平，这步能去掉大部分深层射线
            rough_mask = z_values > (global_low_bound - 0.2)
            
            valid_pts = points[rough_mask]
            valid_indices = linear_indices[rough_mask]

            # --- 2. 局部网格稳健估计 ---
            # 如果直接用 amin，依然会被剩下的漏网之鱼破坏。
            # 我们先计算每个网格的平均高度作为参考基准。
            
            # 计算每个网格的点数和高度总和
            counts = torch.zeros(num_grid, device=device)
            sums = torch.zeros(num_grid, device=device)
            
            ones = torch.ones_like(valid_indices, dtype=torch.float32)
            counts.scatter_add_(0, valid_indices, ones)
            sums.scatter_add_(0, valid_indices, valid_pts[:, 2])
            
            # 计算网格平均高度 (避开零除)
            counts += 1e-6
            grid_means = sums / counts
            del counts, sums
            
            # --- 3. 过滤掉网格内明显低于平均值的点 ---
            # 在同一个网格内，如果某个点比平均值还低很多（比如 > 1.0m），那它一定是残留的地下噪点
            point_grid_means = grid_means[valid_indices]
            local_refine_mask = (valid_pts[:, 2] > (point_grid_means - 0.8))
            del grid_means
            
            final_pts = valid_pts[local_refine_mask]
            final_indices = valid_indices[local_refine_mask]

            # --- 4. 生成最终高程图 ---
            elevation_map_flat = torch.full((num_grid,), float('inf'), device=device)
            elevation_map_flat.scatter_reduce_(0, final_indices, final_pts[:, 2], reduce="amin", include_self=False)
            
            # 填充无效网格
            scene_min_z = points[:, 2].min()
            mask_inf = elevation_map_flat == float('inf')
            elevation_map_flat = torch.where(mask_inf, scene_min_z, elevation_map_flat)
            
            return elevation_map_flat
        
        # ==================== Progressive Morphological Filter 算法====================
        @torch.no_grad()
        def progressive_morphological_filter_optimized(
            points: Tensor,
            max_window_size: int = 20,
            slope: float = 0.05,        # 建议设小，针对平坦路面
            initial_distance: float = 0.15,
            max_distance: float = 2.5,
            cell_size: float = 0.5,     # 建议 0.5，增加稳健性
            exponential: bool = True
        ):
            device = points.device
            N = points.shape[0]

            # --- 1. 预处理：简单的统计去噪 (SOR) ---
            # 目的：剔除悬浮的孤立杂点，防止它们污染初始高度图
            # 如果点云极其密集，可以跳过此步以节省时间
            if N > 0:
                pass 

            # --- 2. 向量化计算网格索引 ---
            coords_min = points[:, :2].min(dim=0)[0]
            coords_max = points[:, :2].max(dim=0)[0]
            
            grid_width = int(torch.ceil((coords_max[0] - coords_min[0]) / cell_size).item()) + 1
            grid_height = int(torch.ceil((coords_max[1] - coords_min[1]) / cell_size).item()) + 1
            
            grid_x = ((points[:, 0] - coords_min[0]) / cell_size).long().clamp(0, grid_width - 1)
            grid_y = ((points[:, 1] - coords_min[1]) / cell_size).long().clamp(0, grid_height - 1)
            linear_indices = grid_y * grid_width + grid_x

            # --- 3. 初始高程图 ---
            # elevation_map_flat = torch.full((grid_height * grid_width,), float('inf'), device=device)
            # elevation_map_flat.scatter_reduce_(0, linear_indices, points[:, 2], reduce="amin", include_self=False)
            
            # 调用增强的初始化函数
            elevation_map_flat = robust_elevation_init(
                points=seed_pts, 
                linear_indices=linear_indices, 
                num_grid=grid_height * grid_width,
                cell_size=cell_size
            )

            scene_min_z = points[:, 2].min()
            mask_inf = elevation_map_flat == float('inf')
            elevation_map_flat = torch.where(mask_inf, scene_min_z, elevation_map_flat)

            # 2. 【关键优化】将高程图移至 CPU 进行形态学操作
            working_map_cpu = elevation_map_flat.view(grid_height, grid_width).cpu()
            del elevation_map_flat  # 释放 GPU 显存
            # --- 4. 优化后的 PMF 迭代 ---
            window_sizes = []
            k = 0
            while True:
                window_size = 2 * (2**k if exponential else k) + 1
                if window_size > max_window_size: break
                window_sizes.append(window_size)
                k += 1

            pbar = tqdm(window_sizes, desc="Progressive Morphological Filter")
            for window_size in pbar:
                pbar.set_description(f"Progressive Morphological Filter(window={window_size})")
                # 计算当前窗口下的高度阈值
                dhp = slope * (window_size - 1) * cell_size + initial_distance
                dhp = min(dhp, max_distance)
                
                # 执行开运算 (Opening)
                padding = window_size // 2

                # 在 CPU 上运行 max_pool2d
                map_4d = working_map_cpu.unsqueeze(0).unsqueeze(0)
                # torch.nn.functional.max_pool2d 在 CPU 下内存管理更保守
                eroded = -torch.nn.functional.max_pool2d(-map_4d, window_size, stride=1, padding=padding)
                dilated = torch.nn.functional.max_pool2d(eroded, window_size, stride=1, padding=padding)
                dilated = dilated.squeeze()
                
                # 截断和更新同样在 CPU 完成
                if dilated.shape != working_map_cpu.shape:
                    dilated = dilated[:grid_height, :grid_width]
                
                mask_non_ground = (working_map_cpu - dilated) > dhp
                working_map_cpu = torch.where(mask_non_ground, dilated, working_map_cpu)

            # 3. 最后将结果移回 GPU 进行 Mask 判定
            working_map = working_map_cpu.to(device)
            # --- 5. 最终判定 ---
            # 使用最后一轮迭代产生的平滑地表作为基准
            point_ground_heights = working_map[grid_y, grid_x]
            
            # 地面点判定：点的高度与估计地表高度差在阈值内
            height_diff = points[:, 2] - point_ground_heights
            # 注意：只保留在上方一定范围内的点，排除掉由于遮挡产生的虚假地下点
            ground_mask = (height_diff < initial_distance) & (height_diff > -initial_distance)
            
            return ground_mask

        # ==================== 主处理流程 ====================
        print(f"Starting PMF ground segmentation on {seed_pts.shape[0]} points...")
        
        ground_mask = progressive_morphological_filter_optimized(
            seed_pts,
            max_window_size=20,    # 对应 PCL 的 setMaxWindowSize
            slope=0.05,             # 对应 PCL 的 setSlope
            initial_distance=0.15,  # 对应 PCL 的 setInitialDistance
            max_distance=2.5,      # 对应 PCL 的 setMaxDistance
            cell_size=0.5         # 网格分辨率
        )
        num_ground_points = ground_mask.sum().item()
        num_object_points = (~ground_mask).sum().item()
        print(f"PMF segmentation complete: {num_ground_points} ground points, {num_object_points} object points")
        
        # 提取路面点和环境点
        road_indices = torch.where(ground_mask)[0]
        env_indices = torch.where(~ground_mask)[0]
        
        # 处理环境点云（去除地下点）
        if not road_only and env_indices.numel() > 0:
            env_pts_candidate = seed_pts[env_indices]
            # 去除地下点 (Z < 0)
            z_filter_mask = env_pts_candidate[:, 2] > 0.0
            num_removed = (~z_filter_mask).sum().item()
            if num_removed > 0:
                print(f"Removed {num_removed} underground noise points.")
            
            # 获取最终有效的环境点索引
            valid_env_indices = env_indices[z_filter_mask]
        else:
            valid_env_indices = env_indices
        
        # 提取环境点数据
        filtered_pts = seed_pts[valid_env_indices]
        filtered_colors = seed_colors[valid_env_indices] if seed_colors is not None else None
        filtered_time = seed_time[valid_env_indices] if seed_time is not None else None
        
        # 提取路面点数据
        filtered_road_pts = seed_pts[road_indices]
        filtered_road_colors = seed_colors[road_indices] if seed_colors is not None else None
        filtered_road_time = seed_time[road_indices] if seed_time is not None else None

        # 对路面点云进行随机采样（与 filter_pts_in_road 保持一致）
        num_samples = 100000
        if num_samples > filtered_road_pts.shape[0]:
            num_samples = filtered_road_pts.shape[0]
        
        if num_samples > 0:
            sampled_idx = torch.randperm(filtered_road_pts.shape[0], device=seed_pts.device)[:num_samples]
            filtered_road_pts = filtered_road_pts[sampled_idx]
            filtered_road_colors = filtered_road_colors[sampled_idx] if filtered_road_colors is not None else None
            filtered_road_time = filtered_road_time[sampled_idx] if filtered_road_time is not None else None
        
        # 返回环境点和路面点
        return filtered_pts, filtered_colors, filtered_time, filtered_road_pts, filtered_road_colors, filtered_road_time
        
    def check_pts_visibility(self, pts_xyz):
        # filter out the lidar points that are not visible from the camera
        pts_xyz = pts_xyz.to(self.device)
        valid_mask = torch.zeros_like(pts_xyz[:, 0], dtype=torch.bool)
        # project lidar points to the image plane
        for cam in self.pixel_source.camera_data.values():
            for frame_idx in range(len(cam)):
                intrinsic_4x4 = torch.nn.functional.pad(
                    cam.intrinsics[frame_idx], (0, 1, 0, 1)
                )
                intrinsic_4x4[3, 3] = 1.0
                lidar2img = intrinsic_4x4 @ cam.cam_to_worlds[frame_idx].inverse()
                projected_points = (
                    lidar2img[:3, :3] @ pts_xyz.T + lidar2img[:3, 3:4]
                ).T
                depth = projected_points[:, 2]
                cam_points = projected_points[:, :2] / (depth.unsqueeze(-1) + 1e-6)
                current_valid_mask = (
                    (cam_points[:, 0] >= 0)
                    & (cam_points[:, 0] < cam.WIDTH)
                    & (cam_points[:, 1] >= 0)
                    & (cam_points[:, 1] < cam.HEIGHT)
                    & (depth > 0)
                )
                valid_mask = valid_mask | current_valid_mask
        return valid_mask

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
        logger.info(
            f"Train timesteps: \n{np.arange(self.start_timestep, self.end_timestep)[train_timesteps]}"
        )
        logger.info(
            f"Test timesteps: \n{np.arange(self.start_timestep, self.end_timestep)[test_timesteps]}"
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
        logger.info(f"Number of train indices: {len(train_indices)}")
        logger.info(f"Train indices: {train_indices}")
        logger.info(f"Number of test indices: {len(test_indices)}")
        logger.info(f"Test indices: {test_indices}")

        # Again, training and testing indices are indices into the full dataset
        # train_indices are img indices, so the length is num_cams * num_train_timesteps
        # but train_timesteps are timesteps, so the length is num_train_timesteps (len(unique_train_timestamps))
        return train_timesteps, test_timesteps, train_indices, test_indices

    def project_lidar_pts_on_images(self, delete_out_of_view_points: bool = True):
        """
        Project the lidar points on the images and attribute the color of the nearest pixel to the lidar point.

        Args:
            delete_out_of_view_points: bool
                If True, the lidar points that are not visible from the camera will be removed.
        """
        static_lidar_points = []
        for idx, cam in enumerate(self.pixel_source.camera_data.values()):
            lidar_depth_maps = []

            for frame_idx in tqdm(
                range(len(cam)),
                desc=f"Processing camera {cam.cam_name}",
                dynamic_ncols=True
            ):
                normed_time = self.pixel_source.normalized_time[frame_idx]

                # get lidar depth on image plane
                closest_lidar_idx = self.lidar_source.find_closest_timestep(normed_time)
                lidar_infos = self.lidar_source.get_lidar_rays(closest_lidar_idx)
                lidar_points_world = lidar_infos["lidar_origins"] + lidar_infos["lidar_viewdirs"] * lidar_infos["lidar_ranges"]
                lidar_mask = lidar_infos["lidar_mask"]

                if cam.undistort:
                    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                        cam.intrinsics[frame_idx].cpu().numpy(),
                        cam.distortions[frame_idx].cpu().numpy(),
                        (cam.WIDTH, cam.HEIGHT),
                        alpha=1,
                    )
                    intrinsic = torch.from_numpy(new_camera_matrix).to(self.device)
                else:
                    intrinsic = cam.intrinsics[frame_idx]
                                
                cam_points, depth, valid_mask = project_points_to_image(
                    xyz=lidar_points_world,
                    K=intrinsic,
                    RT=cam.cam_to_worlds[frame_idx].inverse(),
                    H=cam.HEIGHT,
                    W=cam.WIDTH
                )
                valid_cam_points = cam_points[valid_mask]
                valid_depth = depth[valid_mask]
                
                # used to filter out the lidar points that are visible from the camera
                global_indices = torch.arange(self.lidar_source.num_points, device=self.device)[lidar_mask][valid_mask]
                self.lidar_source.visible_masks[global_indices] = True

                # attribute the color of the nearest pixel to the lidar point
                points_color = cam.images[frame_idx][valid_cam_points[:, 1].long(), valid_cam_points[:, 0].long()]
                self.lidar_source.colors[global_indices] = points_color
                

                if self.depth_mode == DepthMode.SINGLE_FRAME_RAW:
                    final_cam_points = valid_cam_points
                    final_depth = valid_depth

                    # 保存单帧深度图
                    depth_map = torch.zeros(cam.HEIGHT, cam.WIDTH, device=self.device)
                    depth_map[final_cam_points[:, 1].long(), final_cam_points[:, 0].long()] = final_depth
                    lidar_depth_maps.append(depth_map)
                    
                elif self.depth_mode == DepthMode.SINGLE_FRAME_STATIC:
                    static_mask = self._get_static_mask_for_frame(lidar_points_world, frame_idx)
                    # logger.info(f"Frame {frame_idx}: Removed {(~static_mask).sum().item()} points inside 3D bounding boxes.")
                    
                    final_cam_points = cam_points[static_mask & valid_mask]
                    final_depth = depth[static_mask & valid_mask]

                    # 保存单帧深度图
                    depth_map = torch.zeros(cam.HEIGHT, cam.WIDTH, device=self.device)
                    depth_map[final_cam_points[:, 1].long(), final_cam_points[:, 0].long()] = final_depth
                    lidar_depth_maps.append(depth_map)
                
                elif self.depth_mode == DepthMode.MULTI_FRAME_STATIC:
                    static_mask = self._get_static_mask_for_frame(lidar_points_world, frame_idx)
                    # logger.info(f"Frame {frame_idx}: Removed {(~static_mask).sum().item()} points inside 3D bounding boxes.")
                 
                    # 暂存静态点，待所有帧处理完后聚合生成稠密深度图
                    static_points_world = lidar_points_world[static_mask]
                    if idx == 0 and len(static_points_world) > 0:
                        static_lidar_points.append(static_points_world)
                
                else:
                    raise NotImplementedError(f"Depth mode {self.depth_mode} not implemented")
                
            # 单帧模式下直接保存深度图
            if self.depth_mode in [DepthMode.SINGLE_FRAME_RAW, DepthMode.SINGLE_FRAME_STATIC]:
                logger.info(f"Loading sparse depth maps for camera {cam.cam_name}")
                cam.load_depth(torch.stack(lidar_depth_maps, dim=0).float())
        
        # 聚合生成稠密深度
        if self.depth_mode == DepthMode.MULTI_FRAME_STATIC:
            aggregated_world_points = torch.cat(static_lidar_points, dim=0)
            logger.info(f"Aggregated {len(aggregated_world_points)} background points for dense depth")

            for cam in self.pixel_source.camera_data.values():
                dense_depth_maps = []
                for frame_idx in range(len(cam)):
                    if cam.undistort:
                        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                            cam.intrinsics[frame_idx].cpu().numpy(),
                            cam.distortions[frame_idx].cpu().numpy(),
                            (cam.WIDTH, cam.HEIGHT),
                            alpha=1,
                        )
                        intrinsic = torch.from_numpy(new_camera_matrix).to(self.device)
                    else:
                        intrinsic = cam.intrinsics[frame_idx]

                    aggregated_cam_points, aggregated_depth, aggregated_valid_mask = project_points_to_image(
                        xyz=aggregated_world_points,
                        K=intrinsic,
                        RT=cam.cam_to_worlds[frame_idx].inverse(),
                        H=cam.HEIGHT,
                        W=cam.WIDTH
                    )
                    valid_aggregated_cam_points = aggregated_cam_points[aggregated_valid_mask]
                    valid_aggregated_depth = aggregated_depth[aggregated_valid_mask]
                    
                    dense_map = torch.zeros(cam.HEIGHT, cam.WIDTH, device=self.device)
                    if len(aggregated_cam_points) > 0:
                        dense_map[valid_aggregated_cam_points[:, 1].long(), valid_aggregated_cam_points[:, 0].long()] = valid_aggregated_depth
                    dense_depth_maps.append(dense_map)
                
                cam.load_depth(torch.stack(dense_depth_maps, dim=0).float())
                logger.info(f"Loading aggregated depth maps for camera {cam.cam_name}")

        if delete_out_of_view_points:
            self.lidar_source.delete_invisible_pts()

    def _get_static_mask_for_frame(self, lidar_points_world, frame_idx):
        """针对单帧计算静态点掩码"""
        static_mask = torch.ones(lidar_points_world.shape[0], dtype=torch.bool, device=self.device)
        
        visible_instances = torch.where(self.pixel_source.per_frame_instance_mask[frame_idx])[0]
        if len(visible_instances) == 0:
            return static_mask
        
        instances_pose_world = self.pixel_source.instances_pose[frame_idx][visible_instances]
        instances_size = self.pixel_source.instances_size[visible_instances]
        
        points_homo = torch.cat([lidar_points_world, torch.ones(lidar_points_world.shape[0], 1, device=self.device)], dim=-1)
        
        for pose, size in zip(instances_pose_world, instances_size):
            points_local = (torch.inverse(pose) @ points_homo.T).T[:, :3]
            half_size = size / 2
            in_bbox = (
                (points_local[:, 0].abs() <= half_size[0]) &
                (points_local[:, 1].abs() <= half_size[1]) &
                (points_local[:, 2].abs() <= half_size[2])
            )
            static_mask &= ~in_bbox

        return static_mask