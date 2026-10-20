# renderer_manager.py
import os
import threading
import numpy as np
from omegaconf import OmegaConf
from cam.cam_inference_online import Renderer
from infer_utils import *
from lidar.lidar_inference_online import Renderer as LidarRenderer


class CamRendererManager:
    def __init__(
        self,
        resume_from_group0,
        resume_from_group1,
        config_file_group0,
        config_file_group1,
        source_path,
        output_dir
    ):
        print("[Init] Initializing renderer...")

        group0_cam_ids, group0_downscales = self.load_group_config(
            config_file=config_file_group0,
            expected_group_id=0
        )
        group1_cam_ids, group1_downscales = self.load_group_config(
            config_file=config_file_group1,
            expected_group_id=1
        )

        duplicate_cam_ids = set(group0_cam_ids) & set(group1_cam_ids)
        if len(duplicate_cam_ids) > 0:
            raise ValueError(
                f"Camera ids exist in both groups: {sorted(duplicate_cam_ids)}"
            )

        print(
            f"[Init] Loading group0 renderer, "
            f"checkpoint: {resume_from_group0}, cam_ids: {group0_cam_ids}"
        )
        self.renderer_group0 = Renderer(
            resume_from=resume_from_group0,
            cam_ids=group0_cam_ids,
            downscales=group0_downscales,
            output_dir=output_dir,
        )

        print(
            f"[Init] Loading group1 renderer, "
            f"checkpoint: {resume_from_group1}, cam_ids: {group1_cam_ids}"
        )
        self.renderer_group1 = Renderer(
            resume_from=resume_from_group1,
            cam_ids=group1_cam_ids,
            downscales=group1_downscales,
            output_dir=output_dir,
        )

        self.renderer_dict = {
            0: self.renderer_group0,
            1: self.renderer_group1
        }
        self.renderer_lock_dict = {
            0: threading.Lock(),
            1: threading.Lock()
        }

        self.camera_group_dict = {}
        for cam_id in group0_cam_ids:
            self.camera_group_dict[cam_id] = 0
        for cam_id in group1_cam_ids:
            self.camera_group_dict[cam_id] = 1

        print("[Init] Camera group mapping:", self.camera_group_dict)

        # Preload the matrices required for trajectory coordinate transformation
        lidar2ego_dict = extract_lidar_extrinsics(
            json_path=os.path.join(source_path, "data_frame_car_info.json")
        )
        self.lidar2ego = pose_to_transform_matrix(**lidar2ego_dict)

        lidar_pose_txt = find_min_frame_txt(os.path.join(source_path, "lidar_pose"))
        lidar_to_world_start = load_transform_matrix(txt_path=lidar_pose_txt)
        self.cam2lidar = load_transform_matrix(txt_path=os.path.join(source_path, "extrinsics", "0.txt"))

        self.inv_start = np.linalg.inv(lidar_to_world_start)
        self.start_timestamp = extract_start_timestamp(os.path.join(source_path, "timestamps.json"))
        self.frame_interval = 0.1  # 100ms per frame, adjust if needed

        print("[Init] Renderer initialization done.")

    def load_group_config(self, config_file, expected_group_id):
        if config_file is None or not os.path.isfile(config_file):
            raise FileNotFoundError(
                f"group{expected_group_id} config file not found: {config_file}"
            )

        cfg = OmegaConf.load(config_file)

        if "view_parallel" not in cfg:
            raise ValueError(
                f"view_parallel config not found in: {config_file}"
            )

        group_id = cfg.view_parallel.get("group_id", None)
        if group_id is not None and int(group_id) != expected_group_id:
            raise ValueError(
                f"Config group_id mismatch, expected group{expected_group_id}, "
                f"but got group{group_id}: {config_file}"
            )

        view_ids = cfg.view_parallel.get("view_ids", None)
        if view_ids is None:
            raise ValueError(
                f"view_parallel.view_ids not found in: {config_file}"
            )

        cam_ids = [int(cam_id) for cam_id in view_ids]

        pixel_source_cfg = cfg.data.pixel_source
        all_cam_ids = [
            int(cam_id) for cam_id in pixel_source_cfg.get("cameras", cam_ids)
        ]
        all_downscales = pixel_source_cfg.get(
            "downscale_when_loading",
            [1] * len(all_cam_ids)
        )
        all_downscales = list(all_downscales)

        if len(all_cam_ids) != len(all_downscales):
            raise ValueError(
                f"Camera ids and downscales length mismatch in: {config_file}"
            )

        downscale_dict = {
            cam_id: downscale
            for cam_id, downscale in zip(all_cam_ids, all_downscales)
        }

        missing_cam_ids = [
            cam_id for cam_id in cam_ids
            if cam_id not in downscale_dict
        ]
        if len(missing_cam_ids) > 0:
            raise ValueError(
                f"Camera ids {missing_cam_ids} are not found in "
                f"data.pixel_source.cameras: {config_file}"
            )

        downscales = [
            downscale_dict[cam_id]
            for cam_id in cam_ids
        ]

        print(
            f"[Init] Loaded group{expected_group_id} config, "
            f"cam_ids: {cam_ids}, downscales: {downscales}"
        )

        return cam_ids, downscales

    def render_from_pose(self, pose_msg):
        """
        pose_msg: 来自 protobuf 的 MainCarInfo，包括 x,y,z,yaw,pitch,roll
        """

        raw = {
            "x": float(pose_msg.x),
            "y": float(pose_msg.y),
            "z": float(pose_msg.z),
            "yaw": float(pose_msg.yaw),
            "roll": float(pose_msg.roll),
            "pitch": float(pose_msg.pitch),
        }

        pose = pose_to_transform_matrix(**raw)
        ego2world = self.inv_start @ pose

        print("[Render] Pose:\n", pose)
        if 'RESIZE' in pose_msg.sensor_id:
            sensor_name = pose_msg.sensor_id.replace('_RESIZE', '')
            resize_flag = True
        else:
            sensor_name = pose_msg.sensor_id
            resize_flag = False

        cam_id = QCRAFT_CAMERA_DICT[sensor_name]
        print("cam_id: ", cam_id)

        if cam_id not in self.camera_group_dict:
            raise ValueError(
                f"Camera id {cam_id} is not found in any view group"
            )

        group_id = self.camera_group_dict[cam_id]
        renderer = self.renderer_dict[group_id]
        renderer_lock = self.renderer_lock_dict[group_id]

        print("group_id: ", group_id)

        frame_id = round(max(0, (float(pose_msg.timestamp) - self.start_timestamp)) / self.frame_interval)
        print("pose_msg.timestamp: ", pose_msg.timestamp)

        with renderer_lock:
            output_paths = renderer.render_single_frame(
                ego2world,
                cam_id,
                frame_id,
                resize_flag,
            )

        return output_paths

class LidarRendererManager:
    def __init__(self, lidar_checkpoint_path, source_path, output_dir):
        print("[Init] Initializing LiDAR renderer...")

        self.renderer = LidarRenderer(
            lidar_checkpoint_path=lidar_checkpoint_path,
            source_path=source_path,
            output_dir=output_dir
        )

        # Preload the matrices required for trajectory coordinate transformation
        lidar2ego_dict = extract_lidar_extrinsics(
            json_path=os.path.join(source_path, "data_frame_car_info.json")
        )
        self.lidar2ego = pose_to_transform_matrix(**lidar2ego_dict)

        lidar_pose_txt = find_min_frame_txt(os.path.join(source_path, "lidar_pose"))
        lidar_to_world_start = load_transform_matrix(txt_path=lidar_pose_txt)
        
        self.inv_start = np.linalg.inv(lidar_to_world_start)
        self.start_timestamp = extract_start_timestamp(os.path.join(source_path, "timestamps.json"))
        self.frame_interval = 0.1  # 100ms per frame, adjust if needed

        print("[Init] Renderer initialization done.")

    def render_from_pose(self, pose_msg):
        
        raw = {
            "x": pose_msg.x,
            "y": pose_msg.y,
            "z": pose_msg.z,
            "yaw": pose_msg.yaw,
            "roll": pose_msg.roll,
            "pitch": pose_msg.pitch
        }

        lidar_id = QCRAFT_LIDAR_DICT[pose_msg.sensor_id]

        pose = pose_to_transform_matrix(**raw)
        # novel_pose是ego坐标, 计算新视角下的lidar_pose(世界坐标系下)
        lidar2world = pose @ self.lidar2ego
        # 世界坐标系下的lidar_pose 转换到 第一帧lidar_pose定义的world坐标系下
        rel_pose = self.inv_start @ lidar2world

        frame_id = round(max(0, (float(pose_msg.timestamp) - self.start_timestamp)) / self.frame_interval)
        print("pose_msg.timestamp: ", pose_msg.timestamp)

        output_paths = self.renderer.render_single_frame(
            lidar2world,
            lidar_id,
            frame_idx=frame_id,
        )
        # output_paths = self.renderer.render_single_frame(lidar2world, lidar_id)
        print("[Render] Output:", output_paths)

        return output_paths

# initialized only once
cam_renderer_manager: CamRendererManager = None
lidar_renderer_manager: LidarRendererManager = None

def init_renderer_cam(
    resume_from_group0,
    resume_from_group1,
    config_file_group0,
    config_file_group1,
    source_path,
    output_dir
):
    global cam_renderer_manager
    if cam_renderer_manager is None:
        cam_renderer_manager = CamRendererManager(
            resume_from_group0=resume_from_group0,
            resume_from_group1=resume_from_group1,
            config_file_group0=config_file_group0,
            config_file_group1=config_file_group1,
            source_path=source_path,
            output_dir=output_dir
        )
    return cam_renderer_manager

def init_renderer_lidar(lidar_checkpoint_path, source_path, output_dir):
    global lidar_renderer_manager
    if lidar_renderer_manager is None:
        lidar_renderer_manager = LidarRendererManager(lidar_checkpoint_path, source_path, output_dir)
    return lidar_renderer_manager
