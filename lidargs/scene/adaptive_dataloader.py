"""
自适应 Dataloader - 自动从预处理的 sensor_info.json 读取 lidar 内参
支持不同的 lidar 类型（LDR_FRONT, LDR_CENTER等）
"""

import os
import numpy as np
import torch
import json
from collections import defaultdict
from utils.lidar_utils import lidar_to_pano_with_intensities


class AdaptiveDataloader:
    """
    自动读取 sensor_info.json 的 Dataloader
    支持任意 lidar 类型，无需硬编码参数
    """
    MIN_OBJ_POINT_NUM = 20

    @staticmethod
    def instance_id_to_model_id(instance_id):
        return int(instance_id) + 1

    @staticmethod
    def model_id_to_instance_id(model_id):
        model_id = int(model_id)
        if model_id <= 0:
            raise ValueError(f"background model_id={model_id} has no instance_id")
        return model_id - 1

    def __init__(self, args, train=True, train_frame_times=None, dtype=np.float32):
        self.train = train
        self.root_path = args.source_path
        self.case = args.caseid
        self.block_id = args.block_id
        self.preprocess_path = os.path.join(args.source_path, "lidar_preprocessed_data")
        
        if not os.path.exists(self.preprocess_path):
            os.makedirs(self.preprocess_path)
        # ========== 读取 sensor_info.json ==========
        self._load_sensor_info()
        
        # ========== 设置雷达参数 ==========
        self._setup_lidar_params()
        
        # ========== 其他初始化 ==========
        self.max_depth = args.max_depth
        print(f"max_depth: {self.max_depth}")
        
        self.aabb_min = np.ones(3, dtype=np.float32) * (10000000)
        self.aabb_max = np.ones(3, dtype=np.float32) * (-10000000)
        
        self.sensor2baselidar = dict()
        self.pcds = []
        self.pcds_label = []
        self.l2ws = []
        self.timestep_2_frameid = dict()
        self.frameid_2_timestep = []
        self.baselidar2world = []
        self.instance_id_to_raw_id = dict()
        self.raw_id_to_instance_id = dict()
        
        self.train_frame_times = train_frame_times
        self.max_frame_num = len(train_frame_times)
        
        # ========== 加载帧数据（需要先初始化lidar_to_world_start和dynamic_obj_info） ==========
        self.frames_data = self.load_frames_data(train_frame_times)
        print(f"[ Info ] Loaded {len(self.frames_data)} frames")
        
        # ========== 加载点云数据 ==========
        self.pcds, self.l2ws = self.load_pcds(
            frames=self.frames_data,
            train_frame_times=self.train_frame_times,
            frame_num=self.max_frame_num,
        )
        
        self.instance_id_list = [int(key) for key in self.dynamic_obj_info.keys()]
        
        # ========== 加载动态物体点云 ==========
        self.obj_frames_id = dict()
        self.obj_pcd = dict()
        self.obj_o2l = dict()
        
        if self.instance_id_list is not None:
            valid_obj_instance_ids = []
            for obj_instance_id in self.instance_id_list:
                success, obj_pcd = self.load_dynamic_pcd(str(obj_instance_id))
                if not success:
                    continue
                self.obj_pcd[str(obj_instance_id)] = obj_pcd
                valid_obj_instance_ids.append(obj_instance_id)
            self.instance_id_list = valid_obj_instance_ids
        self.model_id_list = [self.instance_id_to_model_id(instance_id) for instance_id in self.instance_id_list]
        
        # ========== 加载静态点云 ==========
        if train:
            self.static_pcd = self.load_static_pcd()
        
        # ========== 生成 range view ==========
        self.range_views, self.masks = self.load_rangeview(self.H_lidar, self.W_lidar)
        self.novel_poses_setting = None

    def _load_sensor_info(self):
        sensor_info_path = os.path.join(self.root_path, "sensor_info.json")
        
        if not os.path.exists(sensor_info_path):
            raise FileNotFoundError(
                f"sensor_info.json not found at {sensor_info_path}\n"
                "Please run preprocessing first to generate sensor_info.json"
            )
        
        with open(sensor_info_path, "r") as f:
            sensor_info = json.load(f)
        
        # 读取主雷达名称
        self.main_lidar_name = sensor_info.get("main_lidar_name", "LDR_FRONT")
        print(f"[ Info ] Main LiDAR: {self.main_lidar_name}")
        
        # 读取 lidar 内参（只读 elevations）
        lidar_inherent = sensor_info.get("lidar_inherent", {})
        
        elevations = lidar_inherent.get("elevations")
        if elevations is None:
            raise ValueError("elevations not found in sensor_info.json")
        
        self.elevations = np.array(elevations, dtype=np.float32)
        self.num_beams = len(self.elevations)
        
        # 转换为弧度并排序（从小到大）
        self.beam_inclinations = np.sort(self.elevations) * np.pi / 180.0
        
        print(f"[ Info ] Loaded {self.num_beams} beam inclinations from sensor_info")
        print(f"[ Info ] Elevation range: {self.elevations.min():.2f}° to {self.elevations.max():.2f}°")
        

    def _setup_lidar_params(self):
        # FOV 写死：向上 +15°，向下 -25°
        self.FOV_UP   =  15.0 * np.pi / 180.0
        self.FOV_DOWN =  25.0 * np.pi / 180.0
        self.FOV_VERTICAL   = self.FOV_UP + self.FOV_DOWN
        self.FOV_HORIZONTAL = 2 * np.pi  # 360°

        # 分辨率写死
        self.H_lidar = 128
        self.W_lidar = 1800

        print(f"[ Info ] LiDAR FOV (fixed): Vertical {self.FOV_VERTICAL*180/np.pi:.1f}° "
              f"(↑{self.FOV_UP*180/np.pi:.1f}° ↓{self.FOV_DOWN*180/np.pi:.1f}°), "
              f"Horizontal {self.FOV_HORIZONTAL*180/np.pi:.0f}°")
        print(f"[ Info ] Range view resolution (fixed): {self.H_lidar}x{self.W_lidar}")

    def load_frames_data(self, train_frame_times):
        """加载指定训练帧的点云数据和位姿信息"""
        lidar_filefolder = os.path.join(self.root_path, "lidar", "bin")
        bin_files = [f for f in os.listdir(lidar_filefolder) if f.endswith(".bin")]
        bin_files = sorted(bin_files, key=lambda x: int(x.split(".")[0]))
        
        lidarpose_filefolder = os.path.join(self.root_path, "lidar_pose")
        
        self.start_frame = train_frame_times[0]
        frame_cnt = train_frame_times[-1] - train_frame_times[0] + 1
        
        # 加载起始帧的lidar2world变换
        self.lidar_to_world_start = np.loadtxt(
            os.path.join(lidarpose_filefolder, str(self.start_frame).zfill(6) + ".txt")
        )
        
        # 加载动态物体信息（在加载帧数据之前）
        self.item_id_2_instance_id, self.dynamic_obj_info = self.load_dynamic_obj_info(
            train_frame_times
        )
        self.dynamic_obj_list = self.load_dynamic_instance_id_list(
            train_frame_times, self.item_id_2_instance_id
        )
        
        frames_data = []
        dynamic_pcd_file_folder = os.path.join(self.preprocess_path, "dynamic_pcd")
        static_pcd_file_folder = os.path.join(self.preprocess_path, "static_pcd")
        
        if not os.path.exists(static_pcd_file_folder):
            os.makedirs(static_pcd_file_folder)
        if not os.path.exists(dynamic_pcd_file_folder):
            os.makedirs(dynamic_pcd_file_folder)
        
        for i in train_frame_times:
            single_frame_data = {}
            single_frame_data["frame_id"] = i
            # 与 Chery_Dataloader 对齐，同时保留 log_time_stamp 字段
            single_frame_data["log_time_stamp"] = i
            
            # 读取 lidar pose
            lidar_pose_i_path = os.path.join(
                lidarpose_filefolder, str(i).zfill(6) + ".txt"
            )
            lidar_to_world_current = np.loadtxt(lidar_pose_i_path)
            lidar_to_world = (
                np.linalg.inv(self.lidar_to_world_start) @ lidar_to_world_current
            )
            single_frame_data["lidar2world"] = lidar_to_world
            
            # 读取点云数据：x y z intensity lidar_id (5列)
            lidar_info = np.fromfile(
                os.path.join(self.root_path, "lidar", "bin", str(i).zfill(6) + ".bin"),
                dtype=np.float32,
            ).reshape(-1, 5)
            
            # 过滤掉 lidar_id 不为0的点（只保留主雷达）
            lidar_info = lidar_info[lidar_info[:, 4] == 0]
            # 自动检测 intensity 量程：100clips 数据为 [0,255]，chery360 预处理数据为 [0,1]
            if len(lidar_info) > 0 and lidar_info[:, 3].max() > 1.5:
                lidar_info[:, 3] = lidar_info[:, 3] / 255.0
            # 强度在0-1之间
            single_frame_data["raw_pcd"] = lidar_info[:, :4]  # x,y,z,intensity
            single_frame_data["lidar_id"] = lidar_info[:, 4]
            
            # 提取动态和静态点云
            dynamic_pcd = []
            static_pcd = []
            
            dynamic_pcd_frame_folder = os.path.join(
                dynamic_pcd_file_folder, str(i).zfill(3)
            )
            if not os.path.exists(dynamic_pcd_frame_folder):
                os.makedirs(dynamic_pcd_frame_folder)
            
            dynamic_instance_id_list = self.dynamic_obj_list[i]
            pointcloud_xyz = single_frame_data["raw_pcd"][:, :4]
            dynamic_obj_infos = []
            
            for instance_id in dynamic_instance_id_list:
                obj_info = self.dynamic_obj_info[instance_id]
                frame_indices = obj_info["frames"]
                if i not in frame_indices:
                    continue

                dynamic_pcd_path = os.path.join(
                    dynamic_pcd_frame_folder,
                    f"{str(i).zfill(3)}_obj{str(instance_id).zfill(3)}.npy",
                )
                idx = frame_indices.index(i)
                center_world = obj_info["poses"][idx][:3, 3]
                center_hom = np.hstack([center_world, 1.0])
                center_lidar_hom = np.linalg.inv(lidar_to_world_current) @ center_hom
                center = center_lidar_hom[:3]
                size = obj_info["sizes"][idx]
                dynamic_obj_infos.append({"center": center, "size": size})

                obj_dynamic_pcd = self.filter_points_in_box(
                    pointcloud_xyz, center, size
                )
                dynamic_pcd.append({instance_id: obj_dynamic_pcd})
                if obj_dynamic_pcd is not None and obj_dynamic_pcd.shape[0] > 0:
                    np.save(dynamic_pcd_path, obj_dynamic_pcd.astype(np.float32))
            
            # 静态点云
            static_pcd_path = os.path.join(
                static_pcd_file_folder, f"{str(i).zfill(3)}_static.npy"
            )
            if len(dynamic_obj_infos) == 0:
                print(f"[ Warning ]: No dynamic objects found for frame: {i}")
            static_pcd = self.filter_static_background_points(
                pointcloud_xyz, dynamic_obj_infos
            )
            np.save(static_pcd_path, static_pcd.astype(np.float32))
            
            single_frame_data["static_pcd"] = static_pcd
            single_frame_data["dynamic_pcd"] = dynamic_pcd
            
            frames_data.append(single_frame_data)
        
        return frames_data

    def load_rangeview(self, H_lidar, W_lidar):
        """生成 range view"""
        range_views = []
        masks = []
        
        for frame_idx in range(self.max_frame_num):
            pano, intensities, mask = lidar_to_pano_with_intensities(
                local_points_with_intensities=self.pcds[frame_idx],
                lidar_H=H_lidar,
                lidar_W=W_lidar,
                lidar_K=None,
                beam_inclinations=self.beam_inclinations,
                max_depth=self.max_depth,
                lidar_hfov=self.FOV_HORIZONTAL,
            )
            
            range_view = np.zeros((H_lidar, W_lidar, 3), dtype=np.float32)
            range_view[:, :, 1] = intensities
            range_view[:, :, 2] = pano
            
            ray_drop = np.where(
                range_view.reshape(-1, 3)[:, 2] <= 0.0, 0.0, 1.0
            ).reshape(H_lidar, W_lidar, 1).astype(np.float32)
            
            image_lidar = np.concatenate(
                [ray_drop, range_view[:, :, 1:2], range_view[:, :, 2:3]], axis=2
            )
            
            range_views.append(image_lidar)
            masks.append(mask)
        
        return range_views, masks

    # ========== 以下方法与原 Chery_Dataloader 对齐 ==========

    def get_dynamic_obj_id_list(self):
        """返回内部 model ids: background=0, dynamic=model_id=instance_id+1"""
        return self.model_id_list

    def get_instance_id_list(self):
        """返回 simplified instance ids (instance_id)."""
        return self.instance_id_list

    def get_raw_id_from_instance_id(self, instance_id):
        return self.instance_id_to_raw_id.get(int(instance_id))

    def load_dynamic_instance_id_list(self, train_frame_times, item_id_2_instance_id):
        """加载每个时间帧的 simplified instance ids (instance_id) 列表"""
        json_path = self.root_path + "/instances/frame_instances.json"
        with open(json_path, "r") as f:
            data = json.load(f)

        frame_object_ids = {}
        for frame_idx, item_ids in data.items():
            if int(frame_idx) not in train_frame_times:
                continue
            obj_ids = []
            for id in item_ids:
                if str(id) in item_id_2_instance_id:
                    obj_ids.append(int(id))
            frame_object_ids[int(frame_idx)] = obj_ids
        return frame_object_ids

    def load_dynamic_obj_info(self, train_frame_times):
        """加载动态障碍物标注 JSON 文件"""
        json_path = self.root_path + "/instances/instances_info.json"
        with open(json_path, "r") as f:
            data = json.load(f)
        
        annotations = {}
        item_id_2_instance_id = {}
        inv_start = np.linalg.inv(self.lidar_to_world_start)
        
        for item_id, obj_data in data.items():
            instance_id = int(item_id)
            class_name = obj_data["class_name"]
            raw_id = int(obj_data["id"])
            frame_ann = obj_data["frame_annotations"]
            frame_indices = frame_ann["frame_idx"]
            
            filtered_frame_indices = []
            obj_to_world_list = []
            box_sizes = []
            
            for frame_idx in frame_indices:
                if frame_idx not in train_frame_times:
                    continue
                filtered_frame_indices.append(frame_idx)
                obj_to_world_list.append(
                    frame_ann["obj_to_world"][frame_ann["frame_idx"].index(frame_idx)]
                )
                box_sizes.append(
                    frame_ann["box_size"][frame_ann["frame_idx"].index(frame_idx)]
                )
            
            frame_indices = filtered_frame_indices
            if len(frame_indices) == 0:
                continue
            
            poses = [np.array(mat, dtype=np.float32) for mat in obj_to_world_list]
            for pose in poses:
                pose = inv_start @ pose
            
            sizes = [list(sz) for sz in box_sizes]
            item_id_2_instance_id[item_id] = instance_id
            self.instance_id_to_raw_id[instance_id] = raw_id
            self.raw_id_to_instance_id[raw_id] = instance_id
            annotations[instance_id] = {
                "class_name": class_name,
                "instance_id": instance_id,
                "raw_id": raw_id,
                "frames": frame_indices,
                "poses": poses,
                "sizes": sizes,
            }
        
        return item_id_2_instance_id, annotations

    def filter_points_in_box(self, pointcloud, obj_center_pos, size, margin=0.5):
        """筛选出以obj_center_pos为中心、尺寸为size的矩形区域内的点云"""
        if pointcloud is None:
            return None
        if pointcloud.ndim != 2 or pointcloud.shape[1] != 4:
            print(
                "[Warning] filter_points_in_box: pointcloud shape is not (N, 4), got",
                pointcloud.shape,
            )
            return None
        
        half_size = np.array(size) / 2 + margin
        min_bounds = np.array(obj_center_pos) - half_size
        max_bounds = np.array(obj_center_pos) + half_size
        
        mask = (
            (pointcloud[:, 0] >= min_bounds[0])
            & (pointcloud[:, 0] <= max_bounds[0])
            & (pointcloud[:, 1] >= min_bounds[1])
            & (pointcloud[:, 1] <= max_bounds[1])
            & (pointcloud[:, 2] >= min_bounds[2])
            & (pointcloud[:, 2] <= max_bounds[2])
        )
        
        return pointcloud[mask]

    def filter_static_background_points(self, pointcloud, dynamic_obj_infos, margin=0.2):
        """从点云中移除动态物体区域的点云，保留静态背景点云"""
        if pointcloud is None:
            return None
        if pointcloud.ndim != 2 or pointcloud.shape[1] != 4:
            print(
                "[Warning] filter_static_background_points: pointcloud shape is not (N, 4)"
            )
            return None
        
        static_mask = np.ones(pointcloud.shape[0], dtype=bool)
        
        for obj_info in dynamic_obj_infos:
            center = obj_info["center"]
            size = obj_info["size"]
            
            half_size = np.array(size) / 2 + margin
            min_bounds = np.array(center) - half_size
            max_bounds = np.array(center) + half_size
            
            dynamic_mask = (
                (pointcloud[:, 0] >= min_bounds[0])
                & (pointcloud[:, 0] <= max_bounds[0])
                & (pointcloud[:, 1] >= min_bounds[1])
                & (pointcloud[:, 1] <= max_bounds[1])
                & (pointcloud[:, 2] >= min_bounds[2])
                & (pointcloud[:, 2] <= max_bounds[2])
            )
            
            static_mask &= ~dynamic_mask
        
        return pointcloud[static_mask]

    def delete_points_in_box_with_corners(self, pointcloud, box_corners):
        """
        根据边界框的8个顶点坐标，删除点云中位于该边界框内的点。
        """
        if pointcloud is None:
            return None
        if pointcloud.ndim != 2 or pointcloud.shape[1] != 3:
            print(
                "[Warning] delete_points_in_box_with_corners: pointcloud shape is not (N, 3), got",
                pointcloud.shape,
            )
            return None

        # 计算边界框的最小和最大坐标
        min_bounds = np.min(box_corners, axis=0)
        max_bounds = np.max(box_corners, axis=0)

        # 创建掩码，标记不在边界框内的点
        mask = (
            (pointcloud[:, 0] < min_bounds[0])
            | (pointcloud[:, 0] > max_bounds[0])
            | (pointcloud[:, 1] < min_bounds[1])
            | (pointcloud[:, 1] > max_bounds[1])
            | (pointcloud[:, 2] < min_bounds[2])
            | (pointcloud[:, 2] > max_bounds[2])
        )

        return pointcloud[mask]

    def filter_dynamic_objects(self, points, bboxes):
        """从世界系点云中过滤掉所有动态物体区域的点"""
        if len(bboxes) == 0:
            return points

        keep_mask = np.ones(len(points), dtype=bool)

        obj_corners = defaultdict(list)
        for item in bboxes:
            obj_corners[item["object_id"]].append(item["corners_world"])

        for corners_list in obj_corners.values():
            all_corners = np.concatenate(corners_list, axis=0)
            min_bound = all_corners.min(axis=0)
            max_bound = all_corners.max(axis=0)

            in_box = (
                (points[:, 0] >= min_bound[0])
                & (points[:, 0] <= max_bound[0])
                & (points[:, 1] >= min_bound[1])
                & (points[:, 1] <= max_bound[1])
                & (points[:, 2] >= min_bound[2])
                & (points[:, 2] <= max_bound[2])
            )
            keep_mask &= ~in_box

        return points[keep_mask]

    def load_dynamic_pcd(self, instance_id_str):
        """
        加载对应obj_id的各帧点云拼成一个完整obj的pcd
        同时每个obj会在哪几帧出现也会在这里处理和存储
        """
        instance_id = int(instance_id_str)
        obj_info = self.dynamic_obj_info[instance_id]
        obj_occurred_frames = obj_info["frames"]
        valid_frames = []
        render_frames = []
        obj_pcd = None
        obj_b2ls = dict()
        l2w_start_inv = np.linalg.inv(self.lidar_to_world_start)
        missing_frame_count = 0
        empty_frame_count = 0
        
        for frame_idx, frame in enumerate(obj_occurred_frames):
            dynamic_obj_path = os.path.join(
                self.preprocess_path,
                "dynamic_pcd",
                str(frame).zfill(3),
                f"{str(frame).zfill(3)}_obj{str(instance_id).zfill(3)}.npy"
            )
            
            l2w = self.l2ws[self.timestep_2_frameid[str(frame)]]
            obj_2_world = np.array(
                obj_info["poses"][frame_idx], dtype=np.float32
            ).reshape(4, 4)
            obj_2_world = l2w_start_inv @ obj_2_world
            obj_b2ls[str(self.timestep_2_frameid[str(frame)])] = (
                np.linalg.inv(l2w) @ obj_2_world
            )
            render_frames.append(frame)

            if os.path.exists(dynamic_obj_path):
                obj_frame_pcd = np.load(dynamic_obj_path).astype(np.float32)
            else:
                missing_frame_count += 1
                continue

            if obj_frame_pcd is None or obj_frame_pcd.shape[0] == 0:
                empty_frame_count += 1
                continue
            
            obj_frame_pcd = obj_frame_pcd.reshape(-1, 4)
            
            valid_frames.append(frame)
            
            curr_frame_b2l = obj_b2ls[str(self.timestep_2_frameid[str(frame)])]
            curr_frame_l2b = np.linalg.inv(curr_frame_b2l)
            obj_frame_pcd_2d = obj_frame_pcd[:, :3].reshape(-1, 3)
            intensity = obj_frame_pcd[:, 3:].reshape(-1, 1)
            obj_frame_pcd_hom = np.hstack([
                obj_frame_pcd_2d,
                np.ones((obj_frame_pcd_2d.shape[0], 1), dtype=np.float32)
            ])
            obj_frame_pcd = obj_frame_pcd_hom @ curr_frame_l2b.T
            obj_frame_pcd = obj_frame_pcd[:, :3]
            
            if obj_pcd is None:
                obj_pcd = obj_frame_pcd
            else:
                obj_pcd = np.vstack([obj_pcd, obj_frame_pcd])
        
        if (
            len(valid_frames) == 0
            or obj_pcd is None
            or len(obj_pcd) < self.MIN_OBJ_POINT_NUM
        ):
            return False, None
        
        self.obj_frames_id[str(instance_id)] = render_frames
        self.obj_o2l[str(instance_id)] = obj_b2ls
        
        obj_pcd_save_path = os.path.join(
            self.preprocess_path, "dynamic_pcd", "object_whole_pcd"
        )
        if not os.path.exists(obj_pcd_save_path):
            os.makedirs(obj_pcd_save_path)
        np.save(
            os.path.join(obj_pcd_save_path, f"obj{str(instance_id).zfill(3)}_whole_pcd.npy"),
            obj_pcd.astype(np.float32),
        )
        if missing_frame_count or empty_frame_count:
            print(
                f"[ Info ] instance_id {instance_id} (raw_id={obj_info['raw_id']}): kept {len(valid_frames)} frames, "
                f"rendered_on={len(render_frames)}, missing={missing_frame_count}, empty={empty_frame_count}"
            )
        
        return True, obj_pcd

    def load_static_pcd(self):
        """加载静态场景点云"""
        static_pcd = []
        static_pcd_path = os.path.join(
            self.preprocess_path, str(self.block_id) + "_static_scene_all_frames.npy"
        )

        pcd_xyzs = []
        for i in range(0, len(self.frames_data)):
            pcd = self.frames_data[i]["static_pcd"][:, :3]
            lidar_to_world = self.frames_data[i]["lidar2world"]
            R = lidar_to_world[:3, :3]
            T = lidar_to_world[:3, 3]
            pcd_transformed = (R @ pcd.T).T + T
            pcd_xyz = pcd_transformed[:, :3]
            pcd_xyzs.append(pcd_xyz)

        static_pcd = np.concatenate(pcd_xyzs, axis=0)
        np.save(static_pcd_path, static_pcd.astype(np.float32))
        print(f"[ Info ] Static scene has {static_pcd.shape[0]} points")
        
        return static_pcd

    def load_pcds(self, frames, train_frame_times, frame_num):
        """
        加载每帧静态点云作为 GT range view，计算场景 AABB。
        使用 static_pcd（已剔除动态 OD bbox 内的点），使背景高斯模型不会在
        动态物体区域学习 raydrop=1，从而避免去掉后处理 mask 后出现的
        乱点和自车扫描空洞四射问题。
        """
        R = np.eye(3, dtype=np.float32)
        T = np.zeros((3, 1), dtype=np.float32)
        self.extrinsic = np.block([[R, T], [0, 0, 0, 1]])

        pcds = []
        l2ws = []
        count_ind = 0

        for f_id, frame in enumerate(frames):
            if count_ind == frame_num:
                break

            if frame["log_time_stamp"] != int(train_frame_times[count_ind]):
                continue

            l2w = np.array(frame["lidar2world"])
            self.baselidar2world.append(l2w)
            sl2w = l2w @ self.extrinsic
            l2ws.append(sl2w)

            pcd = frame["raw_pcd"]

            pcd_world = (
                np.pad(pcd[..., :3], ((0, 0), (0, 1)), constant_values=1) @ l2w.T
            )[:, :3]
            aabb_min = np.min(pcd_world, axis=0)
            aabb_max = np.max(pcd_world, axis=0)
            self.aabb_min = np.minimum(self.aabb_min, aabb_min)
            self.aabb_max = np.maximum(self.aabb_max, aabb_max)

            vehicle_to_laser = np.linalg.inv(self.extrinsic)
            sensor_pcd = (
                np.pad(pcd[..., :3], ((0, 0), (0, 1)), constant_values=1)
                @ vehicle_to_laser.T
            )[:, :3]
            pcd[:, :3] = sensor_pcd[:, :3]
            pcds.append(pcd)
            
            timestep = str(frame["log_time_stamp"])
            self.timestep_2_frameid.update({timestep: count_ind})
            self.frameid_2_timestep.append(timestep)
            count_ind += 1
        
        if count_ind < frame_num:
            raise ValueError("Missing Frame or Abnormal Loading.")
        
        return pcds, l2ws

    def get_all_dynamic_bboxs(self):
        """
        返回所有动态物体的边界框信息（每一帧的所有id）
        """
        if len(self.dynamic_obj_info) == 0:
            return None
        all_dynamic_bboxs = []
        for obj_instance_id, obj_data in self.dynamic_obj_info.items():
            frame_indices = obj_data["frames"]
            for idx, frame_idx in enumerate(frame_indices):
                if frame_idx not in self.train_frame_times:
                    continue
                pose_info = obj_data["poses"][idx]
                size_info = obj_data["sizes"][idx]
                # pose 是标注文件里的全局世界坐标系，转换到相对 world 坐标系
                l2w_start_inv = np.linalg.inv(self.lidar_to_world_start)
                pose_world = l2w_start_inv @ pose_info
                pose_lidar = (
                    np.linalg.inv(self.l2ws[self.timestep_2_frameid[str(frame_idx)]])
                    @ pose_world
                )
                # 计算 bbox 的8个顶点坐标（lidar系下）
                l, w, h = size_info
                x_c, y_c, z_c = pose_lidar[:3, 3]
                R = pose_lidar[:3, :3]
                corners = np.array(
                    [
                        [ l/2,  w/2,  h/2],
                        [ l/2, -w/2,  h/2],
                        [-l/2, -w/2,  h/2],
                        [-l/2,  w/2,  h/2],
                        [ l/2,  w/2, -h/2],
                        [ l/2, -w/2, -h/2],
                        [-l/2, -w/2, -h/2],
                        [-l/2,  w/2, -h/2],
                    ]
                )
                rotated_corners = (R @ corners.T).T
                translated_corners = rotated_corners + np.array([x_c, y_c, z_c])
                # 转换到 world 系
                translated_corners_hom = np.hstack(
                    [translated_corners, np.ones((8, 1), dtype=translated_corners.dtype)]
                )
                translated_corners_world = (
                    self.l2ws[self.timestep_2_frameid[str(frame_idx)]]
                    @ translated_corners_hom.T
                ).T[:, :3]
                bbox_info = {
                    "instance_id": obj_instance_id,
                    "class_name": obj_data["class_name"],
                    "corners_lidar": translated_corners,
                    "corners_world": translated_corners_world,
                    "frame_idx": frame_idx,
                }
                all_dynamic_bboxs.append(bbox_info)
        return all_dynamic_bboxs

    def get_lidar_to_world_start(self):
        return self.lidar_to_world_start

    def get_frames_nums(self):
        """返回总帧数"""
        return self.max_frame_num

    def get_static_pcd(self):
        """返回静态场景点云"""
        return self.static_pcd

    def get_obj_pcd(self, model_id):
        """返回指定物体的点云"""
        instance_id = self.model_id_to_instance_id(model_id)
        return self.obj_pcd[str(instance_id)]

    def get_rangeview(self, frame_idx):
        """返回指定帧的range view [H,W,3]"""
        return self.range_views[frame_idx]

    def get_mask(self, frame_idx):
        """返回指定帧的mask [H,W,3]"""
        return self.masks[frame_idx]

    def getlidar2world(self):
        """返回所有帧的lidar2world变换列表"""
        return self.l2ws

    def get_obj2lidar(self, occurred_frame_idx, model_id, newcar_render=None):
        """返回物体到lidar的变换矩阵 [4,4]"""
        vehicle_to_laser = np.linalg.inv(self.extrinsic)
        instance_id = self.model_id_to_instance_id(model_id)
        return vehicle_to_laser @ self.obj_o2l[str(instance_id)][str(occurred_frame_idx)]

    def get_sensor2baselidar(self, sensorid):
        """字典返回每个雷达系到baselidar系的矩阵"""
        return self.sensor2baselidar[sensorid]

    def get_obj_frames(self, model_id):
        """返回物体出现的帧列表"""
        instance_id = self.model_id_to_instance_id(model_id)
        if str(instance_id) not in self.obj_frames_id:
            return []
        return self.obj_frames_id[str(instance_id)]

    def get_beam_inclination(self):
        """返回beam inclinations（弧度）"""
        return self.beam_inclinations

    def get_lidar_res(self):
        """返回LiDAR分辨率 (W, H)"""
        return self.W_lidar, self.H_lidar

    def get_fov_horizontal(self):
        """返回水平FOV（弧度）"""
        return self.FOV_HORIZONTAL

    def get_fov_up(self):
        """返回向上FOV（弧度）"""
        return self.FOV_UP

    def get_fov_down(self):
        """返回向下FOV（弧度）"""
        return self.FOV_DOWN

    def get_train_frame_times(self):
        """返回训练帧时间列表"""
        return self.train_frame_times

    def set_novel_poses_setting(self, novel_poses_setting):
        """设置新视角渲染配置"""
        self.novel_poses_setting = novel_poses_setting

    def get_novel_poses_setting(self):
        """获取新视角渲染配置"""
        return self.novel_poses_setting
