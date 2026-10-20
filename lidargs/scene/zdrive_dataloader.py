import os
import numpy as np
import torch
from utils.lidar_utils import lidar_to_pano_with_intensities
import open3d as o3d
import json
import shutil


class ZDrive_Dataloader:
    """
    Load ZDrive dataset, output as Waymo format
    """

    FOV_HORIZONTAL = 120 * torch.pi / 180.0
    FOV_UP = 7.0 * torch.pi / 180.0  # 向上7度
    FOV_DOWN = 13.0 * torch.pi / 180.0  # 总共20度垂直视场
    NUM_BEAMS = 100  # 128线
    W_LIDAR = 2400  # 水平分辨率
    H_LIDAR = 100  # 垂直分辨率
    BEAM_INCLINATIONS = [
        -13.03,
        -11.82,
        -10.84,
        -10.03,
        -9.47,
        -9.07,
        -8.66,
        -8.25,
        -7.88,
        -7.47,
        -7.07,
        -6.66,
        -6.26,
        -5.86,
        -5.45,
        -5.05,
        -4.64,
        -4.55,
        -4.45,
        -4.34,
        -4.23,
        -4.14,
        -4.04,
        -3.94,
        -3.83,
        -3.73,
        -3.64,
        -3.53,
        -3.42,
        -3.33,
        -3.23,
        -3.13,
        -3.02,
        -2.92,
        -2.83,
        -2.72,
        -2.62,
        -2.52,
        -2.42,
        -2.32,
        -2.21,
        -2.12,
        -2.02,
        -1.91,
        -1.81,
        -1.71,
        -1.61,
        -1.51,
        -1.41,
        -1.31,
        -1.21,
        -1.11,
        -1.01,
        -0.91,
        -0.81,
        -0.71,
        -0.61,
        -0.51,
        -0.41,
        -0.30,
        -0.20,
        -0.10,
        0.00,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.81,
        0.91,
        1.00,
        1.11,
        1.21,
        1.31,
        1.41,
        1.51,
        1.61,
        1.71,
        1.81,
        1.91,
        2.01,
        2.11,
        2.21,
        2.31,
        2.41,
        2.52,
        2.62,
        3.03,
        3.43,
        3.84,
        4.24,
        4.65,
        5.05,
        5.46,
        5.87,
        6.28,
        6.69,
        7.09,
    ]
    MIN_OBJ_POINT_NUM = 100

    def __init__(self, args, train=True, train_frame_times=None, dtype=np.float32):
        self.train = train
        self.root_path = args.source_path
        self.case = args.caseid
        self.block_id = args.block_id

        self.frames_data = self.load_frames_data(train_frame_times)
        print("[ Info ] this case have {} frames totally".format(len(self.frames_data)))

        # 注意这里是顺序是从小到大，即从 -fov_down 到 +fov_up, 而且是弧度制
        self.beam_inclinations = np.linspace(
            -self.FOV_DOWN, self.FOV_UP, self.NUM_BEAMS, dtype=np.float32
        )

        # 假设雷达位置就是自车位置
        R = np.eye(3, dtype=np.float32)
        T = np.zeros((3, 1), dtype=np.float32)
        self.extrinsic = np.block([[R, T], [0, 0, 0, 1]])  # laser_to_vehicle

        self.W_lidar = int(self.W_LIDAR)
        self.H_lidar = int(self.H_LIDAR)
        self.train_frame_times = train_frame_times
        self.max_frame_num = len(train_frame_times)
        self.max_depth = args.max_depth
        print("max_depth", self.max_depth)
        self.aabb_min = np.ones(3, dtype=np.float32) * (10000000)
        self.aabb_max = np.ones(3, dtype=np.float32) * (-10000000)

        self.sensor2baselidar = dict()  # 记录每个lidar到toplidar的变换矩阵
        self.pcds = []  # 原始每帧点云
        self.pcds_label = []  # LiDAR语义label
        self.l2ws = []  # 每帧的l2w
        self.timestep_2_frameid = dict()  # 通过timsestep查询对应训练的帧的id _ 0 to 50
        self.frameid_2_timestep = []  # 通过frame id 反查询对应训练帧的timestep
        self.baselidar2world = []
        self.pcds, self.l2ws = self.load_pcds(
            frames=self.frames_data,
            train_frame_times=self.train_frame_times,
            frame_num=self.max_frame_num,
        )
        self.obj_id_list = [int(key) for key in self.dynamic_obj_info.keys()]

        self.obj_frames_id = (
            dict()
        )  # 通过obj_id 查询实例出现在哪几帧（列表）(frame id : 0-50)
        self.obj_pcd = dict()  # 通过obj_id 查询实例的拼接后的完整的pcd
        self.obj_o2l = (
            dict()
        )  # 通过obj_id 和对应那一帧的frame id查询实例的o2l ， 字典嵌套了一个字典
        if self.obj_id_list is not None:
            for obj_id in self.obj_id_list:
                success, obj_pcd = self.load_dynamic_pcd(str(obj_id))
                if not success:
                    print(
                        "[ Warning ]: Failed to load dynamic pcd for object_id:", obj_id
                    )
                    continue
                self.obj_pcd[str(obj_id)] = obj_pcd

        if train:
            self.static_pcd = self.load_static_pcd()

        self.range_views, self.masks = self.load_rangeview(self.H_lidar, self.W_lidar)

        self.novel_poses_setting = None

    def load_rangeview(self, H_lidar, W_lidar):
        """
        return [H,W,3] rangeiew
        """
        range_views = []
        masks = []
        for frame_idx in range(self.max_frame_num):
            pano, intensities, mask = lidar_to_pano_with_intensities(
                local_points_with_intensities=self.pcds[frame_idx],
                lidar_H=H_lidar,
                lidar_W=W_lidar,
                beam_inclinations=self.beam_inclinations,
                max_depth=self.max_depth,
            )
            range_view = np.zeros((H_lidar, W_lidar, 3))
            range_view[:, :, 1] = intensities
            range_view[:, :, 2] = pano
            ray_drop = np.where(
                range_view.reshape(-1, 3)[:, 2] <= 0.0, 0.0, 1.0
            ).reshape(H_lidar, W_lidar, 1)
            image_lidar = np.concatenate(
                [
                    ray_drop,
                    np.clip(range_view[:, :, 1, None], 0, 1),
                    range_view[:, :, 2, None],
                ],
                axis=-1,
            )
            range_views.append(image_lidar)
            masks.append(mask)
        return range_views, masks

    def load_pcds(self, frames, train_frame_times, frame_num):
        """
        加载每帧点云 作为gt 以及计算边界 其中selected_sensor = 0 / 1 / 3 / 4 分别代表 ROTOTOP, ROBO_BACK, ROBO_LEFT_FRONT, ROBO_RIGHT_FRONT
        return : 原始每帧点云_baselidar系
        """

        pcds = []
        pcds_label = []
        l2ws = []
        count_ind = 0
        for f_id, frame in enumerate(frames):
            if count_ind == frame_num:
                break
            if frame["log_time_stamp"] != int(train_frame_times[count_ind]):
                continue
            # 这里的 lidar2world 是baselidar系
            l2w = np.array(frame["lidar2world"])
            self.baselidar2world.append(l2w)
            sl2w = l2w @ self.extrinsic  # vehicle2wordl @ lidar2vehicle
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
        return pcds, l2ws  # , pcds_label

    def get_frames_nums(self):
        return self.max_frame_num

    def get_static_pcd(self):
        return self.static_pcd

    def get_obj_pcd(self, object_id):
        return self.obj_pcd[str(object_id)]

    def get_rangeview(self, frame_idx):
        """
        return [H,W,3] numpy
        """
        return self.range_views[frame_idx]

    def get_mask(self, frame_idx):
        """
        return [H,W,3] numpy
        """
        return self.masks[frame_idx]

    def getlidar2world(self):
        """
        return list 所有帧的sensorlidar2world
        """
        return self.l2ws

    def get_obj2lidar(self, occurred_frame_idx, object_id, newcar_render=None):
        """
        return [4,4] numpy 返回obj2lidar的矩阵
        baselidar2sensor @ obj2baselidar
        """
        vehicle_to_laser = np.linalg.inv(self.extrinsic)
        return (
            vehicle_to_laser @ self.obj_o2l[str(object_id)][str(occurred_frame_idx)]
        )  # TODO
        # return self.obj_o2l[str(object_id)][str(occurred_frame_idx)]

    def get_sensor2baselidar(self, sensorid):
        """
        字典返回每个雷达系到baselidar系的矩阵
        """
        return self.sensor2baselidar[self.lidar_map[sensorid]]

    # def get_sensor2baselidar_newcar(self,sensorid):
    #     return self.new_sensor2baselidar[self.lidar_map[sensorid]]

    def get_obj_frames(self, object_id):
        """
        列表返回obj出现的帧
        """
        if str(object_id) not in self.obj_frames_id:
            return []
        return self.obj_frames_id[str(object_id)]

    def get_dynamic_obj_id_list(self):
        """
        列表返回移动的obj的id
        """
        return self.obj_id_list

    def get_beam_inclination(self):
        return self.beam_inclinations

    def get_lidar_res(self):
        return self.W_lidar, self.H_lidar

    def get_fov_horizontal(self):
        return self.FOV_HORIZONTAL

    def get_fov_up(self):
        return self.FOV_UP

    def get_fov_down(self):
        return self.FOV_DOWN

    def load_dynamic_obj_id_list(self, train_frame_times):
        """
        加载每个时间帧的动态障碍物 ID 列表

        返回:
            frame_object_ids: list of str, 包含所有动态障碍物的 ID
        """
        json_path = self.root_path + "/instances/frame_instances.json"
        with open(json_path, "r") as f:
            data = json.load(f)

        frame_object_ids = {}
        for frame_idx, obj_ids in data.items():
            if int(frame_idx) not in train_frame_times:
                continue
            frame_object_ids[int(frame_idx)] = obj_ids
        return frame_object_ids

    def load_dynamic_obj_info(self, train_frame_times):
        """
        加载动态障碍物标注 JSON 文件

        返回:
            annotations: dict, key 为 object_id (str), value 为 dict 包含：
                - class_name: str
                - frames: list of int (出现的帧索引)
                - poses: list of np.ndarray (4x4 齐次变换矩阵, 到第一帧的世界坐标)
                - sizes: list of list [l, w, h] (沿 x/y/z 的 bbox 尺寸)
        """
        json_path = self.root_path + "/instances/instances_info.json"
        with open(json_path, "r") as f:
            data = json.load(f)

        annotations = {}
        inv_start = np.linalg.inv(self.lidar_to_world_start)
        for obj_id, obj_data in data.items():
            class_name = obj_data["class_name"]
            frame_ann = obj_data["frame_annotations"]

            frame_indices = frame_ann["frame_idx"]  # list of int
            filtered_frame_indices = []
            for frame_idx in frame_indices:
                if frame_idx not in train_frame_times:
                    continue
                filtered_frame_indices.append(frame_idx)
            frame_indices = filtered_frame_indices
            if len(frame_indices) == 0:
                continue
            obj_to_world_list = frame_ann["obj_to_world"]  # list of 4x4 matrices
            box_sizes = frame_ann["box_size"]  # list of [l, w, h]

            poses = [np.array(mat, dtype=np.float32) for mat in obj_to_world_list]
            for pose in poses:
                pose = inv_start @ pose
            sizes = [list(sz) for sz in box_sizes]

            annotations[obj_id] = {
                "class_name": class_name,
                "frames": frame_indices,
                "poses": poses,
                "sizes": sizes,
            }

        return annotations

    def load_dynamic_pcd(self, object_id):
        """
        加载对应obj_id的各帧点云拼成一个完整obj的pcd, 同时每个obj会在哪几帧出现也会在这里处理和存储
        reutrn : obj拼接后的点云
        """
        obj_info = self.dynamic_obj_info[object_id]
        obj_occurred_frames = obj_info["frames"]
        obj_pcd = None
        obj_b2ls = dict()
        l2w_start_inv = np.linalg.inv(self.lidar_to_world_start)
        empty_frame_list = []
        for frame in obj_occurred_frames:
            dynamic_obj_path = (
                self.root_path
                + "/lidar"
                + "/dynamic_pcd/"
                + str(frame).zfill(3)
                + "/"
                + str(frame).zfill(3)
                + "_obj"
                + str(object_id).zfill(3)
                + ".txt"
            )
            if not os.path.exists(dynamic_obj_path):
                print(
                    "[ Warning ]: Dynamic object pcd file not found:", dynamic_obj_path
                )
                continue
            obj_frame_pcd = np.loadtxt(dynamic_obj_path).astype(np.float32)
            if obj_frame_pcd is None or obj_frame_pcd.shape[0] == 0:
                empty_frame_list.append(frame)
                print(
                    "[ Warning ]: Dynamic object pcd is empty for obj_id:",
                    object_id,
                    "frame:",
                    frame,
                )
                continue
            obj_frame_pcd = obj_frame_pcd.reshape(-1, 4)
            # 标注文件里的坐标obj_info['poses'],是全局的世界坐标系
            # 实际上我们还基于第一帧构建了另一个相对的world坐标系
            # slef.l2ws是在这个相对的world坐标系下的
            # 全局的世界坐标系 到 相对的world坐标系 的变换矩阵是self.lidar_to_world_start
            # 所以我们需要先把obj_info['poses']转换到相对的world坐标系下，然后再计算obj2lidar
            l2w = self.l2ws[self.timestep_2_frameid[str(frame)]]
            obj_2_world = np.array(
                obj_info["poses"][obj_occurred_frames.index(frame)], dtype=np.float32
            ).reshape(4, 4)
            obj_2_world = l2w_start_inv @ obj_2_world
            obj_b2ls[str(self.timestep_2_frameid[str(frame)])] = (
                np.linalg.inv(l2w) @ obj_2_world
            )
            # obj_frame_pcd是lidar系下的点，统一转到obj系，再拼接
            curr_frame_b2l = obj_b2ls[str(self.timestep_2_frameid[str(frame)])]
            curr_frame_l2b = np.linalg.inv(curr_frame_b2l)
            obj_frame_pcd_2d = obj_frame_pcd[:, :3].reshape(-1, 3)
            intensity = obj_frame_pcd[:, 3:].reshape(-1, 1)
            obj_frame_pcd_hom = np.hstack(
                [
                    obj_frame_pcd_2d,
                    np.ones((obj_frame_pcd_2d.shape[0], 1), dtype=np.float32),
                ]
            )
            obj_frame_pcd = obj_frame_pcd_hom @ curr_frame_l2b.T
            obj_frame_pcd = obj_frame_pcd[:, :3]
            obj_frame_pcd[:, 3:] = intensity
            if obj_pcd is None:
                obj_pcd = obj_frame_pcd
            else:
                obj_pcd = np.vstack([obj_pcd, obj_frame_pcd])
        for empty_frame in empty_frame_list:
            obj_occurred_frames.remove(empty_frame)
        if (
            len(obj_occurred_frames) == 0
            or len(obj_b2ls) != len(obj_occurred_frames)
            or len(obj_pcd) < self.MIN_OBJ_POINT_NUM
        ):
            print("Warning: No valid frames found for object_id:", object_id)
            return False, None

        self.obj_frames_id[str(object_id)] = obj_occurred_frames
        self.obj_o2l[str(object_id)] = obj_b2ls
        # 保存每个obj为一个单独的pcd文件，方便后续查看
        obj_pcd_save_path = os.path.join(
            self.root_path, "mclidar", "dynamic_pcd", "object_whole_pcd"
        )
        if not os.path.exists(obj_pcd_save_path):
            os.makedirs(obj_pcd_save_path)
        np.savetxt(
            os.path.join(
                obj_pcd_save_path, f"obj{str(object_id).zfill(3)}_whole_pcd.txt"
            ),
            obj_pcd,
        )
        return True, obj_pcd

    def get_lidar_to_world_start(self):
        return self.lidar_to_world_start

    def load_frames_data(self, train_frame_times):
        """
        加载指定训练帧的点云数据和位姿信息，并按照动态od标注信息，提取动静态点云，返回每帧数据的列表。
        """
        lidar_filefolder = os.path.join(self.root_path, "mclidar")
        bin_files = [f for f in os.listdir(lidar_filefolder) if f.endswith(".bin")]
        bin_files = sorted(bin_files, key=lambda x: int(x.split(".")[0]))

        lidarpose_filefolder = os.path.join(self.root_path, "lidar_pose")

        self.start_frame = train_frame_times[0]
        frame_cnt = train_frame_times[-1] - train_frame_times[0] + 1

        frames_data = []
        self.lidar_to_world_start = np.loadtxt(
            os.path.join(
                lidarpose_filefolder, str(bin_files[0].split(".")[0]).zfill(3) + ".txt"
            )
        )

        self.dynamic_obj_list = self.load_dynamic_obj_id_list(train_frame_times)
        self.dynamic_obj_info = self.load_dynamic_obj_info(train_frame_times)
        for i in range(self.start_frame, self.start_frame + frame_cnt):
            """
            1. log_time_stamp
            2. lidar2world
            3. raw_pcd
            4. lidar_id
            5. static_pcd
            6. dynamic_obj_pcd
            """
            if i < self.start_frame or i not in train_frame_times:
                continue
            single_frame_data = {}
            single_frame_data["log_time_stamp"] = i

            lidar_pose_i_path = os.path.join(
                lidarpose_filefolder, str(i).zfill(3) + ".txt"
            )
            lidar_to_world_current = np.loadtxt(lidar_pose_i_path)
            lidar_to_world = (
                np.linalg.inv(self.lidar_to_world_start) @ lidar_to_world_current
            )
            single_frame_data["lidar2world"] = lidar_to_world

            # x y z intensity timestamp ring lidar_id
            lidar_info = np.fromfile(
                os.path.join(self.root_path, "mclidar", str(i).zfill(3) + ".bin"),
                dtype=np.float32,
            ).reshape(-1, 7)

            # 过滤掉lidar_id不为0的点
            lidar_info = lidar_info[lidar_info[:, 6] == 0]

            # 强度在0-1之间
            lidar_info[:, 3] = lidar_info[:, 3] / 255.0
            single_frame_data["raw_pcd"] = lidar_info[:, :4]  # x,y,z,intensity
            single_frame_data["lidar_id"] = lidar_info[:, 6]

            # 提取动态和静态点云，如果文件存在，则直接加载，否则进行筛选
            dynamic_pcd = []
            static_pcd = []
            dynamic_pcd_file_folder = os.path.join(
                lidar_filefolder, "dynamic_pcd", str(i).zfill(3)
            )
            static_pcd_file_folder = os.path.join(lidar_filefolder, "static_pcd")
            if not os.path.exists(static_pcd_file_folder):
                os.makedirs(static_pcd_file_folder)
            if not os.path.exists(dynamic_pcd_file_folder):
                os.makedirs(dynamic_pcd_file_folder)

            dynamic_obj_id_list = self.dynamic_obj_list[i]

            # pointcloud_xyz是lidar系下的
            pointcloud_xyz = single_frame_data["raw_pcd"][:, :4]
            dynamic_obj_infos = []
            dynamic_pcd_data = []
            for obj_id in dynamic_obj_id_list:

                obj_info = self.dynamic_obj_info[str(obj_id)]
                frame_indices = obj_info["frames"]
                if i not in frame_indices:
                    continue

                dynamic_pcd_path = os.path.join(
                    dynamic_pcd_file_folder,
                    f"{str(i).zfill(3)}_obj{str(obj_id).zfill(3)}.txt",
                )
                if os.path.exists(dynamic_pcd_path):
                    obj_dynamic_pcd = np.loadtxt(dynamic_pcd_path).astype(np.float32)
                    dynamic_pcd.append({obj_id: obj_dynamic_pcd})
                else:
                    idx = frame_indices.index(i)

                    # 标注文件里的坐标obj_info['poses'],在全局的世界坐标系下
                    # 实际上我们基于第一帧构建了另一个相对的world坐标系
                    # 全局的世界坐标系 到 相对的world坐标系 的变换矩阵是self.lidar_to_world_start
                    center_world = obj_info["poses"][idx][:3, 3]
                    center_hom = np.hstack([center_world, 1.0])
                    center_lidar_hom = (
                        np.linalg.inv(lidar_to_world_current) @ center_hom
                    )
                    center = center_lidar_hom[:3]  # lidar系下的中心点

                    # box_size是lidar系下的
                    size = obj_info["sizes"][idx]

                    dynamic_obj_infos.append({"center": center, "size": size})

                    obj_dynamic_pcd = self.filter_points_in_box(
                        pointcloud_xyz, center, size
                    )
                    dynamic_pcd.append({obj_id: obj_dynamic_pcd})
                    # 保存动态点云到文件
                    dynamic_pcd_path = os.path.join(
                        dynamic_pcd_file_folder,
                        f"{str(i).zfill(3)}_obj{str(obj_id).zfill(3)}.txt",
                    )
                    np.savetxt(dynamic_pcd_path, obj_dynamic_pcd)
            static_pcd_path = os.path.join(
                static_pcd_file_folder, f"{str(i).zfill(3)}_static.txt"
            )
            if os.path.exists(static_pcd_path):
                static_pcd = np.loadtxt(static_pcd_path).astype(np.float32)
            else:
                if len(dynamic_obj_infos) == 0:
                    print("[ Warning ]: No dynamic objects found for frame:", i)
                static_pcd = self.filter_static_background_points(
                    pointcloud_xyz, dynamic_obj_infos
                )
                static_pcd = pointcloud_xyz
                # 保存静态点云到文件
                static_pcd_path = os.path.join(
                    static_pcd_file_folder, f"{str(i).zfill(3)}_static.txt"
                )
                np.savetxt(static_pcd_path, static_pcd)

            single_frame_data["static_pcd"] = static_pcd
            single_frame_data["dynamic_pcd"] = dynamic_pcd

            frames_data.append(single_frame_data)

        return frames_data

    def load_static_pcd(self):
        static_pcd = []
        static_pcd_file_path = os.path.join(
            self.root_path, str(self.block_id) + "_static_scene_all_frames.npy"
        )
        if os.path.exists(static_pcd_file_path):
            static_pcd = np.load(static_pcd_file_path)
        else:
            # 拼接训练帧点云作为静态场景，后续的高斯初始化需要
            pcd_xyzs = []
            for i in range(0, len(self.frames_data)):
                pcd = self.frames_data[i]["static_pcd"][:, :3]
                lidar_to_world = self.frames_data[i]["lidar2world"]
                R = lidar_to_world[:3, :3]
                T = lidar_to_world[:3, 3]
                pcd_transformed = (R @ pcd.T).T + T  # shape (N, 3)
                pcd_xyz = pcd_transformed[:, :3]  # shape: (N_i, 3)
                pcd_xyzs.append(pcd_xyz)
            static_pcd = np.concatenate(pcd_xyzs, axis=0)  # shape: (total_points, 3)
            # 再过滤一次动态od
            all_dynamic_bboxs = self.get_all_dynamic_bboxs()
            static_pcd = self.filter_dynamic_objects(static_pcd, all_dynamic_bboxs)
            np.save(static_pcd_file_path, static_pcd)
            print(f"[ Info ] static scene have {static_pcd.shape[0]} points")

        return static_pcd

    def filter_points_in_box(self, pointcloud, obj_center_pos, size):
        """
        筛选出以obj_center_pos为中心、尺寸为size的矩形区域内的点云

        参数:
            pointcloud: numpy数组，形状为(N, 4)，表示点云数据
            obj_center_pos: 列表或数组，表示矩形中心坐标[x, y, z]
            size: 列表或数组，表示矩形在x、y、z三个维度上的尺寸[l, w, h]

        返回:
            筛选后的点云数据
        """
        if pointcloud is None:
            return None
        if pointcloud.ndim != 2 or pointcloud.shape[1] != 4:
            print(
                "[Warning] filter_points_in_box: pointcloud shape is not (N, 4), got",
                pointcloud.shape,
            )
            return None

        # 计算矩形边界
        half_size = np.array(size) / 2
        min_bounds = np.array(obj_center_pos) - half_size
        max_bounds = np.array(obj_center_pos) + half_size

        # 筛选在矩形边界内的点
        mask = (
            (pointcloud[:, 0] >= min_bounds[0])
            & (pointcloud[:, 0] <= max_bounds[0])
            & (pointcloud[:, 1] >= min_bounds[1])
            & (pointcloud[:, 1] <= max_bounds[1])
            & (pointcloud[:, 2] >= min_bounds[2])
            & (pointcloud[:, 2] <= max_bounds[2])
        )

        return pointcloud[mask]

    def filter_static_background_points(self, pointcloud, dynamic_obj_infos):
        """
        从点云中移除动态物体区域的点云，保留静态背景点云。

        参数:
            pointcloud: numpy数组，形状为(N, 4)，表示点云数据
            dynamic_obj_infos: 列表，包含多个动态物体的信息，每个元素是一个字典，包含：
                - 'center': 物体中心坐标[x, y, z]
                - 'size': 物体尺寸[l, w, h]
        返回:
            静态背景点云数据
        """
        if pointcloud is None:
            return None
        if pointcloud.ndim != 2 or pointcloud.shape[1] != 4:
            print(
                "[Warning] filter_points_in_box: pointcloud shape is not (N, 4), got",
                pointcloud.shape,
            )
            return None

        static_mask = np.ones(pointcloud.shape[0], dtype=bool)

        for obj_info in dynamic_obj_infos:
            center = obj_info["center"]
            size = obj_info["size"]

            half_size = np.array(size) / 2
            min_bounds = np.array(center) - half_size
            max_bounds = np.array(center) + half_size

            obj_mask = (
                (pointcloud[:, 0] >= min_bounds[0])
                & (pointcloud[:, 0] <= max_bounds[0])
                & (pointcloud[:, 1] >= min_bounds[1])
                & (pointcloud[:, 1] <= max_bounds[1])
                & (pointcloud[:, 2] >= min_bounds[2])
                & (pointcloud[:, 2] <= max_bounds[2])
            )

            static_mask &= ~obj_mask

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
        if len(bboxes) == 0:
            return points

        # Step 1: 构建全局 mask（初始全 True：保留所有点）
        keep_mask = np.ones(len(points), dtype=bool)

        # Step 2: 按 object 合并 bbox
        from collections import defaultdict

        obj_corners = defaultdict(list)
        for item in bboxes:
            obj_corners[item["object_id"]].append(item["corners_world"])

        # Step 3: 对每个 object 计算合并 AABB，并更新 mask
        for corners_list in obj_corners.values():
            all_corners = np.concatenate(corners_list, axis=0)
            min_bound = all_corners.min(axis=0)
            max_bound = all_corners.max(axis=0)

            # 向量化判断：点是否在 AABB 内
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

    def get_train_frame_times(self):
        return self.train_frame_times

    def get_all_dynamic_bboxs(self):
        """
        返回所有动态物体的边界框信息
        每一帧的所有id
        """
        if len(self.dynamic_obj_info) == 0:
            return None
        all_dynamic_bboxs = []
        for obj_id, obj_data in self.dynamic_obj_info.items():
            frame_indices = obj_data["frames"]
            for idx, frame_idx in enumerate(frame_indices):
                if frame_idx not in self.train_frame_times:
                    continue
                pose_info = obj_data["poses"][idx]
                size_info = obj_data["sizes"][idx]
                # 将pose转换到lidar系下， pose是标注文件里的全局世界坐标系
                l2w_start_inv = np.linalg.inv(self.lidar_to_world_start)
                pose_world = l2w_start_inv @ pose_info
                pose_lidar = (
                    np.linalg.inv(self.l2ws[self.timestep_2_frameid[str(frame_idx)]])
                    @ pose_world
                )
                # 计算bbox的8个顶点坐标(world系下)
                l, w, h = size_info
                x_c, y_c, z_c = pose_lidar[:3, 3]
                R = pose_lidar[:3, :3]
                corners = np.array(
                    [
                        [l / 2, w / 2, h / 2],
                        [l / 2, -w / 2, h / 2],
                        [-l / 2, -w / 2, h / 2],
                        [-l / 2, w / 2, h / 2],
                        [l / 2, w / 2, -h / 2],
                        [l / 2, -w / 2, -h / 2],
                        [-l / 2, -w / 2, -h / 2],
                        [-l / 2, w / 2, -h / 2],
                    ]
                )
                rotated_corners = (R @ corners.T).T
                translated_corners = rotated_corners + np.array([x_c, y_c, z_c])
                # 转换到world系下
                translated_corners_hom = np.hstack(
                    [
                        translated_corners,
                        np.ones((8, 1), dtype=translated_corners.dtype),
                    ]
                )
                translated_corners_world = (
                    self.l2ws[self.timestep_2_frameid[str(frame_idx)]]
                    @ translated_corners_hom.T
                ).T[:, :3]
                bbox_info = {
                    "object_id": obj_id,
                    "class_name": obj_data["class_name"],
                    "corners_lidar": translated_corners,
                    "corners_world": translated_corners_world,
                    "frame_idx": frame_idx,
                }
                all_dynamic_bboxs.append(bbox_info)
        return all_dynamic_bboxs

    def set_novel_poses_setting(self, novel_poses_setting):
        self.novel_poses_setting = novel_poses_setting

    def get_novel_poses_setting(self):
        return self.novel_poses_setting
