from pathlib import Path
import json
import numpy as np
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import defaultdict


# ==================== 数据结构 ====================

@dataclass
class VehiclePose:
    """单条车辆位姿观测"""
    timestamp: float       # 时间戳 (秒), 来自 header.timestamp / 1e6
    x: float               # 世界坐标 x
    y: float               # 世界坐标 y
    z: float               # 世界坐标 z
    yaw: float             # 朝向角 (弧度)
    pitch: float
    roll: float


@dataclass
class ObjectObs:
    """单条目标观测"""
    timestamp: float       # 时间戳 (秒), 来自 object['timestamp']
    obj_id: str            # track ID, 如 "4684"
    obj_type: str          # 原始类型, 如 "OT_VEHICLE"
    # 世界坐标系下的状态
    x: float               # bounding_box.x (中心点)
    y: float               # bounding_box.y
    z: float               # camera_mea.main_source.pos.z + 0.5 * bbox3d_height
    yaw: float             # 朝向角 (弧度)
    yaw_rate: float        # 角速度 (rad/s)
    # 速度 (m/s, 世界系)
    vx: float
    vy: float
    # 3D 框尺寸
    length: float
    width: float
    height: float


@dataclass
class SmoothedState:
    """卡尔曼滤波后单帧的平滑状态"""
    timestamp: float
    x: float
    y: float
    z: float
    yaw: float
    vx: float
    vy: float
    yaw_rate: float
    length: float
    width: float
    height: float


@dataclass
class SmoothedTrack:
    """一条经卡尔曼滤波平滑后的目标轨迹"""
    obj_id: str
    obj_type: str
    states: List[SmoothedState]   # 按时间升序排列
    t_first: float
    t_last: float

@dataclass
class ObjBbox:
    """目标对象的bounding box标注信息"""
    timestamp: float
    obj_id: str
    obj_type:str
    x: float
    y: float
    z: float
    yaw: float    
    length: float
    width: float
    height: float

# ==================== 类型映射 ====================

TYPE_MAP = {
    "OT_PEDESTRIAN": "Pedestrian",

    "OT_MOTORCYCLIST": "Motorcycle",
    "OT_CYCLIST": "Bicycle",
    "OT_TRICYCLIST": "Bicycle",
    
    "OT_VEHICLE": "Vehicle",               
    "OT_LARGE_VEHICLE": "Vehicle",

    "OT_BARRIER": "TrafficBarrier",
    "OT_BARRIER_ANTI_COLLISION_BUCKET": "TrafficBarrier",
    "OT_BARRIER_ANTI_COLLISION_POST": "TrafficBarrier",
    "OT_BARRIER_CEMENT_PILLAR": "TrafficBarrier",
    "OT_CONE": "TrafficBarrier",           # 锥桶可视为临时交通屏障
    "OT_WARNING_TRIANGLE": "TrafficBarrier", # 三角警示牌

    # 归入 Unknown（目标列表无对应）
    "OT_FOD": "Unknown",                   # 异物碎片
    "OT_VEGETATION": "Unknown",            # 植被
    "OT_UNKNOWN_STATIC": "Unknown",
    "OT_UNKNOWN_MOVABLE": "Unknown",
}

# ==================== 坐标变换（模块级工具函数）====================

def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll)"""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def world_to_vehicle(obj_x: float, obj_y: float, obj_z: float, obj_yaw: float,
                     veh_pose: VehiclePose) -> tuple:
    """
    将目标的世界坐标 + 朝向转换到车辆坐标系.

    位置:  X_veh = R^T · (X_world - t)
    朝向:  yaw_veh = yaw_world - yaw_vehicle  (归一化到 [-π, π])
    """
    t = np.array([veh_pose.x, veh_pose.y, veh_pose.z])
    R = euler_to_rotation_matrix(veh_pose.roll, veh_pose.pitch, veh_pose.yaw)
    R_inv = R.T

    pos_world = np.array([obj_x, obj_y, obj_z])
    pos_veh = R_inv @ (pos_world - t)

    yaw_veh = float((obj_yaw - veh_pose.yaw + np.pi) % (2 * np.pi) - np.pi)

    return pos_veh[0], pos_veh[1], pos_veh[2], yaw_veh


# ==================== 卡尔曼滤波器 ====================

def _normalize_angle(a: float) -> float:
    """将角度归一化到 [-π, π]"""
    return float((a + np.pi) % (2 * np.pi) - np.pi)


class ConstantVelocityKF:
    """
    恒速度模型卡尔曼滤波器.

    状态: [pos_0, ..., pos_{d-1}, vel_0, ..., vel_{d-1}]  (2d 维)
    观测: [pos_0, ..., pos_{d-1}]                         (d 维)

    仅观测位置，速度由滤波器隐式估计.
    """

    def __init__(self,
                 pos_dim: int,
                 init_meas: np.ndarray,
                 init_t: float,
                 q_pos: float = 0.1,     # 位置过程噪声
                 q_vel: float = 1.0,     # 速度过程噪声
                 r_pos: float = 0.05):   # 位置观测噪声
        """
        Args:
            pos_dim:    位置维度 (如 4 表示 x,y,z,yaw)
            init_meas:  初始观测向量, shape = (pos_dim,)
            init_t:     初始时间戳
            q_pos/q_vel: 过程噪声系数 (越大 = 越信任观测, 越小 = 越信任模型)
            r_pos:      观测噪声系数
        """
        self.pos_dim = pos_dim
        self.dim = 2 * pos_dim   # 状态总维度

        # ---- 初始状态: x = [meas, 0, ..., 0] ----
        self.x = np.zeros(self.dim)
        self.x[:pos_dim] = init_meas

        # ---- 初始协方差 ----
        self.P = np.eye(self.dim) * 0.5

        # ---- 观测矩阵: H = [I_d, 0_d] ----
        self.H = np.zeros((pos_dim, self.dim))
        self.H[:, :pos_dim] = np.eye(pos_dim)

        # ---- 噪声参数 ----
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.R = np.eye(pos_dim) * r_pos

        self.last_t = init_t

        # RTS 反向平滑所需中间矩阵 (在 predict 中填充)
        self.F_saved = np.eye(self.dim)
        self.x_pred  = self.x.copy()
        self.P_pred  = self.P.copy()

    def predict(self, t: float):
        """
        从 last_t 预测到 t 时刻.

        F 矩阵在 dt 上使用恒速度模型:
            pos_{t+dt} = pos_t + vel_t * dt
            vel_{t+dt} = vel_t
        """
        dt = t - self.last_t
        if dt <= 0:
            return

        # ---- 状态转移矩阵 F ----
        F = np.eye(self.dim)
        for i in range(self.pos_dim):
            F[i, self.pos_dim + i] = dt

        # ---- 过程噪声 Q (离散化白噪声) ----
        dt2 = dt * dt
        G = np.zeros((self.dim, self.pos_dim * 2))
        for i in range(self.pos_dim):
            G[i, i] = dt2 * 0.5
            G[self.pos_dim + i, i] = dt
            G[i, self.pos_dim + i] = dt2 * 0.5
            G[self.pos_dim + i, self.pos_dim + i] = dt

        Q_block = np.diag([self.q_pos] * self.pos_dim + [self.q_vel] * self.pos_dim)
        Q = G @ Q_block @ G.T

        # ---- 预测 ----
        self.F_saved = F.copy()            # 状态转移矩阵 (给 RTS 用)
        self.x = F @ self.x                # 先做预测
        self.P = F @ self.P @ F.T + Q
        self.x_pred = self.x.copy()        # 预测后状态 (update 前), 给 RTS 用
        self.P_pred = self.P.copy()        # 预测后协方差 (update 前), 给 RTS 用
        self.last_t = t

    def update(self, measurement: np.ndarray):
        """
        用观测更新状态.

        Args:
            measurement: 观测向量, shape = (pos_dim,)
        """
        # 计算创新
        z = measurement
        y = z - self.H @ self.x

        # 如果状态中有角度 (位置最后一维是 yaw 时作归一化)
        # 这里不硬编码, 让调用方自行处理; update 保持通用

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(self.dim) - K @ self.H) @ self.P


class ConstantAccelerationKF:
    """
    恒加速度模型卡尔曼滤波器.

    状态: [pos_0, ..., pos_{d-1}, vel_0, ..., vel_{d-1}, acc_0, ..., acc_{d-1}]  (3d 维)
    观测: [pos_0, ..., pos_{d-1}]                                                  (d 维)

    pos_{t+dt} = pos_t + vel_t·dt + ½·acc_t·dt²
    vel_{t+dt} = vel_t + acc_t·dt
    acc_{t+dt} = acc_t + noise          ← 加速度随机游走

    仅观测位置，速度和加速度由滤波器隐式估计.
    适合 cut-in、加减速等速度变化的场景.
    """

    def __init__(self,
                 pos_dim: int,
                 init_meas: np.ndarray,
                 init_t: float,
                 q_pos: float = 0.1,
                 q_vel: float = 1.0,
                 q_acc: float = 3.0,
                 r_pos: float = 0.05):
        self.pos_dim = pos_dim
        self.dim = 3 * pos_dim   # [pos..., vel..., acc...]

        # ---- 初始状态: x = [meas, 0, ..., 0] ----
        self.x = np.zeros(self.dim)
        self.x[:pos_dim] = init_meas

        # ---- 初始协方差 ----
        self.P = np.eye(self.dim) * 0.5

        # ---- 观测矩阵: H = [I_d, 0_d, 0_d] ----
        self.H = np.zeros((pos_dim, self.dim))
        self.H[:, :pos_dim] = np.eye(pos_dim)

        # ---- 噪声参数 ----
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.q_acc = q_acc
        self.R = np.eye(pos_dim) * r_pos

        self.last_t = init_t

        # RTS 反向平滑所需中间矩阵
        self.F_saved = np.eye(self.dim)
        self.x_pred  = self.x.copy()
        self.P_pred  = self.P.copy()

    def predict(self, t: float):
        """恒加速度模型预测."""
        dt = t - self.last_t
        if dt <= 0:
            return

        dt2 = dt * dt

        # ---- 状态转移矩阵 F ----
        # 每维: pos += vel*dt + 0.5*acc*dt², vel += acc*dt, acc 不变
        F = np.eye(self.dim)
        for i in range(self.pos_dim):
            p = i
            v = self.pos_dim + i
            a = 2 * self.pos_dim + i
            F[p, v] = dt
            F[p, a] = 0.5 * dt2
            F[v, a] = dt

        # ---- 过程噪声 Q (加速度被白噪声驱动) ----
        # G 映射: 每维的过程噪声 → 状态, G = [dt²/2, dt, 1]^T
        G = np.zeros((self.dim, self.pos_dim))
        for i in range(self.pos_dim):
            G[i, i] = 0.5 * dt2
            G[self.pos_dim + i, i] = dt
            G[2 * self.pos_dim + i, i] = 1.0

        Q = G @ (np.eye(self.pos_dim) * self.q_acc) @ G.T

        # ---- 预测 ----
        self.F_saved = F.copy()
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.x_pred = self.x.copy()
        self.P_pred = self.P.copy()
        self.last_t = t

    def update(self, measurement: np.ndarray):
        """标准 KF 更新 (与 CV 相同)."""
        y = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim) - K @ self.H) @ self.P


# ==================== 处理管线 ====================

class BoxExtractor:
    """
    从 objects_proto JSON 中提取 3D 目标框, 经卡尔曼滤波平滑后,
    按图像帧时间戳采样并转换到车辆坐标系, 写入 label_pred/.

    使用:
        extractor = BoxExtractor(parsed_data_root, scene_id, (start, end))
        extractor.load()                     # 读取 JSON
        extractor.smooth()                   # 卡尔曼滤波
        extractor.write_labels()             # 采样 + 坐标变换 + 写入
    """

    def __init__(self,
                 parsed_data_root: str,
                 scene_id: str,
                 time_range: tuple):
        slice_start, slice_end = time_range
        scene_id_s_e = f"{scene_id}_{slice_start}_{slice_end}"

        # ---- 路径 ----
        parsed_scene_dir = Path(parsed_data_root) / scene_id
        self.obj_json_dir   = parsed_scene_dir / f"lite_msg_{slice_start}_{slice_end}" / "objects_proto"
        self.pose_json_dir   = parsed_scene_dir / f"lite_msg_{slice_start}_{slice_end}" / "pose_proto"
        self.img_output_dir = parsed_scene_dir / scene_id_s_e
        self.frame_json_path = self.img_output_dir / "data_frame_seq.json"
        self.label_dir       = self.img_output_dir / "label_pred"
        self.label_dir.mkdir(parents=True, exist_ok=True)

        # ---- 中间数据 ----
        self.veh_poses: List[VehiclePose]           = []
        self.objects:   List[ObjectObs]              = []
        self.tracks:    Dict[str, List[ObjectObs]]   = {}   # obj_id → 观测序列

        # 卡尔曼滤波后的平滑结果
        self.smoothed_veh_poses: List[VehiclePose]       = []
        self.smoothed_tracks:    Dict[str, SmoothedTrack] = {}

        # 图像帧序列 (load 时填充)
        self.data_frame_seq: List[dict] = []

    # ========== 步骤 1: 数据读取 ==========

    def load(self):
        """读取 objects_proto JSON, 填充 veh_poses / objects / tracks."""
        self._load_frame_seq()
        # self._load_vehicle_poses()
        self._load_ego_poses()
        self._load_object_observations()
        self._group_by_id()
        self._print_summary()

    def _load_frame_seq(self):
        """读取 data_frame_seq.json → self.data_frame_seq"""
        with open(self.frame_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.data_frame_seq = [dic for dic in data['data_frame_seq_items']]

    # 弃用
    def _load_vehicle_poses(self):
        """
        遍历 objects_proto/*.json, 提取车辆位姿.
        位姿:  message['pose'] → {x, y, z, yaw, pitch, roll}
        时间戳: message['header']['timestamp'] (微秒字符串) → 秒
        """
        for json_file in sorted(self.obj_json_dir.glob('*.json')):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            msg = data.get('message', {})
            header = msg.get('header', {})
            pose = msg.get('pose', {})

            required = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
            if not pose or not all(k in pose for k in required):
                continue

            ts = int(header.get('timestamp', 0)) / 1_000_000.0

            self.veh_poses.append(VehiclePose(
                timestamp = ts,
                x         = pose['x'],
                y         = pose['y'],
                z         = pose['z'],
                yaw       = pose['yaw'],
                pitch     = pose['pitch'],
                roll      = pose['roll'],
            ))

        self.veh_poses.sort(key=lambda p: p.timestamp)

    def _load_ego_poses(self):
        """
        遍历 pose_proto/*.json, 提取车辆位姿.
        位姿:  message['pose'] → {x, y, z, yaw, pitch, roll}
        时间戳: message['header']['timestamp'] (微秒字符串) → 秒
        """
        for json_file in sorted(self.pose_json_dir.glob('*.json')):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
            msg = data.get('message', {})
            position = msg.get('pos_smooth')
            
            self.veh_poses.append(VehiclePose(
                timestamp = msg['timestamp'],
                x = position['x'],
                y = position['y'],
                z = position['z'],
                yaw = msg['yaw'],
                pitch = msg['pitch'],
                roll = msg['roll']
            ))
        
        self.veh_poses.sort(key=lambda p: p.timestamp)
        

    def _load_object_observations(self):
        """
        遍历 objects_proto/*.json, 提取目标观测.

        读取原则与 transfer_object3dbox_info 一致:
          - 必须有 camera_mea.main_source.pos (否则无法确定 z)
          - 必须有 bbox3d_height
          - x, y 取自 bounding_box
          - z = camera_mea.main_source.pos.z + 0.5 * bbox3d_height
        """
        for json_file in sorted(self.obj_json_dir.glob('*.json')):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            objects = data.get('message', {}).get('objects', [])

            for obj in objects:
                ts = obj.get('timestamp')
                bbox = obj.get('bounding_box', {})
                bbox3d_h = obj.get('bbox3d_height')

                # =====================================
                # if obj.get('id') not in ['1466', '1541', '1461', '1526']:
                #     continue
                # =====================================

                object_pos = obj.get('camera_mea', {}).get('main_source', {}).get('pos', None)
                if object_pos is None or bbox3d_h is None or ts is None or not bbox:
                    continue
                
                vel = obj.get('vel', {})

                self.objects.append(ObjectObs(
                    timestamp = float(ts),
                    obj_id    = str(obj.get('id', '')),
                    obj_type  = TYPE_MAP.get(obj.get('type', ''), 'unknown'),
                    x         = bbox.get('x', 0),
                    y         = bbox.get('y', 0),
                    z         = object_pos['z'] + 0.5 * bbox3d_h,
                    yaw       = obj.get('yaw', 0),
                    yaw_rate  = obj.get('yaw_rate', 0),
                    vx        = vel.get('x', 0),
                    vy        = vel.get('y', 0),
                    length    = bbox.get('length', 0),
                    width     = bbox.get('width', 0),
                    height    = bbox3d_h,
                ))

        self.objects.sort(key=lambda o: o.timestamp)

    def _group_by_id(self):
        """将 objects 按 obj_id 分组 → self.tracks, 每组按时间排序."""
        groups: Dict[str, List[ObjectObs]] = defaultdict(list)
        for obs in self.objects:
            groups[obs.obj_id].append(obs)
        for obj_id in groups:
            groups[obj_id].sort(key=lambda o: o.timestamp)
        self.tracks = dict(groups)

    def _print_summary(self):
        n_tracks = len(self.tracks)
        avg_len = len(self.objects) / n_tracks if n_tracks > 0 else 0
        print(f"[INFO] 车辆位姿: {len(self.veh_poses)}  条")
        print(f"[INFO] 目标观测: {len(self.objects)}  条")
        print(f"[INFO] 目标 track: {n_tracks} 条, 平均每 track {avg_len:.1f} 帧")

    # ========== 步骤 2: height 零值清洗 & 卡尔曼滤波 ==========

    def clean_height_zeros(self, max_zero_ratio: float = 0.3):
        """
        清洗各 object track 中的 height=0 观测:

        1. 若某 track 的 height=0 占比 > max_zero_ratio → 整条 track 删除
        2. 首尾连续的 0 值 → 裁剪掉
        3. 中间孤立的 0 值 → 用整条 track 的非零中位数替换

        Args:
            max_zero_ratio: 零值占比阈值, 超过则删 track, 默认 0.3 (30%)
        """
        drop_ids = []
        trim_count = 0
        fill_count = 0

        for obj_id, obs_list in self.tracks.items():
            heights = np.array([o.height for o in obs_list])
            n_total = len(heights)
            n_zero = int((heights == 0.0).sum())

            # ---- 1. 零值占比过高 → 整条删除 ----
            if n_zero / n_total > max_zero_ratio:
                drop_ids.append(obj_id)
                continue

            # ---- 2. 裁剪首尾连续零值 ----
            first_nonzero = 0
            while first_nonzero < n_total and heights[first_nonzero] == 0.0:
                first_nonzero += 1
            last_nonzero = n_total - 1
            while last_nonzero >= 0 and heights[last_nonzero] == 0.0:
                last_nonzero -= 1

            if first_nonzero > 0 or last_nonzero < n_total - 1:
                trim_count += (first_nonzero + (n_total - 1 - last_nonzero))
                obs_list[:] = obs_list[first_nonzero:last_nonzero + 1]
                heights = heights[first_nonzero:last_nonzero + 1]

            # ---- 3. 中间零值用非零中位数填充 ----
            if len(heights[heights > 0]) > 0:
                median_val = np.median(heights[heights > 0])
            else:
                median_val = 1.0  # 全部都是 0 但占比未超阈值, 用默认值

            for i, o in enumerate(obs_list):
                if o.height == 0.0:
                    o.height = float(median_val)
                    fill_count += 1

        # 删除整条 track
        for oid in drop_ids:
            del self.tracks[oid]

        print(f"[INFO] height 零值清洗: 删除 {len(drop_ids)} 条 track, "
              f"裁剪首尾 {trim_count} 帧, 中位数填充中间 {fill_count} 帧")

    # ========== 步骤 2: 卡尔曼滤波 ==========
    # 需要最少观测帧数才做滤波, 否则直接用原始观测

    MIN_TRACK_LEN = 3

    def smooth(self):
        """
        对 vehicle pose 时序和每条 object track 分别做卡尔曼滤波,
        填充 self.smoothed_veh_poses 和 self.smoothed_tracks.
        """
        print("[INFO] 清洗 height 零值...")
        self.clean_height_zeros()
        print("[INFO] 开始卡尔曼滤波平滑...")
        # 直接使用pose proto, 里面的ego vehcile位姿已经平滑过了
        # self.smoothed_veh_poses = self._smooth_vehicle_poses()
        self.smoothed_veh_poses = list(self.veh_poses)
        self.smoothed_tracks = self._smooth_object_tracks()
        print(f"[INFO] 滤波完成: vehicle pose {len(self.smoothed_veh_poses)} 条, "
              f"object track {len(self.smoothed_tracks)} 条")

    def _smooth_vehicle_poses(self) -> List[VehiclePose]:
        """
        对车辆位姿序列做前向卡尔曼滤波 + RTS 反向平滑.

        状态: [x, y, z, yaw,  vx, vy, vz, vyaw]  (8D)
        """
        if len(self.veh_poses) < 3:
            print("[WARN] 车辆位姿太少, 跳过滤波")
            return list(self.veh_poses)

        pos_dim = 4   # x, y, z, yaw
        n = len(self.veh_poses)

        # ---- 前向滤波 (同时保存 RTS 所需矩阵) ----
        forward_ts:   List[float]       = []
        forward_x:    List[np.ndarray]  = []   # 滤波后状态 x[i]
        forward_P:    List[np.ndarray]  = []   # 滤波后协方差 P[i]
        forward_F:    List[np.ndarray]  = []   # F[i]: t_{i-1}→t_i (i>=1 时有效)
        forward_xpred: List[np.ndarray] = []   # 预测状态 (update 前, i>=1 时有效)
        forward_Ppred: List[np.ndarray] = []   # 预测协方差 (update 前, i>=1 时有效)

        init_meas = np.array([
            self.veh_poses[0].x,
            self.veh_poses[0].y,
            self.veh_poses[0].z,
            self.veh_poses[0].yaw,
        ])
        kf = ConstantVelocityKF(
            pos_dim=pos_dim,
            init_meas=init_meas,
            init_t=self.veh_poses[0].timestamp,
            q_pos=0.05,
            q_vel=0.5,
            r_pos=0.02,
        )

        # 第 0 帧
        forward_ts.append(self.veh_poses[0].timestamp)
        forward_x.append(kf.x.copy())
        forward_P.append(kf.P.copy())
        forward_F.append(np.zeros((kf.dim, kf.dim)))       # 占位
        forward_xpred.append(np.zeros(kf.dim))              # 占位
        forward_Ppred.append(np.zeros((kf.dim, kf.dim)))    # 占位

        for i in range(1, n):
            vp = self.veh_poses[i]
            meas = np.array([vp.x, vp.y, vp.z, vp.yaw])

            est_yaw = kf.x[3]
            meas[3] = est_yaw + _normalize_angle(meas[3] - est_yaw)

            kf.predict(vp.timestamp)
            # 保存预测后的中间矩阵
            forward_F.append(kf.F_saved.copy())
            forward_xpred.append(kf.x_pred.copy())
            forward_Ppred.append(kf.P_pred.copy())

            kf.update(meas)
            kf.x[3] = _normalize_angle(kf.x[3])

            forward_ts.append(vp.timestamp)
            forward_x.append(kf.x.copy())
            forward_P.append(kf.P.copy())

        # ---- RTS 反向平滑 ----
        smoothed_x = [s.copy() for s in forward_x]
        for i in range(n - 2, -1, -1):
            P_i   = forward_P[i]
            F_next = forward_F[i + 1]
            P_pred_next = forward_Ppred[i + 1]
            C = P_i @ F_next.T @ np.linalg.inv(P_pred_next)
            residual = smoothed_x[i + 1] - forward_xpred[i + 1]
            residual[3] = _normalize_angle(residual[3])  # yaw 残差归一化到 [-π, π]
            smoothed_x[i] = forward_x[i] + C @ residual
            smoothed_x[i][3] = _normalize_angle(smoothed_x[i][3])

        # ---- 组装输出 ----
        result: List[VehiclePose] = []
        for i in range(n):
            vp = self.veh_poses[i]
            x = smoothed_x[i]
            result.append(VehiclePose(
                timestamp = forward_ts[i],
                x         = float(x[0]),
                y         = float(x[1]),
                z         = float(x[2]),
                yaw       = float(x[3]),
                pitch     = vp.pitch,
                roll      = vp.roll,
            ))

        return result

    def _smooth_object_tracks(self) -> Dict[str, SmoothedTrack]:
        """
        对每条 object track 做前向卡尔曼滤波 (CA 模型) + RTS 反向平滑.

        状态: [x, y, z, yaw, vx, vy, vz, vyaw, ax, ay, az, ayaw]  (12D)
        观测: [x, y, z, yaw]                                          (4D)

        CA 模型能捕捉速度变化, 对 cut-in/加减速场景优于 CV.
        速度 (vx/vy/yaw_rate) 和尺寸用 EMA 单独平滑.
        """
        result: Dict[str, SmoothedTrack] = {}

        pos_dim = 4        # x, y, z, yaw

        for obj_id, obs_list in self.tracks.items():
            n = len(obs_list)

            if n < self.MIN_TRACK_LEN:
                states = [
                    SmoothedState(
                        timestamp = o.timestamp,
                        x=o.x, y=o.y, z=o.z, yaw=o.yaw,
                        vx=o.vx, vy=o.vy, yaw_rate=o.yaw_rate,
                        length=o.length, width=o.width, height=o.height,
                    )
                    for o in obs_list
                ]
                result[obj_id] = SmoothedTrack(
                    obj_id=obj_id,
                    obj_type=obs_list[0].obj_type,
                    states=states,
                    t_first=states[0].timestamp,
                    t_last=states[-1].timestamp,
                )
                continue

            # ---- 尺寸平滑 (EMA) ----
            def _ema_smooth(values: List[float], alpha: float = 0.6) -> List[float]:
                smoothed = [values[0]]
                for v in values[1:]:
                    smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
                return smoothed

            len_sm = _ema_smooth([o.length for o in obs_list])
            wid_sm = _ema_smooth([o.width  for o in obs_list])
            hgt_sm = _ema_smooth([o.height for o in obs_list])

            # ---- 速度 EMA ----
            vx_ema = obs_list[0].vx
            vy_ema = obs_list[0].vy
            yr_ema = obs_list[0].yaw_rate
            vx_sm: List[float] = [vx_ema]
            vy_sm: List[float] = [vy_ema]
            yr_sm: List[float] = [yr_ema]

            # ---- 前向卡尔曼滤波 (同时保存 RTS 所需矩阵) ----
            forward_ts:    List[float]       = []
            forward_x:     List[np.ndarray]  = []
            forward_P:     List[np.ndarray]  = []
            forward_F:     List[np.ndarray]  = []
            forward_xpred: List[np.ndarray]  = []
            forward_Ppred: List[np.ndarray]  = []

            init_meas = np.array([
                obs_list[0].x,
                obs_list[0].y,
                obs_list[0].z,
                obs_list[0].yaw,
            ])
            kf = ConstantAccelerationKF(
                pos_dim=pos_dim,
                init_meas=init_meas,
                init_t=obs_list[0].timestamp,
                q_pos=0.5,
                q_vel=5.0,
                q_acc=5.0,    # 加速度过程噪声, 越大越能跟踪速度变化
                r_pos=0.1,
            )

            # 第 0 帧
            forward_ts.append(obs_list[0].timestamp)
            forward_x.append(kf.x.copy())
            forward_P.append(kf.P.copy())
            forward_F.append(np.zeros((kf.dim, kf.dim)))
            forward_xpred.append(np.zeros(kf.dim))
            forward_Ppred.append(np.zeros((kf.dim, kf.dim)))

            for i in range(1, n):
                o = obs_list[i]
                meas = np.array([o.x, o.y, o.z, o.yaw])

                est_yaw = kf.x[3]
                meas[3] = est_yaw + _normalize_angle(meas[3] - est_yaw)

                kf.predict(o.timestamp)
                forward_F.append(kf.F_saved.copy())
                forward_xpred.append(kf.x_pred.copy())
                forward_Ppred.append(kf.P_pred.copy())

                kf.update(meas)
                kf.x[3] = _normalize_angle(kf.x[3])

                forward_ts.append(o.timestamp)
                forward_x.append(kf.x.copy())
                forward_P.append(kf.P.copy())

                # 速度 EMA
                alpha_v = 0.5
                vx_ema = alpha_v * o.vx + (1 - alpha_v) * vx_ema
                vy_ema = alpha_v * o.vy + (1 - alpha_v) * vy_ema
                yr_ema = alpha_v * o.yaw_rate + (1 - alpha_v) * yr_ema
                vx_sm.append(vx_ema)
                vy_sm.append(vy_ema)
                yr_sm.append(yr_ema)

            # ---- RTS 反向平滑 ----
            smoothed_x = [s.copy() for s in forward_x]
            for i in range(n - 2, -1, -1):
                P_i = forward_P[i]
                F_next = forward_F[i + 1]
                P_pred_next = forward_Ppred[i + 1]
                C = P_i @ F_next.T @ np.linalg.inv(P_pred_next)
                residual = smoothed_x[i + 1] - forward_xpred[i + 1]
                residual[3] = _normalize_angle(residual[3])  # yaw 残差归一化到 [-π, π]
                smoothed_x[i] = forward_x[i] + C @ residual
                smoothed_x[i][3] = _normalize_angle(smoothed_x[i][3])

            # ---- 组装 SmoothedState ----
            states: List[SmoothedState] = []
            for i in range(n):
                x = smoothed_x[i]
                states.append(SmoothedState(
                    timestamp = forward_ts[i],
                    x         = float(x[0]),
                    y         = float(x[1]),
                    z         = float(x[2]),
                    yaw       = float(x[3]),
                    vx        = vx_sm[i],
                    vy        = vy_sm[i],
                    yaw_rate  = yr_sm[i],
                    length    = len_sm[i],
                    width     = wid_sm[i],
                    height    = hgt_sm[i],
                ))

            result[obj_id] = SmoothedTrack(
                obj_id    = obj_id,
                obj_type  = obs_list[0].obj_type,
                states    = states,
                t_first   = states[0].timestamp,
                t_last    = states[-1].timestamp,
            )

        return result

    # ========== 步骤 3: 采样 + 坐标变换 + 写入 ==========

    def write_labels(self):
        """
        对每个 data_frame_seq_item:
          3a. 在 main_timestamp 处获取车辆位姿
          3b. 在 main_timestamp 处采样每个目标的平滑状态
          3c. world → vehicle 坐标变换
          3d. 写 label JSON 到 label_pred/
        """
        for item in self.data_frame_seq:
            # 3a. 获取目标时间戳以及写入json的文件名
            target_ts = item['main_timestamp']
            label_file_name = item['data_frame_path']
            
            # 3b. 在main_timestamp获取车辆位姿
            veh_pose = self._get_veh_pos_at(target_ts)

            # 3c.在main_timestamp采样相关目标的框信息
            target_bboxes = []
            for track in self.smoothed_tracks.values():
                obj_type = track.obj_type   # 已在 _load_object_observations 中完成映射
                dynamic = ['Pedestrian', 'Motorcycle', 'Bicycle', 'Vehicle']

                if track.t_first <= target_ts <= track.t_last:
                    # 动目标进行采样返回采样目标状态
                    if obj_type in dynamic:
                        obj_bbox = self._sample_track_at(track, target_ts)
                        # 3d. 世界坐标系到车辆坐标系转换
                        obj_bbox.x, obj_bbox.y, obj_bbox.z, obj_bbox.yaw = world_to_vehicle(obj_x=obj_bbox.x, 
                                                                                            obj_y=obj_bbox.y, 
                                                                                            obj_z=obj_bbox.z, 
                                                                                            obj_yaw=obj_bbox.yaw, 
                                                                                            veh_pose=veh_pose)
                        target_bboxes.append(obj_bbox) 
                    # 静目标直接返回采样目标状态
                    else:
                        state = track.states[0]
                        # 3d. 世界坐标系到车辆坐标系转换
                        state.x, state.y, state.z, state.yaw = world_to_vehicle(obj_x=state.x, 
                                                                                obj_y=state.y, 
                                                                                obj_z=state.z, 
                                                                                obj_yaw=state.yaw, 
                                                                                veh_pose=veh_pose)
                        target_bboxes.append(ObjBbox(timestamp=state.timestamp, 
                                                    obj_id=track.obj_id, 
                                                    obj_type=track.obj_type,
                                                    x=state.x, 
                                                    y=state.y, 
                                                    z=state.z, 
                                                    yaw=state.yaw,
                                                    height=state.height, 
                                                    length=state.length, 
                                                    width=state.width))
                
            # 3e. 转为psr格式
            psr_list = []
            for bbox in target_bboxes:
                psr_list.append(self._trans_objbbox_to_psrbbox(bbox))

            # 3f. 写入label.json
            self._write_label_json(label_file_name, psr_list)

    @ staticmethod
    def _lerp(arr, ts_arr, target_ts):
        if target_ts <= ts_arr[0]:
            return arr[0]
        if target_ts >= ts_arr[-1]:
            return arr[-1]
        
        idx = np.searchsorted(ts_arr, target_ts)
        t0, t1 = ts_arr[idx -1], ts_arr[idx]
        alpha = (target_ts - t0) / (t1 - t0)

        return arr[idx -1] + alpha * (arr[idx] - arr[idx - 1])

    def _get_veh_pos_at(self, target_ts):
        ts_arr = [vp.timestamp for vp in self.smoothed_veh_poses]
        x_arr = [vp.x for vp in self.smoothed_veh_poses]
        y_arr = [vp.y for vp in self.smoothed_veh_poses]
        z_arr = [vp.z for vp in self.smoothed_veh_poses]
        yaw_arr = np.unwrap([vp.yaw for vp in self.smoothed_veh_poses])
        pitch_arr = [vp.pitch for vp in self.smoothed_veh_poses]
        roll_arr = [vp.roll for vp in self.smoothed_veh_poses]

        x = self._lerp(x_arr, ts_arr, target_ts)
        y = self._lerp(y_arr, ts_arr, target_ts)
        z = self._lerp(z_arr, ts_arr, target_ts)
        yaw = _normalize_angle(self._lerp(yaw_arr, ts_arr, target_ts))
        pitch = self._lerp(pitch_arr, ts_arr, target_ts)
        roll = self._lerp(roll_arr, ts_arr, target_ts)

        return VehiclePose(timestamp=target_ts,x=x,y=y,z=z,yaw=yaw,pitch=pitch,roll=roll)

    def _sample_track_at(self, track, target_ts):
        ts_arr = [state.timestamp for state in track.states]
        x_arr = [state.x for state in track.states]
        y_arr = [state.y for state in track.states]
        z_arr = [state.z for state in track.states]
        yaw_arr = np.unwrap([state.yaw for state in track.states])

        x = self._lerp(x_arr, ts_arr, target_ts)
        y = self._lerp(y_arr, ts_arr, target_ts)
        z = self._lerp(z_arr, ts_arr, target_ts)
        yaw = _normalize_angle(self._lerp(yaw_arr, ts_arr, target_ts))

        height = np.array([state.height for state in track.states]).mean()
        length = np.array([state.length for state in track.states]).mean()
        width = np.array([state.width for state in track.states]).mean()

        return ObjBbox(timestamp=target_ts, obj_id=track.obj_id, obj_type=track.obj_type,
                       x=x, y=y, z=z, yaw=yaw, height=height, length=length, width=width)
    
    @staticmethod
    def _trans_objbbox_to_psrbbox(objbbox:ObjBbox):
        psrbbox = {
            'timestamp': objbbox.timestamp,
            'obj_id': objbbox.obj_id,
            'obj_type': objbbox.obj_type,
            'psr': {
                'position': {
                    'x': objbbox.x,
                    'y': objbbox.y,
                    'z': objbbox.z,
                },
                'rotation': {
                    'x':0,
                    'y':0,
                    'z':objbbox.yaw
                },
                'scale': {
                    'x':objbbox.length,
                    'y':objbbox.width,
                    'z':objbbox.height
                }
            }
        }
        return psrbbox
    
    def _write_label_json(self, file_name, psr_list):
        label_json_path = self.label_dir / f"{file_name}.json"
        # 写入json
        with open(label_json_path, 'w', encoding='utf-8') as f:
            json.dump(psr_list, f, indent=4)
        


# ==================== 入口 ====================

def main():

    parser = argparse.ArgumentParser(description='从 object_proto.json 中提取 3dbox 信息（带滤波平滑）')
    parser.add_argument('--parsed_data_root', default="home/data/parsed/", help='parsed 数据根目录')
    parser.add_argument('--scene_id', type=str, help='场景 ID')
    parser.add_argument('--time_range', '-t', nargs=2, type=int, metavar=('START', 'END'),
                        required=False, help='切片时间范围, 如 -t 30 60')
    args = parser.parse_args()

    extractor = BoxExtractor(
        parsed_data_root = args.parsed_data_root,
        scene_id         = args.scene_id,
        time_range       = args.time_range,
    )
    extractor.load()
    extractor.smooth()
    extractor.write_labels()


if __name__ == "__main__":
    main()