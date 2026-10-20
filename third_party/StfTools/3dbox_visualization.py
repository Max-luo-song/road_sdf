#!/usr/bin/env python3
"""
将3D标注框反投影到相机图像上。

读取label目录下的3D box标注JSON，结合camera_params.json（相机内外参）和
每个时间戳文件夹下的data_frame.json（车辆位姿），将3D框投影到各相机图像上。

用法:
    python project_3dbox_to_images.py [--data_dir DATA_DIR] [--output_dir OUTPUT_DIR] [--max_frames N]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import cv2
from moviepy.editor import VideoFileClip, clips_array

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

CAMERA_ID_TO_NAME = {
    75: "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H110",
    76: "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H60",
    77: "CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H30",
    97: "CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H15",
    78: "CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99",
    83: "CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H99",
    84: "CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H30",
    79: "CAM_PBQ_FRONT_RIGHT_RESET_OPTICAL_H99",
    85: "CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H99",
    86: "CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H30",
    82: "CAM_PBQ_REAR_RESET_OPTICAL_H50",
}

GRID_VIDEO_CAM = [
            "CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99",
            "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H60",
            "CAM_PBQ_FRONT_RIGHT_RESET_OPTICAL_H99",
            "CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H99",
            "CAM_PBQ_REAR_RESET_OPTICAL_H50",
            "CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H99",
            ]

# ============================================================
# 绘制
# ============================================================

# 不同目标类型的颜色 (BGR)
TYPE_COLORS = {
    'Vehicle': (0, 255, 0),          # 绿色
    'Car': (0, 255, 0),              # 绿色
    'Pedestrian': (0, 0, 255),       # 红色
    'Motorcycle': (255, 0, 0),       # 蓝色
    'Bicycle': (255, 255, 0),        # 青色
    'TrafficBarrier': (0, 255, 255), # 黄色
}

BEV_FRAME_ALPHA = 0.4       # BEV 框填充透明度
BEV_FRAME_EDGE_COLOR = 'lime'
BEV_ARROW_COLOR = 'yellow'
BEV_DOT_COLOR = 'red'

# 3D框的边定义 (按角点索引)
# 底部: 0-1-2-3, 顶部: 4-5-6-7, 竖线: (0,4), (1,5), (2,6), (3,7)
BOTTOM_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]
TOP_EDGES = [(4, 5), (5, 6), (6, 7), (7, 4)]
VERTICAL_EDGES = [(0, 4), (1, 5), (2, 6), (3, 7)]

# ============================================================
# 坐标系约定
# ============================================================
#
# 世界坐标系 (World): 与车辆系原点不同但轴线对齐
# 车辆坐标系 (Vehicle): X-前, Y-左, Z-上
# 相机坐标系 (Camera):   X-右, Y-下, Z-前 (光轴，OpenCV/ROS标准)
#
# 相机 → 车辆的基变换 (当相机"正装"看向车辆前方时):
#   Cam X (右)  → Veh -Y (右 = -左)
#   Cam Y (下)  → Veh -Z (下 = -上)
#   Cam Z (前)  → Veh  X
R_BASE_CAM2VEH = np.array([
    [0,  0,  1],   # Cam X → Veh
    [-1, 0,  0],   # Cam Y → Veh
    [0, -1,  0],   # Cam Z → Veh
])
R_BASE_VEH2CAM = R_BASE_CAM2VEH.T  # = [[0,-1,0],[0,0,-1],[1,0,0]]

# ============================================================
# 坐标系变换 (全部使用 4×4 齐次变换矩阵)
# ============================================================


# ==================工具函数===================
def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Euler角 → 3×3旋转矩阵 (绕固定轴: 先Z, 再Y, 再X)。
    用于绕**车辆坐标系**轴的旋转。
    旋转顺序: R = Rz(yaw) · Ry(pitch) · Rx(roll)
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr, cr]])
    Ry = np.array([[cp, 0, sp],
                   [0, 1, 0],
                   [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0],
                   [sy, cy, 0],
                   [0, 0, 1]])

    return Rz @ Ry @ Rx


def make_transform_4x4(R, t):
    """由 3×3 旋转矩阵 R 和 3×1 平移向量 t 构造 4×4 齐次变换矩阵"""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _transform_points(points, T):
    """
    用 4×4 齐次变换矩阵 T 批量变换点集。

    Args:
        points: (N, 3) ndarray
        T: (4, 4) 齐次变换矩阵

    Returns:
        (N, 3) 变换后的点
    """
    N = points.shape[0]
    points_h = np.column_stack([points, np.ones(N, dtype=np.float64)])  # (N, 4)
    return (T @ points_h.T).T[:, :3]  # (N, 3)


class BboxProjector():

    def __init__(self, data_dir, output_folder):

        # ============设置路径=============
        self.data_dir = Path(data_dir)
        self.label_dir = self.data_dir / 'label_pred'
        self.camera_params_path = self.data_dir / 'camera_params.json'
        self.data_frame_seq_path = self.data_dir / 'data_frame_seq.json'
        self.output_dir = self.data_dir / output_folder
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ============加载数据=============
        self.camera_params = {}
        self.timestamp_folders = []
    
    def load(self):
        self._load_cam_params()
        self._load_timestamp()

    def _load_cam_params(self):
        if not self.camera_params_path.exists():
            print(f"[ERROR] 找不到相机参数文件: {self.camera_params_path}")
            sys.exit(1)
        with open(self.camera_params_path) as f:
            all_cam_params = json.load(f)

        self.camera_params = {
            cam_name: params
            for cam_name, params in all_cam_params.items()
            if cam_name in CAMERA_ID_TO_NAME.values()
        }
        print(f"[INFO] 加载了 {len(self.camera_params)} 个相机的参数")
    
    def _load_timestamp(self):
        if not self.data_frame_seq_path.exists():
            print(f"[ERROR] 找不到时间戳记录文件: {self.data_frame_seq_path}")
            sys.exit(1)
        with open (self.data_frame_seq_path) as f:
            data_frame_seq_items = json.load(f)["data_frame_seq_items"]
        self.timestamp_folders = [d["data_frame_path"] for d in data_frame_seq_items]

        print(f"[INFO] 有 {len(self.timestamp_folders)} 帧数据待处理")
    
    def project(self):

        total_boxes_drawn = 0
        total_images = 0

        for ts in self.timestamp_folders:
            # 标签json路径和图片路径(都和timestamp有关)
            json_name = str(ts) + '.json'
            label_path = self.label_dir / json_name
            images_dir = self.data_dir / ts

            # 从.json中加载label
            if not label_path.exists():
                continue
            with open(label_path) as f:
                labels = json.load(f)
            if not labels:
                continue
            
            # 图片路径
            image_files = sorted([
                f for f in images_dir.iterdir()
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
            ])

            for img_path in image_files:
                cam_name = self._extract_camera_name(img_path.name, self.camera_params.keys())
                if cam_name is None:
                    continue
                if cam_name not in self.camera_params:
                    continue
                    
                # 读取图片对应相机内外参
                cam_param = self.camera_params[cam_name]
                extrinsics = cam_param['camera_to_vehicle_extrinsics']
                intrinsics = cam_param['intrinsics'].copy()  # 后面要修改intrinsics

                # 读取图像
                image = cv2.imread(str(img_path))
                if image is None:
                    continue

                # =============投影每个3d框===============
                for obj in labels:
                    psr = obj['psr']
                    position = psr['position']
                    rotation = psr['rotation']
                    scale = psr['scale']
                    obj_type = obj.get('obj_type', 'unknown')
                    obj_id = obj.get('obj_id', '')

                    try:
                        # 1) 局部系下8个角点
                        corners = self._get_8_corners(position, rotation, scale)

                        # 2) 车辆 → 相机
                        corners_camera = self._vehicle_to_camera(corners, extrinsics)

                        # 3) 相机 → 像素
                        corners_2d, valid = self._project_to_image(corners_camera, intrinsics)

                        # 4) 绘制
                        self._draw_3dbox_on_image(image, corners_2d, valid, obj_type, obj_id)
                        total_boxes_drawn += 1
                    except Exception as e:
                        print(f"    [WARN] 投影框失败 (obj={obj_id}): {e}")
                
                # 保存标注后的图像（按相机名称归入子文件夹）
                cam_out_dir = self.output_dir / cam_name
                cam_out_dir.mkdir(parents=True, exist_ok=True)
                img_save_name = f"{img_path.name}"
                img_save_path = cam_out_dir / img_save_name
                cv2.imwrite(str(img_save_path), image)
                total_images += 1
            
            print(f"  ✓ {ts}: {len(image_files)}张图, {len(labels)}个框")

        print(f"\n{'='*60}")
        print(f"[DONE] 完成!")
        print(f"  处理了 {total_images} 张图像")
        print(f"  绘制了 {total_boxes_drawn} 个3D框")

        print(f"  结果保存到: {self.output_dir}")

    @staticmethod
    def _extract_camera_name(filename, camera_list):
        """
        从图片文件名中提取相机名称。
        文件名格式: {CAMERA_NAME}-{timestamp}.jpg

        通过匹配known_cameras中的最长匹配前缀来确定相机名。
        """
        name = Path(filename).stem
        # 按长度降序排列，优先匹配更长的名称（避免 "CAM_PBQ_FRONT" 匹配到 "CAM_PBQ_FRONT_WIDE" 的部分）
        for cam_name in sorted(camera_list, key=len, reverse=True):
            if name.startswith(cam_name):
                return cam_name
        return None

    @staticmethod
    def _get_8_corners(position, rotation, scale):
        """
        获取3D框的8个角点坐标。

        局部坐标系: x-前(长), y-左(宽), z-上(高), 原点在框的几何中心

        Args:
            position: {x, y, z}  框的位置
            rotation: {x, y, z}  旋转 (roll, pitch, yaw, 弧度)
            scale: {x, y, z}     x=长, y=宽, z=高

        Returns:
            corners: (8, 3) ndarray, 8个角点坐标
        """
        l, w, h = scale['x'], scale['y'], scale['z']
        tx, ty, tz = position['x'], position['y'], position['z']


        # 局部坐标系下的8个角点 (中心在原点)
        corners_local = np.array([
            [+l / 2, +w / 2, -h / 2],  # 0: 前-左-下
            [+l / 2, -w / 2, -h / 2],  # 1: 前-右-下
            [-l / 2, -w / 2, -h / 2],  # 2: 后-右-下
            [-l / 2, +w / 2, -h / 2],  # 3: 后-左-下
            [+l / 2, +w / 2, +h / 2],  # 4: 前-左-上
            [+l / 2, -w / 2, +h / 2],  # 5: 前-右-上
            [-l / 2, -w / 2, +h / 2],  # 6: 后-右-上
            [-l / 2, +w / 2, +h / 2],  # 7: 后-左-上
        ], dtype=np.float64)

        # 绕Z,Y,X旋转 (yaw/pitch/roll → Rz·Ry·Rx)，构成 4×4 齐次变换
        roll_x, pitch_y, yaw_z = rotation['x'], rotation['y'], rotation['z']
        R = euler_to_rotation_matrix(roll_x, pitch_y, yaw_z)
        T_box = make_transform_4x4(R, np.array([tx, ty, tz]))

        return _transform_points(corners_local, T_box)
    
    @staticmethod
    def _vehicle_to_camera(points_vehicle, extrinsics):
        """
        车辆坐标系 → 相机坐标系。

        extrinsics (camera_to_vehicle) 定义了相机在车辆系下的完整位姿:
            X_veh = R_ypr · R_BASE_CAM2VEH · X_cam + t
        其中 R_ypr = Rz(yaw)·Ry(pitch)·Rx(roll) 绕**车辆系**轴的额外旋转。

        逆变换 (车辆→相机):
            T_camera←vehicle = [R_BASE_VEH2CAM·R_ypr^T | -R_BASE_VEH2CAM·R_ypr^T·t]

        Args:
            points_vehicle: (N, 3) 车辆系点
            extrinsics: {x, y, z, roll, pitch, yaw}

        Returns:
            (N, 3) 相机系点
        """
        t = np.array([extrinsics['x'], extrinsics['y'], extrinsics['z']])
        R_ypr = euler_to_rotation_matrix(extrinsics['roll'], extrinsics['pitch'], extrinsics['yaw'])
        R_veh2cam = R_BASE_VEH2CAM @ R_ypr.T
        T_v2c = make_transform_4x4(R_veh2cam, -R_veh2cam @ t)
        return _transform_points(points_vehicle, T_v2c)

    @staticmethod
    def _project_to_image(points_camera, intrinsics):
        """
        相机坐标系 → 像素坐标 (含畸变)。

        使用OpenCV Rational畸变模型:
            radial = (1 + k1·r² + k2·r⁴ + k3·r⁶) / (1 + k4·r² + k5·r⁴ + k6·r⁶)
        当k4=k5=k6=0时退化为标准RadTan模型。

        Args:
            points_camera: (N, 3) 相机系点 (z>0 在相机前方)
            intrinsics: {fx, fy, cx, cy, k1-k6, p1, p2}

        Returns:
            points_2d: (N, 2) 像素坐标
            valid: (N,) bool, True表示该点在相机前方
        """
        fx = intrinsics['fx']
        fy = intrinsics['fy']
        cx = intrinsics['cx']
        cy = intrinsics['cy']

        # k = np.array([intrinsics.get(f'k{i}', 0.0) for i in range(1, 7)], dtype=np.float64)
        # p = np.array([intrinsics.get(f'p{i}', 0.0) for i in range(1, 3)], dtype=np.float64)

        z = points_camera[:, 2]
        valid = z > 0

        # x = points_camera[:, 0]
        # y = points_camera[:, 1]

        # # 归一化平面
        # xn = np.divide(x, z, where=valid, out=np.zeros_like(x))
        # yn = np.divide(y, z, where=valid, out=np.zeros_like(y))

        # r2 = xn ** 2 + yn ** 2
        # r4 = r2 ** 2
        # r6 = r2 ** 3

        # # 径向畸变 (Rational model)
        # radial_num = 1 + k[0] * r2 + k[1] * r4 + k[2] * r6
        # radial_den = 1 + k[3] * r2 + k[4] * r4 + k[5] * r6
        # radial = np.divide(radial_num, radial_den, where=valid, out=np.ones_like(radial_num))

        # xd = xn * radial
        # yd = yn * radial

        # # 切向畸变
        # if np.any(np.abs(p) > 1e-10):
        #     xd = xd + (2 * p[0] * xn * yn + p[1] * (r2 + 2 * xn ** 2))
        #     yd = yd + (p[0] * (r2 + 2 * yn ** 2) + 2 * p[1] * xn * yn)

        # u = fx * xd + cx
        # v = fy * yd + cy
        # 归一化平面 → 像素 (纯针孔, 不施加畸变)
        u = np.divide(fx * points_camera[:, 0], z, where=valid, out=np.zeros(points_camera.shape[0]))
        v = np.divide(fy * points_camera[:, 1], z, where=valid, out=np.zeros(points_camera.shape[0]))
        u = u + cx
        v = v + cy

        return np.column_stack([u, v]), valid
        
    @staticmethod
    def _draw_3dbox_on_image(image, corners_2d, valid, obj_type='Vehicle', obj_id=''):
        """
        在图像上绘制3D框。

        Args:
            image: BGR图像 (会被原地修改)
            corners_2d: (8, 2) 角点像素坐标
            valid: (8,) 有效标志
            obj_type: 目标类型
            obj_id: 目标ID
        """
        if not np.all(valid):
            return  # 任何角点在相机后方则跳过

        # 检查是否有 NaN 或 Inf
        if not np.all(np.isfinite(corners_2d)):
            return

        # 检查角点是否在合理像素范围内，避免 int32 溢出
        if np.any(np.abs(corners_2d) > 1e5):
            return

        # 视锥体裁剪: 检查投影框是否与图像区域有交集
        # 窄 FOV 相机 (如 H30) 上, 大量目标虽在相机前方但完全不在 FOV 内,
        # 投影后角点落在图像外 (如 u=2000~4000), 画出多余线条。跳过这些框。
        h_img, w_img = image.shape[:2]
        u_min, v_min = corners_2d.min(axis=0)
        u_max, v_max = corners_2d.max(axis=0)
        if u_max < 0 or u_min > w_img or v_max < 0 or v_min > h_img:
            return

        # cv2.line 本身会对越界线段做裁剪, 不需要手动 clip 角点到图像边界.
        # 手动 clip 会导致部分在画面外的框坍缩为一个点/线.
        corners = corners_2d.astype(np.int32)
        color = TYPE_COLORS.get(obj_type, (255, 255, 255))

        # 底部四条边
        for i1, i2 in BOTTOM_EDGES:
            cv2.line(image, tuple(corners[i1]), tuple(corners[i2]), color, 2)

        # 顶部四条边
        for i1, i2 in TOP_EDGES:
            cv2.line(image, tuple(corners[i1]), tuple(corners[i2]), color, 2)

        # 四条竖线
        for i1, i2 in VERTICAL_EDGES:
            cv2.line(image, tuple(corners[i1]), tuple(corners[i2]), color, 2)

        # 在顶部面的中心标注obj_id
        top_center = np.mean(corners[4:8], axis=0).astype(np.int32)
        label = f"{obj_type[:4]}_{obj_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image,
                    (top_center[0] - tw // 2 - 2, top_center[1] - th - 4),
                    (top_center[0] + tw // 2 + 2, top_center[1]),
                    color, -1)
        cv2.putText(image, label, (top_center[0] - tw // 2, top_center[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    def visualize(self, bev_video_path=None):
        """
        各相机图片 → 单相机视频 → 6 相机网格视频, 可选与 BEV 视频左右拼接.

        Args:
            bev_video_path: BEV 视频路径 (Path or str), 为 None 则跳过拼接
        """
        video_paths = []

        # ---- 1. 每个相机图片 → 视频 ----
        for cam_name in GRID_VIDEO_CAM:
            cam_img_dir = self.output_dir / cam_name

            image_files = sorted([
                f for f in cam_img_dir.iterdir()
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
            ])
            output_name = str(cam_name) + ".mp4"

            self._images_to_video(images=image_files,
                                  output=self.output_dir / output_name,
                                  fps=10.0)

            if cam_name in GRID_VIDEO_CAM:
                 video_paths.append(self.output_dir / output_name)

        # ---- 2. 6 相机网格视频 ----
        grid_path = self.output_dir / 'grid_video.mp4'
        self._create_video_grid(video_paths, rows=2, cols=3,
                                output=grid_path, fps=10.0)

        # ---- 3. 与 BEV 视频左右拼接 ----
        if bev_video_path is not None and Path(bev_video_path).exists():
            print("[INFO] 拼接 grid + BEV 左右视频...")
            grid_clip = VideoFileClip(str(grid_path))
            bev_clip  = VideoFileClip(str(bev_video_path))

            # 统一高度 (以 grid 为准)
            target_h = grid_clip.h
            if bev_clip.h != target_h:
                bev_clip = bev_clip.resize(height=target_h)

            final = clips_array([[grid_clip, bev_clip]])
            final_path = self.output_dir / 'final_video.mp4'
            final.to_videofile(str(final_path), fps=10.0)

            grid_clip.close()
            bev_clip.close()
            final.close()
            print(f"[INFO] 最终视频已保存至: {final_path}")

    def _images_to_video(self,
                        images: list[Path],
                        output: str = "output.mp4",
                        fps: float = 10.0,
                        width: int | None = None,
                        height: int | None = None,
                        codec: str = "mp4v",
                        keep_aspect: bool = True,
                        verbose: bool = True,
                    ) -> str:
        """
        将图片列表拼接为一个视频。

        参数:
            images: 图片文件路径列表
            output: 输出视频文件路径
            fps: 帧率
            width, height: 强制输出分辨率（None 则取首图尺寸）
            codec: FourCC 编码器（mp4v → .mp4, xvid → .avi, avc1 → .mp4 H.264）
            keep_aspect: 缩放时是否保持宽高比（letterbox）
            verbose: 是否打印进度
        """
        if not images:
            raise ValueError("图片列表为空")

        # 读取首图，确定视频尺寸
        first = cv2.imread(str(images[0]))
        if first is None:
            raise ValueError(f"无法读取图片: {images[0]}")

        h, w = first.shape[:2]

        if width is None and height is None:
            width, height = w, h
        elif width is None:
            width = int(height * w / h)
        elif height is None:
            height = int(width * h / w)

        target_size = (width, height)

        if verbose:
            print(f"输入尺寸: {w}x{h}")
            print(f"输出尺寸: {width}x{height}")
            print(f"图片数量: {len(images)}")
            print(f"帧率:     {fps} fps")
            print(f"时长:     {len(images) / fps:.1f} 秒")
            print(f"编码器:   {codec}")
            print(f"输出文件: {output}")

        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output, fourcc, fps, target_size)

        if not writer.isOpened():
            raise RuntimeError(f"无法创建视频写入器（编码器 {codec} 可能不支持此扩展名）")

        total = len(images)
        for i, img_path in enumerate(images):
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"⚠ 跳过损坏图片: {img_path}", file=sys.stderr)
                continue

            # 缩放到目标尺寸
            if (frame.shape[1], frame.shape[0]) != target_size:
                if keep_aspect:
                    frame = self.__letterbox_resize(frame, width, height)
                else:
                    frame = cv2.resize(frame, target_size)

            writer.write(frame)

            if verbose and (i + 1) % max(1, total // 20) == 0:
                pct = (i + 1) / total * 100
                print(f"\r进度: {i + 1}/{total} ({pct:.0f}%)", end="", flush=True)

        writer.release()
        if verbose:
            print(f"\r完成! {total} 张图片已写入 {output}")

        return output

    def __letterbox_resize(img, target_w: int, target_h: int) -> "cv2.Mat":
        """保持宽高比缩放到目标尺寸，不足部分用黑色填充（letterbox/pillarbox）。"""
        h, w = img.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        canvas = (0, 0, 0)  # 黑色背景
        if len(img.shape) == 3:
            canvas = (0, 0, 0)
        result = cv2.copyMakeBorder(
            resized,
            top=(target_h - new_h) // 2,
            bottom=target_h - new_h - (target_h - new_h) // 2,
            left=(target_w - new_w) // 2,
            right=target_w - new_w - (target_w - new_w) // 2,
            borderType=cv2.BORDER_CONSTANT,
            value=canvas,
        )
        return result
        # 视频拼接

        # bev2d框

    @staticmethod
    def _create_video_grid(video_paths, rows, cols, output, fps):
        """
        将多个视频拼成 rows 行 cols 列的网格
        
        参数:
            video_paths: list, 视频文件路径列表（按行排列，从左到右）
            rows: int, 行数
            cols: int, 列数
            output_path: str, 输出文件路径
            fps: int, 输出视频的帧率
        """
        total_needed = rows * cols
        if len(video_paths) != total_needed:
            raise ValueError(f"需要恰好 {total_needed} 个视频，但传入了 {len(video_paths)} 个")
        
        # 加载所有视频
        clips = [VideoFileClip(str(path)) for path in video_paths]

        # 构建二维列表
        grid = []
        for i in range(rows):
            row_start = i * cols
            grid.append(clips[row_start : row_start + cols])

        # 拼接
        final_clip = clips_array(grid)
        # 导出
        final_clip.to_videofile(str(output), fps=fps)

        print(f"视频已保存至: {output}")
        
        # 释放资源
        for clip in clips:
            clip.close()
        final_clip.close()

    @staticmethod
    def _draw_bev_frame(ax, labels, max_range=50):
        """
        在 matplotlib axes 上绘制单帧 BEV 视图: 同心圆网格 + 2D 目标框。

        Args:
            ax: matplotlib axes
            labels: list of dict, label JSON 中的对象列表
            max_range: 最大显示半径 (m), 默认 50
        """
        ax.clear()
        ax.set_facecolor('black')

        # ---- 1. 同心圆网格 (每 10m) ----
        for r in range(10, max_range + 1, 10):
            circle = plt.Circle((0, 0), r, color='white', linewidth=0.5,
                                fill=False, alpha=0.6)
            ax.add_patch(circle)

        # ---- 2. 坐标范围 & 比例 ----
        margin = max_range + 5
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)
        ax.set_aspect('equal')
        ax.axis('off')

        # ---- 3. 绘制每个目标的 2D 框 ----
        for obj in labels:
            psr = obj['psr']
            position = psr['position']
            rotation = psr['rotation']
            scale = psr['scale']

            x   = position['x']
            y   = position['y']
            yaw = rotation['z']
            length = scale['x']
            width  = scale['y']

            # 车辆系 (X-前, Y-左) → 绘图系 (u-右, v-上)
            u_center = -y
            v_center =  x

            # 局部角点 (中心原点, x-前, y-左)
            corners = np.array([
                [-length/2, -width/2],
                [ length/2, -width/2],
                [ length/2,  width/2],
                [-length/2,  width/2],
            ])
            rot_z = np.array([[np.cos(yaw), -np.sin(yaw)],
                              [np.sin(yaw),  np.cos(yaw)]])
            corners_rot = corners @ rot_z.T

            # 变换到绘图坐标
            corners_plot = np.zeros_like(corners_rot)
            corners_plot[:, 0] = -corners_rot[:, 1] + u_center
            corners_plot[:, 1] =  corners_rot[:, 0] + v_center

            # 半透明填充矩形
            poly = patches.Polygon(corners_plot, closed=True,
                                   color=BEV_FRAME_EDGE_COLOR,
                                   alpha=BEV_FRAME_ALPHA, linewidth=1.0)
            ax.add_patch(poly)

            # 中心点
            ax.plot(u_center, v_center, 'o', color=BEV_DOT_COLOR, markersize=2)

            # 朝向箭头
            arrow_len = length / 2
            arrow_end_u = - (y + np.sin(yaw) * arrow_len)
            arrow_end_v =   x + np.cos(yaw) * arrow_len
            ax.arrow(u_center, v_center,
                     arrow_end_u - u_center, arrow_end_v - v_center,
                     head_width=0.5, head_length=0.8,
                     fc=BEV_ARROW_COLOR, ec=BEV_ARROW_COLOR, alpha=0.9)

    def bev(self):
        """
        生成 BEV 2D 框可视化视频, 并与 6 相机拼接视频左右组合为最终输出.
        """
        bev_frames_dir = self.output_dir / 'bev_frames'
        bev_frames_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        frame_paths = []

        for ts in self.timestamp_folders:
            label_path = self.label_dir / f"{ts}.json"
            labels = []
            if label_path.exists():
                with open(label_path) as f:
                    labels = json.load(f)

            self._draw_bev_frame(ax, labels)

            frame_path = bev_frames_dir / f"{ts}.png"
            fig.savefig(str(frame_path), dpi=80, facecolor='black', pad_inches=0)
            frame_paths.append(frame_path)

        plt.close(fig)

        if not frame_paths:
            print("[WARN] BEV: 没有生成任何帧")

        # 生成 BEV 视频
        return self._images_to_video(
            images=sorted(bev_frames_dir.iterdir()),
            output=self.output_dir / 'bev_video.mp4',
            fps=10.0,
        )

def main():
    parser = argparse.ArgumentParser(description='将3D标注框反投影到相机图像上')
    parser.add_argument('--data_dir', type=str,
                        default='/home/chery/Projects/Data/20260526_145008_QCJPSD851972/20260526_145008_QCJPSD851972_3096_3121/',
                        help='解析好的数据文件夹')
    parser.add_argument('--output_folder', type=str, default='visualization',
                        help='输出目录 (默认: data_dir/visualization)')

    args = parser.parse_args()

    projector = BboxProjector(data_dir=args.data_dir, 
                              output_folder=args.output_folder)

    projector.load()

    projector.project()

    bev_path = projector.bev()

    projector.visualize(bev_video_path=bev_path)


if __name__ == '__main__':
    main()

