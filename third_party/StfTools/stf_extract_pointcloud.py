#!/usr/bin/env python3
"""
STF点云提取 - 自动从 lite_run_info.pb.bin 读取内外参
支持使用 scan.pose 做运动补偿，将同一帧所有 scan 对齐到 mid_pose。
"""

import sys
import os
import struct
import argparse
import array
import math
import time
from pathlib import Path
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stf_reader import StfReader

try:
    import cramjam
    HAS_CRAMJAM = True
except ImportError:
    HAS_CRAMJAM = False

DEG_TO_RAD = math.pi / 180.0
ATX_METERS_PER_TICK = 0.005

# ---- 预计算旋转矩阵工具函数 ----
_IDENT_ROT = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _make_rot_matrix(yaw: float, pitch: float, roll: float) -> tuple:
    """由 yaw/pitch/roll 构建 3×3 旋转矩阵，返回 9 元组（行主序）。"""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    return (
        cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
        -sp,     cp * sr,                cp * cr,
    )


def _make_inv_rot_matrix(yaw: float, pitch: float, roll: float) -> tuple:
    """旋转矩阵的逆（=转置），返回 9 元组。"""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    return (
        cy * cp, sy * cp, -sp,
        cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr,
        cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr,
    )


def _mat_mul(a: tuple, b: tuple) -> tuple:
    """3×3 矩阵乘法（9 元组 × 9 元组）。"""
    return (
        a[0]*b[0] + a[1]*b[3] + a[2]*b[6],
        a[0]*b[1] + a[1]*b[4] + a[2]*b[7],
        a[0]*b[2] + a[1]*b[5] + a[2]*b[8],
        a[3]*b[0] + a[4]*b[3] + a[5]*b[6],
        a[3]*b[1] + a[4]*b[4] + a[5]*b[7],
        a[3]*b[2] + a[4]*b[5] + a[5]*b[8],
        a[6]*b[0] + a[7]*b[3] + a[8]*b[6],
        a[6]*b[1] + a[7]*b[4] + a[8]*b[7],
        a[6]*b[2] + a[7]*b[5] + a[8]*b[8],
    )


def _build_scan_transform(
    scan_pose: "VehiclePose",
    mid_pose: "VehiclePose",
    ext_rot: Optional[tuple],
    ext_inv_rot: Optional[tuple],
    extrinsics: Optional["LidarExtrinsics"],
    motion_compensate: bool,
    apply_extrinsics: bool,
):
    """预计算一个 scan 的合并变换矩阵，避免逐点重复三角函数。

    将运动补偿 + 外参的多次旋转合并为单次 R@(x,y,z) + t，返回 (R_comb, t_comb)，
    均为 9/3 元组。无需变换时返回 (None, None)。
    
    lidar坐标系 -->> 车辆坐标系 -->> 世界坐标系 -->> mid_scan车辆坐标系 
    """
    if not (motion_compensate or (apply_extrinsics and ext_rot is not None)):
        return None, None

    has_ext = ext_rot is not None
    scan_rot = _make_rot_matrix(scan_pose.yaw, scan_pose.pitch, scan_pose.roll)
    stx, sty, stz = scan_pose.x, scan_pose.y, scan_pose.z

    if motion_compensate:
        mid_inv_rot = _make_inv_rot_matrix(mid_pose.yaw, mid_pose.pitch, mid_pose.roll)
        mtx, mty, mtz = mid_pose.x, mid_pose.y, mid_pose.z

    # ---- 根据 flags 组合变换链 ----
    if motion_compensate and apply_extrinsics and has_ext:
        # result = inv(T_mid) @ T_scan @ T_ext @ p
        R_tmp = _mat_mul(mid_inv_rot, scan_rot)
        R = _mat_mul(R_tmp, ext_rot)
        t = _vec_add3(
            _mat_vec_mul(R_tmp, extrinsics.x, extrinsics.y, extrinsics.z),
            _mat_vec_mul(mid_inv_rot, stx, sty, stz),
            _mat_vec_mul(mid_inv_rot, -mtx, -mty, -mtz),
        )
    elif motion_compensate and has_ext:
        # result = inv(T_ext) @ inv(T_mid) @ T_scan @ T_ext @ p
        AB = _mat_mul(ext_inv_rot, mid_inv_rot)
        AB_scan = _mat_mul(AB, scan_rot)
        R = _mat_mul(AB_scan, ext_rot)
        ext_x, ext_y, ext_z = extrinsics.x, extrinsics.y, extrinsics.z
        t = _vec_add4(
            _mat_vec_mul(AB_scan, ext_x, ext_y, ext_z),
            _mat_vec_mul(AB, stx, sty, stz),
            _mat_vec_mul(AB, -mtx, -mty, -mtz),
            _mat_vec_mul(ext_inv_rot, -ext_x, -ext_y, -ext_z),
        )
    elif motion_compensate:
        # result = inv(T_mid) @ T_scan @ p  (无外参)
        R = _mat_mul(mid_inv_rot, scan_rot)
        t = _vec_add2(
            _mat_vec_mul(mid_inv_rot, stx, sty, stz),
            _mat_vec_mul(mid_inv_rot, -mtx, -mty, -mtz),
        )
    elif apply_extrinsics and has_ext:
        # result = T_ext @ p
        R = ext_rot
        t = (extrinsics.x, extrinsics.y, extrinsics.z)

    return R, t


def _mat_vec_mul(r: tuple, x: float, y: float, z: float) -> tuple:
    """3×3 矩阵 × 向量，返回 3 元组。"""
    return (
        r[0] * x + r[1] * y + r[2] * z,
        r[3] * x + r[4] * y + r[5] * z,
        r[6] * x + r[7] * y + r[8] * z,
    )


def _vec_add2(a: tuple, b: tuple) -> tuple:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec_add3(a: tuple, b: tuple, c: tuple) -> tuple:
    return (a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2])


def _vec_add4(a: tuple, b: tuple, c: tuple, d: tuple) -> tuple:
    return (a[0] + b[0] + c[0] + d[0], a[1] + b[1] + c[1] + d[1], a[2] + b[2] + c[2] + d[2])


# Same as BuildPandarAT128ToPandarNonlinearMappingTableInFloat().
# LIDAR_PANDAR_ATX uses LidarVendor::kPandarSolidAT128.
PANDAR_ATX_TO_PANDAR_NONLINEAR = [
    0.0, 36.4555, 46.3156, 55.8149, 64.8608, 73.3606, 81.2289,
    88.4991, 95.3026, 101.7735, 108.0459, 114.2540, 120.4670,
    126.5875, 132.4914, 138.0547, 143.1533, 147.6802, 151.6887,
    155.3209, 158.7201, 162.0295, 165.3911, 168.8539, 172.3087,
    175.6306, 178.6948, 181.3767, 183.5722, 185.3001, 186.6241,
    187.6079, 188.3150, 188.8088, 189.1386, 189.3382, 189.4402,
    189.4773, 189.4825, 189.4825, 189.4825, 189.4825, 189.4825,
    189.4825, 189.4825, 189.4825, 189.4825, 189.4825, 189.4825,
    189.4825, 189.4984, 189.5840, 189.7989, 190.2026, 190.8549,
    191.8267, 193.2907, 195.4726, 198.5986, 202.8949, 208.5831,
    215.5753, 223.2854, 231.0823, 238.3348, 244.4119, 248.7984,
    251.6200, 253.2218, 253.9491, 254.1474,
] + [255.0] * (256 - 71)


def atx_normalized_intensity(raw_intensity: int) -> float:
    raw = max(0, min(255, int(raw_intensity)))
    # C++: uint16_t(table[i] * 256) >> 8, equivalent to floor(table[i]).
    nonlinear = int(PANDAR_ATX_TO_PANDAR_NONLINEAR[raw])
    return nonlinear / 255.0

class LidarIntrinsics:
    def __init__(self):
        self.elevations: List[float] = []
        self.even_azimuth_offsets: List[float] = []
        self.odd_azimuth_offsets: List[float] = []

    @property
    def num_beams(self) -> int:
        return len(self.elevations)


class LidarExtrinsics:
    def __init__(self, x=0.0, y=0.0, z=0.0, yaw=0.0, pitch=0.0, roll=0.0):
        self.x, self.y, self.z = x, y, z
        self.yaw, self.pitch, self.roll = yaw, pitch, roll


class VehiclePose:
    def __init__(self, x=0.0, y=0.0, z=0.0, yaw=0.0, pitch=0.0, roll=0.0):
        self.x, self.y, self.z = x, y, z
        self.yaw, self.pitch, self.roll = yaw, pitch, roll


def load_lidar_params(run_dir: str) -> Tuple[Optional[LidarIntrinsics], Optional[LidarExtrinsics]]:
    """从 lite_run_info.pb.bin 加载激光雷达参数"""
    bin_path = os.path.join(run_dir, "lite_run_info.pb.bin")
    if not os.path.exists(bin_path):
        return None, None

    try:
        from proto_from_descriptors import get_message_class
        LiteRun = get_message_class("qcraft.LiteRun")
        if LiteRun is None:
            return None, None

        with open(bin_path, "rb") as f:
            lite_run = LiteRun()
            lite_run.ParseFromString(f.read())

        car_info = lite_run.car_info
        if not car_info.HasField("run_params") or not car_info.run_params.HasField("v2_vehicle_params"):
            return None, None

        v2 = car_info.run_params.v2_vehicle_params
        if len(v2.lidars) == 0:
            return None, None

        lidar = v2.lidars[0]
        intrinsics, extrinsics = None, None

        if lidar.HasField("inherent") and lidar.inherent.HasField("intrinsics"):
            intr = lidar.inherent.intrinsics
            intrinsics = LidarIntrinsics()
            intrinsics.elevations = list(intr.elevations)
            intrinsics.even_azimuth_offsets = list(intr.even_azimuth_offsets)
            intrinsics.odd_azimuth_offsets = list(intr.odd_azimuth_offsets)

        if lidar.HasField("installation") and lidar.installation.HasField("extrinsics"):
            ext = lidar.installation.extrinsics
            extrinsics = LidarExtrinsics(ext.x, ext.y, ext.z, ext.yaw, ext.pitch, ext.roll)

        return intrinsics, extrinsics
    except Exception as e:
        print(f"加载参数失败: {e}", file=sys.stderr)
        return None, None


class PointCloudExtractor:
    def __init__(
        self,
        output_dir: str,
        binary_mode: bool = False,
        verbose: bool = False,
        apply_extrinsics: bool = False,
        motion_compensate: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_count = 0
        self.binary_mode = binary_mode
        self.verbose = verbose
        self.apply_extrinsics = apply_extrinsics
        self.motion_compensate = motion_compensate
        self.intrinsics: Optional[LidarIntrinsics] = None
        self.extrinsics: Optional[LidarExtrinsics] = None
        self._ext_rot: Optional[tuple] = None        # 预计算外参旋转矩阵
        self._ext_inv_rot: Optional[tuple] = None     # 预计算外参逆旋转矩阵

    def load_params(self, run_dir: str):
        self.intrinsics, self.extrinsics = load_lidar_params(run_dir)
        if self.intrinsics is None or self.intrinsics.num_beams == 0:
            print("错误: 无法加载激光雷达内参", file=sys.stderr)
            sys.exit(1)

        print(f"内参: {self.intrinsics.num_beams} beams")
        
        if self.extrinsics:
            print(f"外参: x={self.extrinsics.x:.2f} y={self.extrinsics.y:.2f} z={self.extrinsics.z:.2f}")
            # 预计算外参旋转矩阵（所有帧共用）
            self._ext_rot = _make_rot_matrix(self.extrinsics.yaw, self.extrinsics.pitch, self.extrinsics.roll)
            self._ext_inv_rot = _make_inv_rot_matrix(self.extrinsics.yaw, self.extrinsics.pitch, self.extrinsics.roll)
        elif self.motion_compensate:
            print("警告: 未加载外参，运动补偿将使用单位 lidar->vehicle 外参", file=sys.stderr)

    def extract_from_stf(self, stf_file: str, limit: int = None):
        reader = StfReader(stf_file)
        if not reader.open():
            print(f"无法打开: {stf_file}", file=sys.stderr)
            return

        messages = reader.get_messages()
        if not messages:
            print("无消息", file=sys.stderr)
            return

        print(f"处理 {len(messages)} 条消息...")

        for msg in messages:
            if limit and self.extracted_count >= limit:
                break

            data = msg.get("data", b"")
            timestamp = msg.get("timestamp", 0)

            points = self._parse_pointcloud(data)
            if points and len(points) > 100:
                pcd_filename = f"pointcloud-LDR_FRONT-{timestamp}.pcd"
                self._save_pcd(points, self.output_dir / pcd_filename)
                self.extracted_count += 1

                if self.verbose:
                    print(f"[{self.extracted_count}] {len(points)} pts -> {pcd_filename}")
                elif self.extracted_count % 10 == 0:
                    print(f"已提取: {self.extracted_count}", end="\r")

        print(f"\n完成: {self.extracted_count} 个文件 -> {self.output_dir}")

    def _parse_pointcloud(self, data: bytes) -> Optional[List[Tuple[float, float, float, float]]]:
        if len(data) < 100:
            if self.verbose:
                print(f"  [DEBUG] data too short: {len(data)} bytes")
            return None

        decompressed = self._decompress_snappy(data)
        if decompressed and len(decompressed) > len(data):
            data = decompressed
            if self.verbose:
                print(f"  [DEBUG] snappy decompressed: {len(data)} bytes")

        if not self._is_spin_format(data):
            if self.verbose:
                # 打印前几个字节帮助判断格式
                num_scans = struct.unpack("<i", data[0:4])[0] if len(data) >= 4 else 0
                ts = struct.unpack("<d", data[4:12])[0] if len(data) >= 12 else 0.0
                print(f"  [DEBUG] not spin format: num_scans={num_scans}, ts={ts}, len={len(data)}")
            return None

        result = self._parse_spin_format(data)
        if result is None or len(result) <= 100:
            if self.verbose:
                print(f"  [DEBUG] _parse_spin_format returned {len(result) if result else 0} points (need >100)")
        return result

    def _is_spin_format(self, data: bytes) -> bool:
        if len(data) < 20:
            return False
        try:
            num_scans = struct.unpack("<i", data[0:4])[0]
            timestamp = struct.unpack("<d", data[4:12])[0]
            return 50 < num_scans < 5000 and 1e9 < timestamp < 2e9
        except Exception:
            return False

    def _parse_spin_format(self, data: bytes) -> Optional[List[Tuple[float, float, float, float]]]:
        if len(data) < 50:
            return None

        try:
            num_scans = struct.unpack("<i", data[0:4])[0]
            if num_scans <= 0 or num_scans > 5000:
                return None

            ref_timestamp = struct.unpack("<d", data[4:12])[0]
            valid_scans = self._find_valid_scan_headers(data, ref_timestamp)
            if len(valid_scans) < 10:
                return None

            mid_pose = valid_scans[len(valid_scans) // 2][3]
            num_beams = self.intrinsics.num_beams
            elevations = self.intrinsics.elevations
            even_az = self.intrinsics.even_azimuth_offsets
            odd_az = self.intrinsics.odd_azimuth_offsets

            points = []
            for scan_offset, _, azimuth_deg, scan_pose in valid_scans:
                # Scan serialized layout:
                # timestamp(8) + azimuth(8) + pose(24) + extension(4) = 44 bytes.
                ext_offset = scan_offset + 40
                extension = struct.unpack("<I", data[ext_offset:ext_offset + 4])[0]
                parity_flag = (extension >> 31) & 0x1
                az_offsets = odd_az if parity_flag > 0 else even_az

                # 预计算本 scan 的合并变换 (R|t)，避免逐点算三角函数
                R_comb, t_comb = _build_scan_transform(
                    scan_pose, mid_pose,
                    self._ext_rot, self._ext_inv_rot, self.extrinsics,
                    self.motion_compensate, self.apply_extrinsics,
                )

                pos = scan_offset + 44

                for beam_idx in range(num_beams):
                    if pos >= len(data):
                        break

                    nr = data[pos]
                    pos += 1

                    if 0 < nr < 10:
                        if pos + 4 > len(data):
                            break

                        range_raw = struct.unpack("<H", data[pos:pos + 2])[0]
                        raw_intensity = data[pos + 2]
                        intensity = atx_normalized_intensity(raw_intensity)
                        pos += 4 + (nr - 1) * 4

                        if 100 < range_raw < 50000 and beam_idx < len(elevations):
                            el_deg = elevations[beam_idx]
                            az_offset = az_offsets[beam_idx] if az_offsets and beam_idx < len(az_offsets) else 0.0

                            az_rad = (azimuth_deg + az_offset) * DEG_TO_RAD
                            el_rad = el_deg * DEG_TO_RAD
                            r = range_raw * ATX_METERS_PER_TICK

                            cos_el = math.cos(el_rad)

                            # 原始点在 lidar frame。
                            x = r * cos_el * math.sin(az_rad)
                            y = r * cos_el * math.cos(az_rad)
                            z = r * math.sin(el_rad)

                            if R_comb is not None:
                                # 合并变换：R_comb @ (x,y,z) + t_comb
                                rx = R_comb[0]*x + R_comb[1]*y + R_comb[2]*z + t_comb[0]
                                ry = R_comb[3]*x + R_comb[4]*y + R_comb[5]*z + t_comb[1]
                                rz = R_comb[6]*x + R_comb[7]*y + R_comb[8]*z + t_comb[2]
                                x, y, z = rx, ry, rz

                            if self._is_valid_point(x, y, z, r):
                                points.append((x, y, z, float(intensity)))
                    elif nr >= 10:
                        break

            return points if len(points) > 100 else None
        except Exception as e:
            if self.verbose:
                print(f"解析 spin 失败: {e}", file=sys.stderr)
            return None
    
    def _motion_compensate_to_mid_pose(
        self,
        x: float,
        y: float,
        z: float,
        scan_pose: VehiclePose,
        mid_pose: VehiclePose,
    ) -> Tuple[float, float, float]:
        # p_world = T_scan_pose * T_lidar_to_vehicle * p_lidar
        if self.extrinsics:
            vx, vy, vz = self._apply_extrinsics(x, y, z)
        else:
            vx, vy, vz = x, y, z

        wx, wy, wz = self._pose_transform_point(scan_pose, vx, vy, vz)

        # p_mid_vehicle = inv(T_mid_pose) * p_world
        mx, my, mz = self._pose_inverse_transform_point(mid_pose, wx, wy, wz)

        # p_mid_lidar = inv(T_lidar_to_vehicle) * p_mid_vehicle
        if self.extrinsics:
            return self._apply_inverse_extrinsics(mx, my, mz)

        return mx, my, mz
    
    def _read_pose(self, data: bytes, offset: int) -> VehiclePose:
        x, y, z, yaw, pitch, roll = struct.unpack("<ffffff", data[offset:offset + 24])
        return VehiclePose(x, y, z, yaw, pitch, roll)

    def _pose_transform_point(self, pose: VehiclePose, x: float, y: float, z: float) -> Tuple[float, float, float]:
        rx, ry, rz = self._rotate_ypr(pose.yaw, pose.pitch, pose.roll, x, y, z)
        return rx + pose.x, ry + pose.y, rz + pose.z

    def _pose_inverse_transform_point(self, pose: VehiclePose, x: float, y: float, z: float) -> Tuple[float, float, float]:
        return self._inverse_transform_ypr_xyz(pose.x, pose.y, pose.z, pose.yaw, pose.pitch, pose.roll, x, y, z)

    def _apply_extrinsics(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        ext = self.extrinsics
        rx, ry, rz = self._rotate_ypr(ext.yaw, ext.pitch, ext.roll, x, y, z)
        return rx + ext.x, ry + ext.y, rz + ext.z

    def _apply_inverse_extrinsics(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        ext = self.extrinsics
        return self._inverse_transform_ypr_xyz(ext.x, ext.y, ext.z, ext.yaw, ext.pitch, ext.roll, x, y, z)

    def _rotate_ypr(self, yaw: float, pitch: float, roll: float, x: float, y: float, z: float) -> Tuple[float, float, float]:
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)

        x_r = cy * cp * x + (cy * sp * sr - sy * cr) * y + (cy * sp * cr + sy * sr) * z
        y_r = sy * cp * x + (sy * sp * sr + cy * cr) * y + (sy * sp * cr - cy * sr) * z
        z_r = -sp * x + cp * sr * y + cp * cr * z
        return x_r, y_r, z_r

    def _inverse_transform_ypr_xyz(
        self,
        tx: float,
        ty: float,
        tz: float,
        yaw: float,
        pitch: float,
        roll: float,
        x: float,
        y: float,
        z: float,
    ) -> Tuple[float, float, float]:
        dx, dy, dz = x - tx, y - ty, z - tz

        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)

        r00 = cy * cp
        r01 = cy * sp * sr - sy * cr
        r02 = cy * sp * cr + sy * sr
        r10 = sy * cp
        r11 = sy * sp * sr + cy * cr
        r12 = sy * sp * cr - cy * sr
        r20 = -sp
        r21 = cp * sr
        r22 = cp * cr

        return (
            r00 * dx + r10 * dy + r20 * dz,
            r01 * dx + r11 * dy + r21 * dz,
            r02 * dx + r12 * dy + r22 * dz,
        )

    def _find_valid_scan_headers(self, data: bytes, ref_ts: float) -> List[Tuple[int, float, float, VehiclePose]]:
        """依次解析 scan header，通过遍历 nr 计算出每个 scan 的精确字节数来跳跃。

        Scan 序列化布局: header(44) + 逐 beam 点数据
        每 beam: nr(1) + [0 或 4+(nr-1)*4 字节]; nr>=10 表示 scan 结束。
        """
        valid = []
        num_beams = self.intrinsics.num_beams
        data_len = len(data)

        # 第一个 scan header 在 offset 12: 跳过 num_scans(int32) + ref_timestamp(double)
        offset = 12

        while offset < data_len - 44:
            # ----- 校验 scan header -----
            try:
                ts = struct.unpack("<d", data[offset:offset + 8])[0]
                az = struct.unpack("<d", data[offset + 8:offset + 16])[0]
            except Exception:
                break

            if abs(ts - ref_ts) < 1 and -180 < az < 360:
                pose = self._read_pose(data, offset + 16)
                valid.append((offset, ts, az, pose))

                # ----- 跳过点数据，找到下一个 scan header -----
                scan_pos = offset + 44
                for _ in range(num_beams):
                    if scan_pos >= data_len:
                        break
                    nr = data[scan_pos]
                    scan_pos += 1
                    if 0 < nr < 10:
                        if scan_pos + 4 > data_len:
                            break
                        scan_pos += 4 + (nr - 1) * 4
                    elif nr >= 10:
                        break
                    # nr == 0: 已消费 1 字节，无额外数据

                offset = scan_pos  # 直接跳到下一个 scan header
            else:
                if valid:
                    # 已找到过有效 scan，后续 header 不匹配 → 结束
                    break
                # 首个 scan 尚未找到：尝试下一个字节（对齐容错）
                offset += 1

        return valid
    
    def _decompress_snappy(self, data: bytes) -> Optional[bytes]:
        if HAS_CRAMJAM:
            try:
                return bytes(cramjam.snappy.decompress_raw(data))
            except Exception:
                try:
                    return bytes(cramjam.snappy.decompress(data))
                except Exception:
                    try:
                        return bytes(cramjam.zstd.decompress(data))
                    except Exception:
                        pass
        return self._decompress_snappy_manual(data)


    def _decompress_snappy_manual(self, data: bytes) -> Optional[bytes]:
        try:
            pos = 0
            length, pos = self._read_varint(data, pos)
            if length <= 0 or length > 100_000_000:
                return None

            output = bytearray(length)
            out_pos = 0

            while pos < len(data) and out_pos < length:
                tag = data[pos]
                pos += 1
                tag_type = tag & 0x03

                if tag_type == 0:
                    lit_len = tag >> 2
                    if lit_len < 60:
                        lit_len += 1
                    else:
                        extra = lit_len - 59
                        if pos + extra > len(data):
                            break
                        lit_len = sum(data[pos + i] << (i * 8) for i in range(extra)) + 1
                        pos += extra

                    if pos + lit_len > len(data) or out_pos + lit_len > length:
                        break
                    output[out_pos:out_pos + lit_len] = data[pos:pos + lit_len]
                    pos += lit_len
                    out_pos += lit_len
                elif tag_type in (1, 2, 3):
                    if tag_type == 1:
                        copy_len = 4 + ((tag >> 2) & 0x07)
                        if pos >= len(data):
                            break
                        offset = data[pos] + ((tag & 0xE0) << 3)
                        pos += 1
                    elif tag_type == 2:
                        copy_len = 1 + (tag >> 2)
                        if pos + 1 >= len(data):
                            break
                        offset = data[pos] + (data[pos + 1] << 8)
                        pos += 2
                    else:
                        copy_len = 1 + (tag >> 2)
                        if pos + 3 >= len(data):
                            break
                        offset = sum(data[pos + i] << (i * 8) for i in range(4))
                        pos += 4

                    if offset > out_pos or offset == 0:
                        break

                    src = out_pos - offset
                    for i in range(copy_len):
                        if out_pos >= length:
                            break
                        output[out_pos] = output[src + (i % offset)]
                        out_pos += 1

            return bytes(output[:out_pos]) if out_pos >= length * 0.9 else None
        except Exception:
            return None

    def _read_varint(self, data: bytes, offset: int) -> Tuple[int, int]:
        result, shift, pos = 0, 0, offset
        while pos < len(data):
            b = data[pos]
            result |= (b & 0x7F) << shift
            pos += 1
            if b & 0x80 == 0:
                return result, pos
            shift += 7
            if shift >= 64:
                break
        return result, pos

    def _is_valid_point(self, x: float, y: float, z: float, r: float) -> bool:
        return (
            0.3 < r < 200
            and abs(x) < 200
            and abs(y) < 200
            and abs(z) < 50
            and not (math.isnan(x) or math.isnan(y) or math.isnan(z))
        )

    def _save_pcd(self, points: List[Tuple[float, float, float, float]], filepath: Path):
        n = len(points)
        header = f"""# .PCD v0.7
VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {n}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {n}
DATA {'binary' if self.binary_mode else 'ascii'}
"""
        with open(filepath, "wb" if self.binary_mode else "w") as f:
            if self.binary_mode:
                f.write(header.encode("ascii"))
                for x, y, z, i in points:
                    f.write(struct.pack("<ffff", x, y, z, i))
            else:
                f.write(header)
                for x, y, z, i in points:
                    # f.write(f"{x:.4f} {y:.4f} {z:.4f} {i:.0f}\n")
                    f.write(f"{x:.4f} {y:.4f} {z:.4f} {i:.6f}\n")

def main():
    parser = argparse.ArgumentParser(description="STF点云提取")
    parser.add_argument("run_dir", help="数据包目录")
    parser.add_argument('pcd_file', help='某个pcd part文件名')
    parser.add_argument("-o", "--output", default="pointclouds", help="输出目录")
    parser.add_argument("-l", "--limit", type=int, help="限制数量")
    parser.add_argument("-b", "--binary", action="store_true", help="Binary格式")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    parser.add_argument("--apply-extrinsics", action="store_true", help="输出 mid vehicle frame；默认输出 mid lidar frame")
    parser.add_argument("--no-motion-compensation", action="store_true", help="关闭 scan.pose 到 mid_pose 的运动补偿")
    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        print(f"错误: 目录不存在: {args.run_dir}", file=sys.stderr)
        sys.exit(1)

    stf_file = os.path.join(args.run_dir, args.pcd_file)
    if not os.path.exists(stf_file):
        print("错误: 未找到 lidar_data.stf", file=sys.stderr)
        sys.exit(1)

    extractor = PointCloudExtractor(
        args.output,
        args.binary,
        args.verbose,
        args.apply_extrinsics,
        motion_compensate=not args.no_motion_compensation,
    )
    extractor.load_params(args.run_dir)

    t_start = time.time()
    extractor.extract_from_stf(stf_file, args.limit)
    elapsed = time.time() - t_start
    print(f"总耗时: {elapsed:.2f}s")


if __name__ == "__main__":
    main()