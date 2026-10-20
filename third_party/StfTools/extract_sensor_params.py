#!/usr/bin/env python3
"""
lite_run_info.pb.bin 提取传感器内外参
"""

import sys
import os
import json
import argparse
from typing import Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============== 映射字典 ==============
LIDAR_ID_TO_NAME = {
    0: "LDR_UNKNOWN", 1: "LDR_CENTER", 2: "LDR_FRONT_BLIND", 3: "LDR_LEFT_BLIND",
    4: "LDR_RIGHT_BLIND", 5: "LDR_REAR_BLIND", 6: "LDR_FRONT_LEFT", 7: "LDR_FRONT_RIGHT",
    8: "LDR_FRONT_LEFT_BLIND", 9: "LDR_FRONT_RIGHT_BLIND", 10: "LDR_FRONT", 11: "LDR_REAR",
    12: "LDR_REAR_LEFT", 13: "LDR_REAR_RIGHT", 14: "LDR_FRONT_LEFT_GT", 15: "LDR_FRONT_RIGHT_GT",
    16: "LDR_FRONT_LEFT_BLIND_GT", 17: "LDR_FRONT_RIGHT_BLIND_GT",
}

CAMERA_ID_TO_NAME = {
    50: "CAM_PBQ_FRONT_WIDE", 52: "CAM_PBQ_FRONT_TELE", 53: "CAM_PBQ_FRONT_LEFT",
    54: "CAM_PBQ_FRONT_RIGHT", 55: "CAM_PBQ_REAR_LEFT", 56: "CAM_PBQ_REAR_RIGHT",
    57: "CAM_PBQ_REAR", 58: "CAM_PBQ_FRONT_FISHEYE", 59: "CAM_PBQ_LEFT_FISHEYE",
    60: "CAM_PBQ_RIGHT_FISHEYE", 61: "CAM_PBQ_REAR_FISHEYE", 67: "CAM_PBQ_FRONT_TELE_CROP_VIRTUAL",
    75: "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H110", 76: "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H60",
    77: "CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H30", 78: "CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99",
    79: "CAM_PBQ_FRONT_RIGHT_RESET_OPTICAL_H99", 82: "CAM_PBQ_REAR_RESET_OPTICAL_H50",
    83: "CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H99", 84: "CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H30",
    85: "CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H99", 86: "CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H30",
    97: "CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H15",
}

# ============== 辅助函数 ==============
def _load_lite_run(bin_path: str):
    """加载 lite_run.pb.bin 文件"""
    from proto_from_descriptors import get_message_class
    
    LiteRun = get_message_class("qcraft.LiteRun")
    if LiteRun is None:
        return None
    
    lite_run = LiteRun()
    with open(bin_path, 'rb') as f:
        lite_run.ParseFromString(f.read())
    return lite_run


def _get_v2_vehicle_params(lite_run):
    """获取 v2_vehicle_params"""
    car_info = lite_run.car_info
    if not car_info.HasField('run_params') or not car_info.run_params.HasField('v2_vehicle_params'):
        return None
    return car_info.run_params.v2_vehicle_params


def _extract_distort_coeffs(distort):
    """提取畸变系数"""
    return {
        "k1": distort.k1, "k2": distort.k2, "k3": distort.k3,
        "k4": distort.k4, "k5": distort.k5, "k6": distort.k6,
        "p1": distort.p1, "p2": distort.p2,
    }


def _extract_extrinsics(extrinsics):
    """提取外参"""
    return {
        "x": extrinsics.x, "y": extrinsics.y, "z": extrinsics.z,
        "roll": extrinsics.roll, "pitch": extrinsics.pitch, "yaw": extrinsics.yaw,
    }


# ============== 主要提取函数 ==============
def extract_camera_params(run_dir: str) -> Optional[Dict[str, Any]]:
    """提取相机内外参数"""
    bin_path = os.path.join(run_dir, "lite_run_info.pb.bin")
    if not os.path.exists(bin_path):
        print(f"文件不存在: {bin_path}", file=sys.stderr)
        return None

    try:
        lite_run = _load_lite_run(bin_path)
        if lite_run is None:
            return None
        
        v2 = _get_v2_vehicle_params(lite_run)
        if v2 is None or len(v2.cameras) == 0:
            return None

        cameras_info = {}
        for camera in v2.cameras:
            cam_name = CAMERA_ID_TO_NAME.get(camera.installation.camera_id)
            if cam_name is None:
                continue
                
            cameras_info[cam_name] = {
                "camera_to_vehicle_extrinsics": _extract_extrinsics(
                    camera.installation.camera_to_vehicle_extrinsics
                ),
                "intrinsics": {
                    "cx": camera.inherent.intrinsics.camera_matrix.cx,
                    "cy": camera.inherent.intrinsics.camera_matrix.cy,
                    "fx": camera.inherent.intrinsics.camera_matrix.fx,
                    "fy": camera.inherent.intrinsics.camera_matrix.fy,
                    **_extract_distort_coeffs(camera.inherent.intrinsics.distort_coeffs),
                }
            }

            # 处理虚拟相机
            if camera.installation.set_virtual_camera:
                for vt_camera in camera.installation.virtual_cameras:
                    vt_name = CAMERA_ID_TO_NAME.get(vt_camera.camera_id)
                    if vt_name is None:
                        continue
                        
                    cameras_info[vt_name] = {
                        "camera_to_vehicle_extrinsics": {
                            **_extract_extrinsics(camera.installation.camera_to_vehicle_extrinsics),
                            "yaw": vt_camera.extrinsics.yaw,
                            "pitch": vt_camera.extrinsics.pitch,
                            "roll": vt_camera.extrinsics.roll,
                        },
                        "intrinsics": {
                            "cx": vt_camera.intrinsics.cx,
                            "cy": vt_camera.intrinsics.cy,
                            "fx": vt_camera.intrinsics.fx,
                            "fy": vt_camera.intrinsics.fy,
                            **_extract_distort_coeffs(camera.inherent.intrinsics.distort_coeffs),
                        }
                    }

        return cameras_info
    
    except Exception as e:
        print(f"加载相机参数失败: {e}", file=sys.stderr)
        return None


def extract_lidar_params(run_dir: str) -> Optional[Dict[str, Any]]:
    """提取激光雷达参数"""
    bin_path = os.path.join(run_dir, "lite_run_info.pb.bin")
    if not os.path.exists(bin_path):
        print(f"文件不存在: {bin_path}", file=sys.stderr)
        return None

    try:
        lite_run = _load_lite_run(bin_path)
        if lite_run is None:
            return None
        
        v2 = _get_v2_vehicle_params(lite_run)
        if v2 is None or len(v2.lidars) == 0:  # 修复：检查 lidars
            return None

        lidar_list = []
        for lidar in v2.lidars:
            lidar_dict = {
                "model": lidar.model,
                "key": lidar.key,
                "common": {
                    "type": lidar.common.type,
                    "spinning_lidar_params": {
                        "num_beams": lidar.common.spinning_lidar_params.num_beams,
                        "num_scans_per_spin": lidar.common.spinning_lidar_params.num_scans_per_spin,
                        "azimuth_resolution": lidar.common.spinning_lidar_params.azimuth_resolution,
                    },
                    "max_num_returns": lidar.common.max_num_returns,
                    "meters_per_tick": lidar.common.meters_per_tick,
                },
                "inherent": {
                    "serial_no": lidar.inherent.serial_no,
                },
                "installation": {
                    "lidar_id": LIDAR_ID_TO_NAME.get(lidar.installation.lidar_id, "LDR_UNKNOWN"),
                    "extrinsics": {
                        **_extract_extrinsics(lidar.installation.extrinsics),
                        "calibration_time": lidar.installation.extrinsics.calibration_time,
                        "calibration_engineer": lidar.installation.extrinsics.calibration_engineer,
                        "calibration_run": lidar.installation.extrinsics.calibration_run,
                    },
                    "enabled": lidar.installation.enabled,
                    "port": lidar.installation.port,
                    "ip": lidar.installation.ip,
                    "is_multicast": lidar.installation.is_multicast,
                },
            }
            lidar_list.append(lidar_dict)

        return {"lidar_params": lidar_list}
    
    except Exception as e:
        print(f"加载激光雷达参数失败: {e}", file=sys.stderr)
        return None


# ============== 主函数 ==============
def main():
    parser = argparse.ArgumentParser(description='传感器参数提取')
    parser.add_argument('run_dir', help='数据包目录')
    parser.add_argument('-o', '--output', default='.', help='输出目录')
    parser.add_argument('--camera-only', action='store_true', help='仅提取相机参数')
    parser.add_argument('--lidar-only', action='store_true', help='仅提取激光雷达参数')

    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        print(f"错误: 目录不存在: {args.run_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # 提取相机参数
    if not args.lidar_only:
        cameras_info = extract_camera_params(args.run_dir)
        if cameras_info:
            output_path = os.path.join(args.output, 'camera_params.json')
            with open(output_path, 'w') as f:
                json.dump(cameras_info, f, indent=4)
            print(f"相机参数已保存: {output_path}")
        else:
            print("警告: 未找到相机参数")

    # 提取激光雷达参数
    if not args.camera_only:
        lidars_info = extract_lidar_params(args.run_dir)
        if lidars_info:
            output_path = os.path.join(args.output, 'data_frame_car_info.json')
            with open(output_path, 'w') as f:
                json.dump(lidars_info, f, indent=4)
            print(f"激光雷达参数已保存: {output_path}")
        else:
            print("警告: 未找到激光雷达参数")


if __name__ == '__main__':
    main()