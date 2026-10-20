import os
import re
import json
import numpy as np


QCRAFT_CAMERA_DICT = {
    "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H110": 0,
    "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H60": 1,
    "CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H30": 2,
    "CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H15": 3,
    "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_LEFT_H60": 4,
    "CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99": 5,
    "CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H99": 6,
    "CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H30": 7,
    "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_RIGHT_H60": 8,
    "CAM_PBQ_FRONT_RIGHT_RESET_OPTICAL_H99": 9,
    "CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H99": 10,
    "CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H30": 11,
    "CAM_PBQ_REAR_RESET_OPTICAL_H50": 12,
}

QCRAFT_LIDAR_DICT = {
    "LDR_UNKNOWN": 0,
    "LDR_CENTER": 1,
    "LDR_FRONT_BLIND": 2,
    "LDR_LEFT_BLIND": 3,
    "LDR_RIGHT_BLIND": 4,
    "LDR_REAR_BLIND": 5,
    "LDR_FRONT_LEFT": 6,
    "LDR_FRONT_RIGHT": 7,
    "LDR_FRONT_LEFT_BLIND": 8,
    "LDR_FRONT_RIGHT_BLIND": 9,
    "LDR_FRONT": 10,
    "LDR_REAR": 11,
    "LDR_REAR_LEFT": 12,
    "LDR_REAR_RIGHT": 13,
    "LDR_FRONT_LEFT_GT": 14,
    "LDR_FRONT_RIGHT_GT": 15,
    "LDR_FRONT_LEFT_BLIND_GT": 16,
    "LDR_FRONT_RIGHT_BLIND_GT": 17,
    "LDR_REAR_BLIND_GT": 18,
}


def find_min_frame_txt(dir_path):
    """
    在目录下查找编号最小的 txt 文件。
    文件名可以是 000.txt、001.txt、0032.txt、12.txt 等任意位数。
    返回最小编号文件的完整路径。
    """
    txt_files = [f for f in os.listdir(dir_path) if f.endswith(".txt")]
    if not txt_files:
        raise FileNotFoundError("目录中没有 txt 文件")

    # 从文件名提取数字部分，例如 '0032.txt' -> 32
    def extract_number(name):
        match = re.search(r'(\d+)\.txt$', name)
        return int(match.group(1)) if match else float('inf')

    # 找最小编号的文件
    min_file = min(txt_files, key=extract_number)

    return os.path.join(dir_path, min_file)


def load_transform_matrix(txt_path):
    """
    从 txt 文件中读取 4x4 的齐次变换矩阵。
    文件要求包含 4 行，每行 4 个浮点数。
    """
    matrix = np.loadtxt(txt_path)
    if matrix.shape != (4, 4):
        raise ValueError(f"矩阵尺寸应为 4x4，但读取到的是 {matrix.shape}")
    return matrix


def extract_lidar_extrinsics(json_path):
    """
    从包含 lidar_params -> installation -> extrinsics 的 JSON 文件中
    提取 x, y, z, yaw, pitch, roll。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 假设 lidar_params 是列表，取LDR_FRONT的 extrinsics
    if len(data["lidar_params"]) == 1:
        params = data["lidar_params"][0]["installation"]["extrinsics"]
    else:
        for i in range(len(data["lidar_params"])):
            if data["lidar_params"][i]["installation"]["lidar_id"] == "LDR_FRONT":
                params = data["lidar_params"][i]["installation"]["extrinsics"]
                break

    fields = ["x", "y", "z", "yaw", "pitch", "roll"]
    result = {key: params.get(key) for key in fields}

    return result


def euler_to_rotation_matrix(yaw, pitch, roll):
    # Z 轴旋转（yaw）
    R_z = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], 
         [np.sin(yaw), np.cos(yaw), 0], 
         [0, 0, 1]]
    )
    # Y 轴旋转（pitch）
    R_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    # X 轴旋转（roll）      
    R_x = np.array(
        [[1, 0, 0], 
         [0, np.cos(roll), -np.sin(roll)], 
         [0, np.sin(roll), np.cos(roll)]]
    )

    R = R_z @ R_y @ R_x
    return R

# 将欧拉角和位置转换为4x4变换矩阵
def pose_to_transform_matrix(x, y, z, yaw, pitch, roll):
    R = euler_to_rotation_matrix(yaw, pitch, roll)
    T = np.array([x, y, z])
    T_matrix = np.eye(4)
    T_matrix[:3, :3] = R
    T_matrix[:3, 3] = T
    return T_matrix

def extract_start_timestamp(json_path):
    """
    从 timestamps.json 文件中提取起始时间戳。
    假设 JSON 文件结构为 {
    "FRAME": {
        "000000": 1761382840.000581,
        "000001": 1761382840.10058,
        "000002": 1761382840.200584,, ...}。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    timestamps = data.get("FRAME")
    if timestamps is None:
        raise ValueError("JSON 文件中缺少 'FRAME' 字段")
    
    # start_timestamp是字典第一个键值对的值
    start_timestamp = next(iter(timestamps.values()))
    
    return start_timestamp
