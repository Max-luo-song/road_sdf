### function：把当前omnire输出的结果格式修正为oriny要求的格式
### 直接使用源数据的pcd文件
### 源数据格式：    images_source = "/data4/gls/code/chery_scene_reconstruction/output/qcraft_20250702_133223_Q2517/20251007_lidar+cam0_1_2_3_4_5_6_8_9_10_12/videos/full_set_40000_rgbs"
import os
import shutil
import glob
from pathlib import Path
import numpy as np
import time

# 定义相机映射字典
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

# 创建反向映射字典
CAMERA_REVERSE_DICT = {v: k for k, v in QCRAFT_CAMERA_DICT.items()}

def get_timestamp_from_raw(raw_folder):
    """
    从raw文件夹中提取时间戳信息
    返回结构: {timestamp_folder: {camera_index: timestamp_suffix}}
    """
    timestamp_map = {}
    
    # 查找所有时间戳文件夹
    timestamp_folders = [f for f in os.listdir(raw_folder) 
                        if os.path.isdir(os.path.join(raw_folder, f)) and '.' in f]
    
    for folder in timestamp_folders:
        timestamp_map[folder] = {}
        
        # 查找该文件夹中的所有图像文件
        image_files = glob.glob(os.path.join(raw_folder, folder, "*.jpg"))
        
        for img_file in image_files:
            filename = os.path.basename(img_file)
            
            # 解析文件名格式: 20250702_133223_Q2517-CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99-1751434405.4459.jpg
            parts = filename.split('-')
            if len(parts) >= 3:
                camera_name = parts[1]  # CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99
                timestamp_suffix = parts[2].replace('.jpg', '')  # 1751434405.4459
                
                # 将相机名称映射到索引
                if camera_name in QCRAFT_CAMERA_DICT:
                    camera_index = QCRAFT_CAMERA_DICT[camera_name]
                    timestamp_map[folder][camera_index] = timestamp_suffix
    
    return timestamp_map

def get_pcd_timestamp_from_raw(raw_folder):
    """
    从raw文件夹中提取PCD文件的时间戳信息
    """
    pcd_timestamp_map = {}
    
    # 查找所有时间戳文件夹
    timestamp_folders = [f for f in os.listdir(raw_folder) 
                        if os.path.isdir(os.path.join(raw_folder, f)) and '.' in f]
    
    for folder in timestamp_folders:
        # 查找该文件夹中的PCD文件
        pcd_files = glob.glob(os.path.join(raw_folder, folder, "*.pcd"))
        if pcd_files:
            # 取第一个PCD文件来解析时间戳格式
            sample_file = os.path.basename(pcd_files[0])
            # 解析PCD文件名格式: 20250702_133223_Q2517-LDR_FRONT-1751434405.5515-ego.pcd
            parts = sample_file.split('-')
            if len(parts) >= 4:
                timestamp_suffix = parts[2]  # 1751434405.5515
                pcd_timestamp_map[folder] = timestamp_suffix
    
    return pcd_timestamp_map

def process_data():
    # 定义路径
    images_source = "/data4/gls/code/chery_scene_reconstruction/output/qcraft_20250702_133223_Q2517/20251007_lidar+cam0_1_2_3_4_5_6_8_9_10_12/videos/full_set_40000_rgbs"
    raw_base = "/data4/gls/code/chery_scene_reconstruction/data/qcraft/raw/20250702_133223_Q2517/20250702_133223_Q2517_60_75"
    pcd_source_base = "/data4/gls/code/chery_scene_reconstruction/data/qcraft/raw/20250702_133223_Q2517/20250702_133223_Q2517_60_75"
    output_base = "/data4/gls/code/chery_scene_reconstruction/data/qcraft/oriny/20250702_133223_Q2517/20250702_133223_Q2517_60_75"
    
    # 创建输出目录
    os.makedirs(output_base, exist_ok=True)
    
    # 从raw文件夹获取时间戳映射
    print("正在从raw文件夹提取时间戳信息...")
    image_timestamp_map = get_timestamp_from_raw(raw_base)
    pcd_timestamp_map = get_pcd_timestamp_from_raw(raw_base)
    
    print(f"找到 {len(image_timestamp_map)} 个时间戳文件夹的图像映射")
    print(f"找到 {len(pcd_timestamp_map)} 个时间戳文件夹的PCD映射")
    
    # 步骤1: 处理图像文件
    print("正在处理图像文件...")
    image_files = sorted(glob.glob(os.path.join(images_source, "*.png")))
    
    # 按时间戳分组图像
    timestamp_images = {}
    for img_file in image_files:
        filename = os.path.basename(img_file)
        # 解析文件名格式: 000_000.png
        parts = filename.split('_')
        if len(parts) >= 2:
            timestamp_key = parts[0]  # 如 '000', '001' 等
            camera_index = int(parts[1].split('.')[0])  # 如 0, 1, 2 等
            
            if timestamp_key not in timestamp_images:
                timestamp_images[timestamp_key] = {}
            timestamp_images[timestamp_key][camera_index] = img_file
    
    print(f"找到 {len(timestamp_images)} 个时间戳的图像数据")
    
    # 为每个时间戳创建文件夹并复制/重命名图像
    for timestamp_key, camera_dict in timestamp_images.items():
        # 查找对应的时间戳文件夹名称
        # 这里需要建立 processed 中的 000, 001 等与 raw 中时间戳文件夹的映射
        # 由于没有直接的映射关系，我们按顺序匹配
        timestamp_folders = sorted(list(image_timestamp_map.keys()))
        
        if len(timestamp_folders) > int(timestamp_key):
            raw_timestamp_folder = timestamp_folders[int(timestamp_key)]
        else:
            print(f"警告: 时间戳 {timestamp_key} 超出范围，跳过")
            continue
            
        if raw_timestamp_folder not in image_timestamp_map:
            print(f"警告: 时间戳文件夹 {raw_timestamp_folder} 未在映射中找到，跳过")
            continue
            
        output_timestamp_dir = os.path.join(output_base, raw_timestamp_folder)
        os.makedirs(output_timestamp_dir, exist_ok=True)
        
        # 复制每个视角的图像
        for camera_idx, source_img in camera_dict.items():
            if camera_idx in image_timestamp_map[raw_timestamp_folder]:
                timestamp_suffix = image_timestamp_map[raw_timestamp_folder][camera_idx]
                camera_name = CAMERA_REVERSE_DICT[camera_idx]
                # 将PNG转换为JPG格式
                new_filename = f"20250702_133223_Q2517-{camera_name}-{timestamp_suffix}.jpg"
                dest_path = os.path.join(output_timestamp_dir, new_filename)
                
                # 复制文件（PNG到JPG，如果需要格式转换可以在这里添加）
                shutil.copy2(source_img, dest_path)
                print(f"复制图像: {source_img} -> {new_filename}")
            else:
                print(f"警告: 时间戳文件夹 {raw_timestamp_folder} 中未找到相机 {camera_idx} 的时间戳信息")
        time.sleep(10000)
        # 步骤2: 复制PCD文件
        pcd_source_dir = os.path.join(pcd_source_base, raw_timestamp_folder)
        if os.path.exists(pcd_source_dir):
            # 查找PCD文件
            pcd_files = glob.glob(os.path.join(pcd_source_dir, "*.pcd"))
            for pcd_file in pcd_files:
                pcd_filename = os.path.basename(pcd_file)
                pcd_dest = os.path.join(output_timestamp_dir, pcd_filename)
                shutil.copy2(pcd_file, pcd_dest)
                print(f"复制PCD文件: {pcd_filename}")
        else:
            print(f"警告: PCD源文件夹不存在 {pcd_source_dir}")
        
        # 步骤3: 从raw文件夹复制txt文件
        raw_timestamp_dir = os.path.join(raw_base, raw_timestamp_folder)
        if os.path.exists(raw_timestamp_dir):
            # 复制所有txt文件
            for txt_file in glob.glob(os.path.join(raw_timestamp_dir, "*.txt")):
                shutil.copy2(txt_file, output_timestamp_dir)
                print(f"复制txt文件: {os.path.basename(txt_file)}")
    
    # 步骤4: 从raw文件夹复制label文件夹和其他文件
    print("正在复制label文件夹和其他文件...")
    
    # 复制label文件夹
    label_source = os.path.join(raw_base, "label")
    label_dest = os.path.join(output_base, "label")
    if os.path.exists(label_source):
        if os.path.exists(label_dest):
            shutil.rmtree(label_dest)
        shutil.copytree(label_source, label_dest)
        print("复制label文件夹完成")
    
    # 复制json和txt文件
    for pattern in ["*.json", "*.txt"]:
        for file_path in glob.glob(os.path.join(raw_base, pattern)):
            if os.path.isfile(file_path) and os.path.basename(file_path) not in ['README.txt']:  # 避免复制不必要的文件
                shutil.copy2(file_path, output_base)
                print(f"复制文件: {os.path.basename(file_path)}")
    
    print("数据处理完成!")

if __name__ == "__main__":
    process_data()