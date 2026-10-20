import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

def create_mask_overlay_video():
    # 配置参数
    base_dir = "data/qcraft/processed/training/20251025_163358_QCOYSD504206_1595_1610"
    image_dir = os.path.join(base_dir, "images")
    mask_dir = os.path.join(base_dir, "road_masks")
    output_dir = "output/qcraft_20251025_163358_QCOYSD504206_1595_1610/mask_road_vis"
    
    # 相机编号
    camera_ids = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12]
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有时间戳
    timestamps = sorted(set(
        os.path.basename(f).split('_')[0] 
        for f in glob(os.path.join(image_dir, "*.png"))
    ))
    
    print(f"找到 {len(timestamps)} 个时间戳")
    
    # 为每个相机生成视频
    for cam_id in camera_ids:
        print(f"\n处理相机 {cam_id}...")
        
        # 收集该相机的所有图片路径
        image_paths = []
        mask_paths = []
        
        for timestamp in timestamps:
            img_path = os.path.join(image_dir, f"{timestamp}_{cam_id}.png")
            mask_path = os.path.join(mask_dir, f"{timestamp}_{cam_id}.png")
            
            if os.path.exists(img_path) and os.path.exists(mask_path):
                image_paths.append(img_path)
                mask_paths.append(mask_path)
        
        if not image_paths:
            print(f"相机 {cam_id} 没有找到匹配的图片，跳过")
            continue
        
        print(f"相机 {cam_id} 找到 {len(image_paths)} 张图片")
        
        # 读取第一张图片获取视频参数
        first_img = cv2.imread(image_paths[0])
        height, width = first_img.shape[:2]
        
        # 创建视频写入器
        output_path = os.path.join(output_dir, f"camera_{cam_id}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 10  # 可以根据需要调整
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 处理每一帧
        for img_path, mask_path in tqdm(zip(image_paths, mask_paths), 
                                       total=len(image_paths), 
                                       desc=f"相机 {cam_id}"):
            # 读取原始图片
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # 读取mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            
            # 确保mask尺寸与图片一致
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height))
            
            # 创建彩色mask（绿色半透明）
            colored_mask = np.zeros_like(img)
            colored_mask[mask > 127] = [0, 255, 0]  # 绿色
            
            # 叠加mask到原图
            alpha = 0.4  # 透明度，可以调整
            overlay = cv2.addWeighted(img, 1, colored_mask, alpha, 0)
            
            # 写入视频
            video_writer.write(overlay)
        
        video_writer.release()
        print(f"相机 {cam_id} 视频已保存到: {output_path}")
    
    print("\n所有视频生成完成！")

if __name__ == "__main__":
    create_mask_overlay_video()
