# 对比两次训练输出的评估结果
# 找出指标变好的情况，并展示变化最大的帧
import os
import json
import argparse
import glob
import numpy as np
import wandb
import imageio.v2 as imageio
import re
import cv2
import csv
import torch
from omegaconf import OmegaConf
from typing import Dict, List, Tuple, Optional, Set
import logging
import base64
from datetime import datetime
try:
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as OpenpyxlImage
    from openpyxl.utils import get_column_letter
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

logger = logging.getLogger()

# 全局 LPIPS 模型实例（只加载一次）
_lpips_model = None
_lpips_device = None

def get_lpips_model():
    """获取全局 LPIPS 模型实例，只加载一次"""
    global _lpips_model, _lpips_device
    if _lpips_model is None:
        try:
            import lpips
            _lpips_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _lpips_model = lpips.LPIPS(net='alex').to(_lpips_device)
            _lpips_model.eval()
            logger.info(f"LPIPS model loaded successfully on {_lpips_device}")
        except Exception as e:
            logger.warning(f"Failed to load LPIPS model: {e}")
            _lpips_model = None
    return _lpips_model

# 需要对比的17个指标
METRICS_TO_COMPARE = [
    "image_metrics/full/psnr",
    "image_metrics/full/ssim",
    "image_metrics/full/lpips",
    "image_metrics/full/psnr_no_ego",
    "image_metrics/full/ssim_no_ego",
    "image_metrics/full/lpips_no_ego",
    "image_metrics/full/psnr_with_ego",
    "image_metrics/full/ssim_with_ego",
    "image_metrics/full/lpips_with_ego",
    "image_metrics/full/occupied_psnr",
    "image_metrics/full/occupied_ssim",
    "image_metrics/full/masked_psnr",
    "image_metrics/full/masked_ssim",
    "image_metrics/full/human_psnr",
    "image_metrics/full/human_ssim",
    "image_metrics/full/vehicle_psnr",
    "image_metrics/full/vehicle_ssim",
]


def load_metrics_file(metrics_file: str) -> Dict:
    # 加载评估指标文件
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def find_latest_metrics_file(log_dir: str, prefix: str = "images_full") -> Optional[str]:
    # 查找最新的评估指标文件
    metrics_dir = os.path.join(log_dir, "metrics")
    if not os.path.exists(metrics_dir):
        # 尝试 metrics_eval
        metrics_dir = os.path.join(log_dir, "metrics_eval")
        if not os.path.exists(metrics_dir):
            return None
    
    pattern = os.path.join(metrics_dir, f"{prefix}_*.json")
    files = glob.glob(pattern)
    if not files:
        return None
    
    # 返回最新的文件（按修改时间）
    latest_file = max(files, key=os.path.getmtime)
    return latest_file


def get_camera_ids_from_config(log_dir: str) -> Optional[List[int]]:
    """
    从 config.yaml 文件中读取相机配置
    返回相机ID列表，如果无法读取则返回None
    """
    config_path = os.path.join(log_dir, "config.yaml")
    if not os.path.exists(config_path):
        return None
    
    try:
        cfg = OmegaConf.load(config_path)
        # 尝试从 data.pixel_source.cameras 获取
        if hasattr(cfg, 'data') and hasattr(cfg.data, 'pixel_source') and hasattr(cfg.data.pixel_source, 'cameras'):
            cameras = cfg.data.pixel_source.cameras
            
            # 处理 OmegaConf 的 ListConfig 类型
            if cameras is not None:
                try:
                    # 尝试直接转换为列表（支持 list, tuple, 和 OmegaConf ListConfig）
                    cam_list = list(cameras)
                    # 确保所有元素都是整数
                    return [int(cam) for cam in cam_list if cam is not None]
                except (TypeError, ValueError) as e:
                    # 如果转换失败，尝试作为字符串处理
                    if isinstance(cameras, str):
                        return [int(x.strip()) for x in cameras.split(',') if x.strip()]
                    logger.debug(f"无法解析相机配置: {e}")
    except Exception as e:
        logger.warning(f"Failed to load camera config from {config_path}: {e}")
    
    return None


def get_camera_ids_from_run_name(run_name: str) -> Optional[List[int]]:
    """
    从 run_name 中解析相机ID
    例如: "202511240615_lidar+cam0_1_2_3_4_5_7_8_10" -> [0, 1, 2, 3, 4, 5, 7, 8, 10]
    """
    # 匹配 cam0_1_2_3 或 cam0,1,2,3 或 cam0_1_2_3_4_5_7_8_10 等格式
    pattern = r'cam(\d+(?:[_,]\d+)*)'
    match = re.search(pattern, run_name)
    if match:
        cam_str = match.group(1)
        # 处理下划线和逗号分隔
        cam_ids = []
        for part in re.split(r'[_,]', cam_str):
            try:
                cam_ids.append(int(part))
            except ValueError:
                continue
        return cam_ids if cam_ids else None
    return None


def get_camera_ids(log_dir: str) -> Optional[List[int]]:
    """
    获取训练使用的相机ID列表
    优先从 config.yaml 读取，如果失败则从 run_name 解析
    """
    # 方法1: 从 config.yaml 读取
    cam_ids = get_camera_ids_from_config(log_dir)
    if cam_ids:
        return cam_ids
    
    # 方法2: 从 run_name 解析
    run_name = os.path.basename(log_dir)
    cam_ids = get_camera_ids_from_run_name(run_name)
    if cam_ids:
        return cam_ids
    
    logger.warning(f"Could not determine camera IDs from {log_dir}")
    return None


def get_common_camera_ids(cam_ids1: Optional[List[int]], cam_ids2: Optional[List[int]]) -> Set[int]:
    """
    获取两次训练共同的相机ID集合
    """
    if cam_ids1 is None or cam_ids2 is None:
        return set()
    
    set1 = set[int](cam_ids1)
    set2 = set(cam_ids2)
    common = set1 & set2
    
    return common


def compare_metrics(metrics1: Dict, metrics2: Dict) -> Dict[str, Dict]:
    # 对比两次评估的指标
    # 返回: {metric_name: {"old": value1, "new": value2, "diff": diff, "improved": bool}}
    comparison = {}
    
    for metric in METRICS_TO_COMPARE:
        if metric not in metrics1 or metric not in metrics2:
            logger.warning(f"Metric {metric} not found in one or both results")
            continue
        
        old_value = metrics1[metric]
        new_value = metrics2[metric]

        # 跳过无效值（-1 或 None）
        if old_value is None or new_value is None or old_value == -1 or new_value == -1:
            logger.debug(f"Skip metric {metric} due to invalid value (old={old_value}, new={new_value})")
            continue
        
        # 判断指标是否变好
        # 对于 PSNR，越大越好；对于 SSIM，越大越好；对于 LPIPS，越小越好
        if "lpips" in metric.lower():
            improved = new_value < old_value  # LPIPS 越小越好，数值降低是提升
            diff = old_value - new_value  # 正数表示改进（LPIPS降低）
            # 对于LPIPS，improvement_pct应该基于降低的绝对值，但保持正数表示改进
            improvement_pct = (diff / old_value * 100) if old_value != 0 else 0
        else:
            improved = new_value > old_value  # PSNR/SSIM 越大越好
            diff = new_value - old_value  # 正数表示改进
            improvement_pct = (diff / old_value * 100) if old_value != 0 else 0
        
        comparison[metric] = {
            "old": old_value,
            "new": new_value,
            "diff": diff,
            "improved": improved,
            "improvement_pct": improvement_pct
        }
    
    return comparison


def extract_frame_cam_from_filename(filename: str) -> Optional[Tuple[int, int]]:
    """
    从文件名中提取帧ID和相机ID
    支持格式:
    1. {frame:03d}_{cam_id:03d}.png (例如: 000_003.png)
    2. frame_{timestamp:06d}_cam{cam_id}_{key}.png (例如: frame_000000_cam3_rgbs.png)
    
    返回: (frame_id, cam_id) 或 None
    """
    # 格式1: 000_003.png
    match1 = re.match(r'(\d+)_(\d+)\.png', filename)
    if match1:
        try:
            frame_id = int(match1.group(1))
            cam_id = int(match1.group(2))
            return (frame_id, cam_id)
        except:
            pass
    
    # 格式2: frame_000000_cam3_rgbs.png
    match2 = re.search(r'frame_(\d+).*cam(\d+)', filename)
    if match2:
        try:
            frame_id = int(match2.group(1))
            cam_id = int(match2.group(2))
            return (frame_id, cam_id)
        except:
            pass
    
    return None


# def compute_overall_score(metrics: Dict, metrics_weights: Optional[Dict[str, float]] = None) -> float:
#     """
#     根据权重计算评估结果的综合分数
    
#     Args:
#         metrics: 指标字典，格式为 {"image_metrics/full/psnr": 25.5, ...}
#         metrics_weights: 指标权重字典，如果为None则使用默认权重
        
#     Returns:
#         综合分数（float）
#     """
#     if metrics_weights is None:
#         # 默认权重（基于17个指标的重要性）
#         metrics_weights = {
#             'psnr': 0.15, 'ssim': 0.15, 'lpips': 0.10,
#             'psnr_no_ego': 0.08, 'ssim_no_ego': 0.08, 'lpips_no_ego': 0.05,
#             'psnr_with_ego': 0.05, 'ssim_with_ego': 0.05, 'lpips_with_ego': 0.03,
#             'occupied_psnr': 0.06, 'occupied_ssim': 0.06,
#             'masked_psnr': 0.04, 'masked_ssim': 0.04,
#             'human_psnr': 0.02, 'human_ssim': 0.02,
#             'vehicle_psnr': 0.02, 'vehicle_ssim': 0.02,
#         }
    
#     total_score = 0.0
#     total_weight = 0.0
    
#     for metric in METRICS_TO_COMPARE:
#         if metric not in metrics:
#             continue

#         value = metrics[metric]

#         # 跳过无效值（-1 或 None）
#         if value is None or value == -1:
#             continue
        
#         metric_key = metric.split('/')[-1]  # 提取指标名称，如 "psnr", "ssim", "lpips"
        
#         # 获取权重
#         weight = metrics_weights.get(metric_key, 0.0)
#         if weight == 0.0:
#             continue
        
#         # 对于 LPIPS，需要取负值（因为越小越好）
#         if "lpips" in metric.lower():
#             value = -value  # 转换为越大越好的形式
        
#         # 加权求和
#         total_score += value * weight
#         total_weight += weight
    
#     return total_score / total_weight if total_weight > 0 else 0.0

# =========================
# 归一化函数
# =========================

def normalize_psnr(psnr: float, min_psnr: float = 15.0, max_psnr: float = 30.0) -> float:
    """
    将 PSNR 归一化到 [0, 1] 范围
    - PSNR 越大越好，所以直接线性映射
    - 低于 min_psnr 的值会被 clip 为 0
    - 高于 max_psnr 的值会被 clip 为 1
    在 3DGS + 自动驾驶 / 多视角场景中，这是合理的
    如果你将来用于 室内 NeRF / synthetic scene，需要调整
    """
    if psnr < min_psnr:
        return 0.0
    if psnr > max_psnr:
        return 1.0
    return (psnr - min_psnr) / (max_psnr - min_psnr)


def normalize_ssim(ssim: float, min_ssim: float = 0.5, max_ssim: float = 0.95) -> float:
    """
    将 SSIM 归一化到 [0, 1] 范围
    - SSIM 越大越好，所以直接线性映射
    - 低于 min_ssim 的值会被 clip 为 0
    - 高于 max_ssim 的值会被 clip 为 1

    """
    if ssim < min_ssim:
        return 0.0
    if ssim > max_ssim:
        return 1.0
    return (ssim - min_ssim) / (max_ssim - min_ssim)


def normalize_lpips(lpips: float, min_lpips: float = 0.0, max_lpips: float = 0.6) -> float:
    """
    将 LPIPS 归一化到 [0, 1] 范围
    - LPIPS 越小越好，所以取反：1.0 - normalized_lpips
    - LPIPS=0 时返回 1.0（最好）
    - LPIPS=max_lpips 时返回 0.0（最差）
    在 LPIPS ∈ [0.0, 0.2] 时，人眼差异很明显，但数值变化不大。
    所以对 norm 进行开根号，在高质量区间（norm 接近 1）数值变化被放大，
    在低质量区间（norm 接近 0），数值变化被压缩
    """
    if lpips < min_lpips:
        lpips = min_lpips
    if lpips > max_lpips:
        lpips = max_lpips
    norm = 1.0 - (lpips / max_lpips)
    return norm


# =========================
# 区域权重（同一指标内部）
# =========================

# 区域权重（同一指标内部）
# 注意：不同指标可能支持不同的区域
# - PSNR/SSIM: full, no_ego, with_ego, occupied, masked, vehicle, human
# - LPIPS: full, no_ego, with_ego (其他区域可能不存在)
REGION_WEIGHTS = {
    "full": 0.35,
    "no_ego": 0.15,
    "with_ego": 0.15,
    "occupied": 0.15,
    "masked": 0.10,
    "vehicle": 0.10,
    "human": 0.00,   # 若存在可开启，否则不影响整体
}


# =========================
# 核心函数
# =========================

def compute_overall_score(metrics: Dict[str, float]) -> float:
    """
    适用于 3DGS 场景重建的综合评分函数
    
    计算流程：
    1. 对每个指标类型（PSNR, SSIM, LPIPS）分别归一化到 [0, 1] 范围
    2. 对每个指标类型，按区域权重聚合所有有效区域的归一化值
    3. 如果某些区域缺失，权重会重新归一化分配给有效区域
    4. 最后将三个一级指标按权重（0.4, 0.3, 0.3）融合得到最终分数

    Args:
        metrics: 指标字典，key 示例：
                 image_metrics/full/psnr (全图 PSNR)
                 image_metrics/full/ssim (全图 SSIM)
                 image_metrics/full/lpips (全图 LPIPS)
                 image_metrics/full/psnr_no_ego (无 ego 区域 PSNR)
                 image_metrics/full/occupied_psnr (占用区域 PSNR)
                 ...
                 注意：值为 -1 或 None 的指标会被视为无效并跳过

    Returns:
        综合分数 ∈ [0, 1]，越大越好
    """

    # -------------------------
    # 工具函数：读取区域指标
    # -------------------------
    def get_metric(metric_name: str, region: str):
        """从 metrics 字典中获取指定区域和类型的指标值"""
        if region == "full":
            key = f"image_metrics/full/{metric_name}"
        else:
            key = f"image_metrics/full/{metric_name}_{region}"
        value = metrics.get(key, None)
        # 跳过无效值（-1 表示该指标不可用）
        if value is None or value == -1:
            return None
        return value

    def aggregate_metric(metric_name: str, normalize_fn) -> float:
        """
        聚合某个指标类型的所有区域值
        
        Args:
            metric_name: 指标名称，如 "psnr", "ssim", "lpips"
            normalize_fn: 归一化函数
            
        Returns:
            聚合后的归一化分数 [0, 1]
        """
        score = 0.0
        weight_sum = 0.0

        for region, w in REGION_WEIGHTS.items():
            value = get_metric(metric_name, region)
            if value is None:
                continue
            # 归一化并加权累加
            score += normalize_fn(value) * w
            weight_sum += w

        # 重新归一化权重，确保即使某些区域缺失，权重总和仍为1
        if weight_sum > 0:
            return score / weight_sum
        else:
            # 所有区域都无效，返回 0
            return 0.0

    # -------------------------
    # 二级指标聚合（按区域加权）
    # -------------------------
    psnr_score = aggregate_metric("psnr", normalize_psnr)
    ssim_score = aggregate_metric("ssim", normalize_ssim)
    lpips_score = aggregate_metric("lpips", normalize_lpips)

    # -------------------------
    # 一级指标融合（最终得分）
    # -------------------------
    # PSNR: 0.4, SSIM: 0.3, LPIPS: 0.3
    # 这些权重可以根据实际需求调整
    overall_score = (
        0.4 * psnr_score +
        0.3 * ssim_score +
        0.3 * lpips_score
    )

    return float(np.clip(overall_score, 0.0, 1.0))

def find_best_improved_frames_per_camera(
    log_dir1: str,
    log_dir2: str,
    common_cam_ids: Optional[Set[int]] = None,
    metrics1: Optional[Dict] = None,
    metrics2: Optional[Dict] = None,
    reverse: bool = False
) -> Dict[int, Dict]:
    """
    对每个相机ID分别找出指标提升最大的帧（或下降最大的帧，如果reverse=True）
    返回: {cam_id: {"frame_idx": int, "cam_id": int, "image_path_old": str, "image_path_new": str, "improvement_score": float}}
    
    Args:
        reverse: 如果为True，找出新结果比旧结果差的帧（即旧结果相对新结果提升最大的帧）
    """
    def find_images_dir(base_dir: str) -> Optional[str]:
        """查找 full_set_*_rgbs 目录（videos_eval 优先，其次 videos）"""
        patterns = [
            os.path.join(base_dir, "videos_eval", "full_set_*_rgbs"),
            os.path.join(base_dir, "videos", "full_set_*_rgbs"),
        ]
        for pattern in patterns:
            dirs = glob.glob(pattern)
            if dirs:
                def extract_number(path):
                    match = re.search(r'full_set_(\d+)_rgbs', path)
                    return int(match.group(1)) if match else 0
                dirs.sort(key=extract_number, reverse=True)
                return dirs[0]
        return None

    def group_images_by_cam_frame(files, common_ids):
        grouped = {}
        for f in files:
            result = extract_frame_cam_from_filename(os.path.basename(f))
            if result:
                frame_id, cam_id = result
                if common_ids is None or cam_id in common_ids:
                    grouped[(cam_id, frame_id)] = f
        return grouped

    def find_gt_dir(base_dir: str) -> Optional[str]:
        """查找 full_set_*_gt_rgbs 目录（videos_eval 优先，其次 videos）"""
        patterns = [
            os.path.join(base_dir, "videos_eval", "full_set_*_gt_rgbs"),
            os.path.join(base_dir, "videos", "full_set_*_gt_rgbs"),
        ]
        for pattern in patterns:
            dirs = glob.glob(pattern)
            if dirs:
                def extract_number(path):
                    match = re.search(r'full_set_(\d+)_gt_rgbs', path)
                    return int(match.group(1)) if match else 0
                dirs.sort(key=extract_number, reverse=True)
                return dirs[0]
        return None

    def load_gt_grouped(gt_dir: Optional[str], common_ids) -> Dict[Tuple[int, int], str]:
        grouped = {}
        if not gt_dir:
            return grouped
        gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
        for f in gt_files:
            result = extract_frame_cam_from_filename(os.path.basename(f))
            if result:
                frame_id, cam_id = result
                if common_ids is None or cam_id in common_ids:
                    grouped[(cam_id, frame_id)] = f
        return grouped

    def compute_improvement_scores(common_keys, grouped1, grouped2, gt_grouped1, gt_grouped2, reverse):
        import torch
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr
        
        frames_by_cam = {}
        total_frames = len(common_keys)
        logger.info(f"Processing {total_frames} frames across {len(set(cam_id for cam_id, _ in common_keys))} cameras...")
        
        # 批量处理LPIPS以提高GPU利用率
        batch_size = 32  # 批量大小
        loss_fn = get_lpips_model()
        use_lpips = loss_fn is not None
        
        # 准备批量数据
        batch_data = []
        frame_info = []  # 保存每帧的元信息
        
        # 第一步：扫描所有图像，找到最大尺寸
        logger.info("Scanning images to determine target size...")
        max_h, max_w = 0, 0
        valid_frames = []
        for cam_id, frame_id in common_keys:
            img1_path = grouped1[(cam_id, frame_id)]
            img2_path = grouped2[(cam_id, frame_id)]
            gt_path = gt_grouped1.get((cam_id, frame_id)) or gt_grouped2.get((cam_id, frame_id))
            
            if not gt_path or not os.path.exists(gt_path):
                continue
            
            try:
                # 只读取尺寸信息，不加载完整图像
                img1 = imageio.imread(img1_path)
                img2 = imageio.imread(img2_path)
                gt_img = imageio.imread(gt_path)
                
                if img1 is None or img2 is None or gt_img is None:
                    continue
                
                # 找到三张图像的最小尺寸（确保都能resize到这个尺寸）
                h = min(img1.shape[0], img2.shape[0], gt_img.shape[0])
                w = min(img1.shape[1], img2.shape[1], gt_img.shape[1])
                
                max_h = max(max_h, h)
                max_w = max(max_w, w)
                valid_frames.append((cam_id, frame_id, img1_path, img2_path, gt_path))
            except:
                continue
        
        # 使用固定尺寸或最大尺寸（取512的倍数，便于GPU处理）
        target_h = ((max_h + 31) // 32) * 32  # 向上取整到32的倍数
        target_w = ((max_w + 31) // 32) * 32
        # 限制最大尺寸，避免内存问题
        target_h = min(target_h, 512)
        target_w = min(target_w, 512)
        
        logger.info(f"Target image size: {target_h}x{target_w}")
        logger.info(f"Loading {len(valid_frames)} valid frames and preparing batches...")
        
        for idx, (cam_id, frame_id, img1_path, img2_path, gt_path) in enumerate(valid_frames):
            frames_by_cam.setdefault(cam_id, [])
            
            try:
                img1 = imageio.imread(img1_path)
                img2 = imageio.imread(img2_path)
                gt_img = imageio.imread(gt_path)
                
                if img1 is None or img2 is None or gt_img is None:
                    frame_info.append({
                        'cam_id': cam_id,
                        'frame_id': frame_id,
                        'img1_path': img1_path,
                        'img2_path': img2_path,
                        'gt_path': gt_path,
                        'batch_idx': -1
                    })
                    continue
                
                # 统一resize到目标尺寸
                img1 = cv2.resize(img1, (target_w, target_h))
                img2 = cv2.resize(img2, (target_w, target_h))
                gt_img = cv2.resize(gt_img, (target_w, target_h))
                
                # 归一化到[0,1]
                img1_norm = img1.astype(float) / 255.0
                img2_norm = img2.astype(float) / 255.0
                gt_norm = gt_img.astype(float) / 255.0
                
                batch_idx = len(batch_data)
                # 根据reverse决定old和new
                if reverse:
                    # log_dir1相对于log_dir2的改进：old=img2(log_dir2), new=img1(log_dir1)
                    old_norm = img2_norm
                    new_norm = img1_norm
                else:
                    # log_dir2相对于log_dir1的改进：old=img1(log_dir1), new=img2(log_dir2)
                    old_norm = img1_norm
                    new_norm = img2_norm
                
                batch_data.append({
                    'old_norm': old_norm,
                    'new_norm': new_norm,
                    'gt_norm': gt_norm,
                    'img1': img1,
                    'img2': img2,
                    'gt_img': gt_img
                })
                
                frame_info.append({
                    'cam_id': cam_id,
                    'frame_id': frame_id,
                    'img1_path': img1_path,
                    'img2_path': img2_path,
                    'gt_path': gt_path,
                    'batch_idx': batch_idx
                })
                
            except Exception as e:
                logger.debug(f"Error loading images for frame ({cam_id}, {frame_id}): {e}")
                frame_info.append({
                    'cam_id': cam_id,
                    'frame_id': frame_id,
                    'img1_path': img1_path,
                    'img2_path': img2_path,
                    'gt_path': gt_path,
                    'batch_idx': -1
                })
                continue
            
            if (idx + 1) % 100 == 0:
                logger.info(f"  Loaded {idx + 1}/{len(valid_frames)} frames...")
        
        # 批量计算LPIPS
        lpips_scores = {}
        if use_lpips and batch_data:
            logger.info(f"Computing LPIPS for {len(batch_data)} frames in batches of {batch_size}...")
            with torch.no_grad():
                for batch_start in range(0, len(batch_data), batch_size):
                    batch_end = min(batch_start + batch_size, len(batch_data))
                    batch = batch_data[batch_start:batch_end]
                    
                    # 准备批量张量
                    gt_tensors = []
                    old_tensors = []
                    new_tensors = []
                    
                    for item in batch:
                        gt_tensor = torch.from_numpy(item['gt_norm']).permute(2, 0, 1).unsqueeze(0).float().to(_lpips_device)
                        old_tensor = torch.from_numpy(item['old_norm']).permute(2, 0, 1).unsqueeze(0).float().to(_lpips_device)
                        new_tensor = torch.from_numpy(item['new_norm']).permute(2, 0, 1).unsqueeze(0).float().to(_lpips_device)
                        
                        gt_tensors.append(gt_tensor)
                        old_tensors.append(old_tensor)
                        new_tensors.append(new_tensor)
                    
                    # 拼接成批量
                    gt_batch = torch.cat(gt_tensors, dim=0)
                    old_batch = torch.cat(old_tensors, dim=0)
                    new_batch = torch.cat(new_tensors, dim=0)
                    
                    # 批量计算LPIPS
                    lpips_gt_old = loss_fn(gt_batch, old_batch).cpu().numpy()
                    lpips_gt_new = loss_fn(gt_batch, new_batch).cpu().numpy()
                    
                    # 保存结果
                    for i, batch_idx in enumerate(range(batch_start, batch_end)):
                        # 确保从numpy数组中提取标量值
                        lpips_old_val = lpips_gt_old[i]
                        lpips_new_val = lpips_gt_new[i]
                        if isinstance(lpips_old_val, np.ndarray):
                            lpips_old_val = lpips_old_val.item()
                        if isinstance(lpips_new_val, np.ndarray):
                            lpips_new_val = lpips_new_val.item()
                        lpips_scores[batch_idx] = {
                            'lpips_old': float(lpips_old_val),
                            'lpips_new': float(lpips_new_val)
                        }
                    
                    if (batch_start // batch_size + 1) % 10 == 0:
                        logger.info(f"  Processed {batch_end}/{len(batch_data)} frames for LPIPS")
        
        # 计算每帧的综合分数
        logger.info("Computing overall scores for all frames...")
        for info in frame_info:
            cam_id = info['cam_id']
            frame_id = info['frame_id']
            batch_idx = info['batch_idx']
            
            if batch_idx == -1:
                # 没有有效数据，分数为0
                frames_by_cam[cam_id].append((frame_id, 0.0))
                continue
            
            batch_item = batch_data[batch_idx]
            old_norm = batch_item['old_norm']
            new_norm = batch_item['new_norm']
            gt_norm = batch_item['gt_norm']
            
            scores = {}
            
            # 计算原始指标值
            psnr_old = psnr(gt_norm, old_norm, data_range=1.0)
            psnr_new = psnr(gt_norm, new_norm, data_range=1.0)
            
            ssim_old = ssim(gt_norm, old_norm, data_range=1.0, channel_axis=2 if len(gt_norm.shape) == 3 else None)
            ssim_new = ssim(gt_norm, new_norm, data_range=1.0, channel_axis=2 if len(gt_norm.shape) == 3 else None)
            
            # LPIPS（从批量结果中获取）
            if batch_idx in lpips_scores:
                lpips_old = lpips_scores[batch_idx]['lpips_old']
                lpips_new = lpips_scores[batch_idx]['lpips_new']
            else:
                lpips_old = 0.0
                lpips_new = 0.0
            
            # 先归一化指标值，再计算改进（与 compute_overall_score 保持一致）
            # 这样可以统一不同指标的数值尺度
            psnr_old_norm = normalize_psnr(psnr_old)
            psnr_new_norm = normalize_psnr(psnr_new)
            scores['psnr'] = psnr_new_norm - psnr_old_norm
            
            ssim_old_norm = normalize_ssim(ssim_old)
            ssim_new_norm = normalize_ssim(ssim_new)
            scores['ssim'] = ssim_new_norm - ssim_old_norm
            
            # LPIPS 归一化后也是越大越好
            lpips_old_norm = normalize_lpips(lpips_old)
            lpips_new_norm = normalize_lpips(lpips_new)
            scores['lpips'] = lpips_new_norm - lpips_old_norm
            
            # 直接使用全图指标的改进值计算综合分数
            # 
            # 注意：由于区域指标改进值是全图改进值的线性变换（乘以某个系数），
            # 加权聚合后仍然是全图改进值的线性变换，因此可以直接使用全图改进值。
            # 
            # 数学证明：
            # 设全图改进值为 I_full，区域改进值为 I_region = I_full * c (c为系数)
            # 加权聚合：I_agg = (I_full * w_full + I_full * c * w_region) / (w_full + w_region)
            #         = I_full * (w_full + c * w_region) / (w_full + w_region)
            #         = I_full * k (k为常数)
            # 因此，聚合结果仍然是全图改进值的线性变换，可以直接使用全图改进值。
            
            # 直接使用全图改进值，按一级指标权重融合（与 compute_overall_score 保持一致）
            # PSNR: 0.4, SSIM: 0.3, LPIPS: 0.3
            improvement_score = (
                0.4 * np.clip(scores['psnr'], -1.0, 1.0) +
                0.3 * np.clip(scores['ssim'], -1.0, 1.0) +
                0.3 * np.clip(scores['lpips'], -1.0, 1.0)
            )
            frames_by_cam[cam_id].append((frame_id, improvement_score))
        
        processed_frames = len(frame_info)
        logger.info(f"Completed processing {processed_frames}/{total_frames} frames")
        return frames_by_cam

    def select_best_frames(frames_by_cam, grouped1, grouped2, gt_grouped1, gt_grouped2):
        best_frames = {}
        for cam_id, frame_scores in frames_by_cam.items():
            if not frame_scores:
                continue
            frame_scores.sort(key=lambda x: x[1], reverse=True)
            best_frame_id, best_improvement = frame_scores[0]
            key = (cam_id, best_frame_id)
            if key in grouped1 and key in grouped2:
                # 查找GT图片路径（优先从gt_grouped1，如果没有则从gt_grouped2）
                gt_path = gt_grouped1.get(key) or gt_grouped2.get(key) or ""
                best_frames[cam_id] = {
                    "frame_idx": best_frame_id,
                    "cam_id": cam_id,
                    "image_path_old": grouped1[key],
                    "image_path_new": grouped2[key],
                    "image_path_gt": gt_path,
                    "improvement_score": best_improvement
                }
                logger.info(f"Cam {cam_id}: Best improved frame {best_frame_id} with improvement score {best_improvement:.4f}")
        return best_frames

    # 主流程
    images_dir1 = find_images_dir(log_dir1)
    images_dir2 = find_images_dir(log_dir2)
    if not images_dir1 or not images_dir2:
        logger.warning(f"Cannot find image directories for comparison")
        logger.warning(f"  Dir1: {images_dir1}, Dir2: {images_dir2}")
        return {}

    logger.info(f"Found image directories:")
    logger.info(f"  Run 1: {images_dir1}")
    logger.info(f"  Run 2: {images_dir2}")

    image_files1 = sorted(glob.glob(os.path.join(images_dir1, "*.png")))
    image_files2 = sorted(glob.glob(os.path.join(images_dir2, "*.png")))
    if not image_files1 or not image_files2:
        logger.warning(f"No image files found in directories")
        return {}

    grouped1 = group_images_by_cam_frame(image_files1, common_cam_ids)
    grouped2 = group_images_by_cam_frame(image_files2, common_cam_ids)
    common_keys = set(grouped1.keys()) & set(grouped2.keys())

    gt_dir1 = find_gt_dir(log_dir1)
    gt_dir2 = find_gt_dir(log_dir2)
    gt_grouped1 = load_gt_grouped(gt_dir1, common_cam_ids)
    gt_grouped2 = load_gt_grouped(gt_dir2, common_cam_ids)

    logger.info("Preloading LPIPS model...")
    get_lpips_model()

    # 使用与 compute_overall_score 相同的权重结构
    # 不再使用动态权重，确保计算一致性
    frames_by_cam = compute_improvement_scores(
        common_keys, grouped1, grouped2, gt_grouped1, gt_grouped2, reverse
    )

    best_frames = select_best_frames(frames_by_cam, grouped1, grouped2, gt_grouped1, gt_grouped2)
    return best_frames


def main(args):
    # 初始化 wandb
    if args.enable_wandb:
        wandb.init(
            project=args.wandb_project or "scene_recon_compare",
            name=args.wandb_run_name or f"compare_{os.path.basename(args.log_dir1)}_{os.path.basename(args.log_dir2)}",
            entity=args.wandb_entity or "1zzhaozz-nanjing-university",
        )
    
    # 获取两次训练的相机配置
    logger.info("Detecting camera configurations...")
    cam_ids1 = get_camera_ids(args.log_dir1)
    cam_ids2 = get_camera_ids(args.log_dir2)
    
    if cam_ids1:
        logger.info(f"Run 1 camera IDs: {sorted(cam_ids1)}")
    else:
        logger.warning(f"Could not detect camera IDs for {args.log_dir1}")
    
    if cam_ids2:
        logger.info(f"Run 2 camera IDs: {sorted(cam_ids2)}")
    else:
        logger.warning(f"Could not detect camera IDs for {args.log_dir2}")
    
    # 获取共同的相机ID
    common_cam_ids = get_common_camera_ids(cam_ids1, cam_ids2)
    
    if cam_ids1 and cam_ids2:
        if len(common_cam_ids) == 0:
            logger.warning("No common camera IDs found! Comparison may not be meaningful.")
        elif len(common_cam_ids) < len(set(cam_ids1)) or len(common_cam_ids) < len(set(cam_ids2)):
            logger.info(f"Common camera IDs: {sorted(common_cam_ids)}")
            logger.info(f"Run 1 only cameras: {sorted(set(cam_ids1) - common_cam_ids)}")
            logger.info(f"Run 2 only cameras: {sorted(set(cam_ids2) - common_cam_ids)}")
            logger.info("Will only compare images from common cameras.")
        else:
            logger.info(f"Both runs use the same camera configuration: {sorted(common_cam_ids)}")
    
    # 加载两次评估的指标
    logger.info(f"Loading metrics from {args.log_dir1}")
    metrics_file1 = find_latest_metrics_file(args.log_dir1, "images_full")
    if not metrics_file1:
        raise FileNotFoundError(f"Cannot find metrics file in {args.log_dir1}")
    
    logger.info(f"Loading metrics from {args.log_dir2}")
    metrics_file2 = find_latest_metrics_file(args.log_dir2, "images_full")
    if not metrics_file2:
        raise FileNotFoundError(f"Cannot find metrics file in {args.log_dir2}")
    
    metrics1 = load_metrics_file(metrics_file1)
    metrics2 = load_metrics_file(metrics_file2)
    
    logger.info(f"Metrics file 1: {metrics_file1}")
    logger.info(f"Metrics file 2: {metrics_file2}")
    
    # 只处理全图指标（image_metrics/full/）
    full_metrics = [m for m in METRICS_TO_COMPARE if m.startswith("image_metrics/full/")]
    
    # 对比指标
    comparison = compare_metrics(metrics1, metrics2)
    
    # 只保留全图指标的对比结果
    full_comparison = {k: v for k, v in comparison.items() if k in full_metrics}
    
    # 找出改进的指标
    improved_metrics = {k: v for k, v in full_comparison.items() if v["improved"]}
    
    logger.info(f"\n{'='*80}")
    logger.info(f"指标对比结果（仅全图指标）")
    logger.info(f"{'='*80}")
    logger.info(f"总共对比 {len(full_comparison)} 个全图指标")
    logger.info(f"改进的指标: {len(improved_metrics)} 个")
    logger.info(f"退化的指标: {len(full_comparison) - len(improved_metrics)} 个")
    logger.info(f"\n改进的指标详情:")
    
    # 按改进幅度排序
    sorted_improved = sorted(
        improved_metrics.items(),
        key=lambda x: abs(x[1]["diff"]),
        reverse=True
    )
    
    wandb_data = {}
    wandb_images = {}
    
    # 提取文件夹名称用于显示
    dir1_name = os.path.basename(args.log_dir1.rstrip('/'))
    dir2_name = os.path.basename(args.log_dir2.rstrip('/'))
    
    # 计算两次评估结果的综合分数
    logger.info(f"\n计算两次评估结果的综合分数...")
    score1 = compute_overall_score(metrics1)
    score2 = compute_overall_score(metrics2)
    
    logger.info(f"{dir1_name} 评估综合分数: {score1:.6f}")
    logger.info(f"{dir2_name} 评估综合分数: {score2:.6f}")
    
    # 记录综合分数到 wandb
    if args.enable_wandb:
        wandb_data[f"overall_score/{dir1_name}"] = score1
        wandb_data[f"overall_score/{dir2_name}"] = score2
        wandb_data["overall_score/diff"] = score2 - score1
    
    # 根据综合分数决定方向
    # 如果第一次高于第二次，计算第一次相对于第二次的最佳改进帧（reverse=True）
    # 如果第二次高于第一次，计算第二次相对于第一次的最佳改进帧（reverse=False）
    reverse = score1 > score2
    
    if reverse:
        logger.info(f"{dir1_name} 评估结果更好（{score1:.6f} > {score2:.6f}），将计算 {dir1_name} 相对于 {dir2_name} 的最佳改进帧")
    else:
        logger.info(f"{dir2_name} 评估结果更好（{score2:.6f} > {score1:.6f}），将计算 {dir2_name} 相对于 {dir1_name} 的最佳改进帧")
    
    # 获取每个相机的最佳改进帧（根据综合分数决定方向）
    best_frames_per_cam = {}
    if len(common_cam_ids) > 0:
        logger.info(f"\n开始查找每个相机的最佳改进帧...")
        best_frames_per_cam = find_best_improved_frames_per_camera(
            args.log_dir1,
            args.log_dir2,
            common_cam_ids=common_cam_ids,
            metrics1=metrics1,
            metrics2=metrics2,
            reverse=reverse
        )
        logger.info(f"找到 {len(best_frames_per_cam)} 个相机的最佳改进帧")
        
        # 输出最佳改进帧信息到控制台
        if best_frames_per_cam:
            logger.info(f"\n{'='*80}")
            if reverse:
                logger.info(f"各相机最佳改进帧信息（{dir1_name} 相对于 {dir2_name} 提升最大的帧）")
            else:
                logger.info(f"各相机最佳改进帧信息（{dir2_name} 相对于 {dir1_name} 提升最大的帧）")
            logger.info(f"{'='*80}")
            for cam_id in sorted(best_frames_per_cam.keys()):
                best_frame = best_frames_per_cam[cam_id]
                logger.info(f"\n相机 {cam_id}:")
                logger.info(f"  最佳改进帧ID: {best_frame.get('frame_idx', 'N/A')}")
                logger.info(f"  改进分数: {best_frame.get('improvement_score', 0):.6f}")
                logger.info(f"  {dir1_name} 图像路径: {best_frame.get('image_path_old', 'N/A')}")
                logger.info(f"  {dir2_name} 图像路径: {best_frame.get('image_path_new', 'N/A')}")
    else:
        logger.warning("没有共同的相机ID，无法查找最佳改进帧")
    
    for metric_name, metric_data in sorted_improved:
        logger.info(f"\n{metric_name}:")
        logger.info(f"  {dir1_name}: {metric_data['old']:.6f}")
        logger.info(f"  {dir2_name}: {metric_data['new']:.6f}")
        logger.info(f"  变化: {metric_data['diff']:+.6f} ({metric_data['improvement_pct']:+.2f}%)")
        
        # 记录到 wandb
        if args.enable_wandb:
            wandb_data[f"comparison/{metric_name}/old"] = metric_data['old']
            wandb_data[f"comparison/{metric_name}/new"] = metric_data['new']
            wandb_data[f"comparison/{metric_name}/diff"] = metric_data['diff']
            wandb_data[f"comparison/{metric_name}/improvement_pct"] = metric_data['improvement_pct']
    
    # 为所有相机输出对比图片到 wandb（只输出一次，不按指标重复）
    if args.enable_wandb and best_frames_per_cam:
        logger.info(f"\n开始上传最佳改进帧图片到 wandb...")
        for cam_id, best_frame in best_frames_per_cam.items():
            if best_frame.get("image_path_old") and best_frame.get("image_path_new"):
                if os.path.exists(best_frame["image_path_old"]) and os.path.exists(best_frame["image_path_new"]):
                    try:
                        img_old = imageio.imread(best_frame["image_path_old"])
                        img_new = imageio.imread(best_frame["image_path_new"])
                        
                        frame_idx = best_frame.get('frame_idx', 'N/A')
                        
                        # 上传图像，使用文件夹名称
                        wandb_images[f"best_improvement/cam{cam_id}_{dir1_name}"] = wandb.Image(
                            img_old,
                            caption=f"Cam {cam_id}, Frame {frame_idx} - {dir1_name}"
                        )
                        
                        # 上传新图像
                        wandb_images[f"best_improvement/cam{cam_id}_{dir2_name}"] = wandb.Image(
                            img_new,
                            caption=f"Cam {cam_id}, Frame {frame_idx} - {dir2_name}"
                        )
                        
                        logger.info(f"  已上传相机 {cam_id} 的最佳改进帧 (Frame {frame_idx})")
                    except Exception as e:
                        logger.warning(f"Error loading images for cam {cam_id}: {e}")
    
    # 记录所有对比结果到 wandb
    if args.enable_wandb:
        logger.info(f"\n{'='*80}")
        logger.info("开始上传结果到 wandb...")
        
        # 上传指标数据
        logger.info(f"  上传指标数据: {len(wandb_data)} 个数据点")
        wandb.log(wandb_data)
        
        # 上传图片
        if wandb_images:
            logger.info(f"  上传对比图片: {len(wandb_images)} 张图片")
            wandb.log(wandb_images)
        
        # 创建对比表格（只包含全图指标）
        table_data = []
        for metric_name, metric_data in full_comparison.items():
            table_data.append([
                metric_name,
                f"{metric_data['old']:.6f}",
                f"{metric_data['new']:.6f}",
                f"{metric_data['diff']:+.6f}",
                f"{metric_data['improvement_pct']:+.2f}%",
                "✓" if metric_data['improved'] else "✗"
            ])
        # 追加总体分数行
        overall_diff = score2 - score1
        overall_pct = (overall_diff / score1 * 100) if score1 != 0 else 0.0
        table_data.append([
            "overall_score",
            f"{score1:.6f}",
            f"{score2:.6f}",
            f"{overall_diff:+.6f}",
            f"{overall_pct:+.2f}%",
            "✓" if score2 > score1 else "✗"
        ])
        
        table = wandb.Table(
            columns=["Metric", dir1_name, dir2_name, "Difference", "Improvement %", "Improved"],
            data=table_data
        )
        logger.info(f"  上传指标对比表格: {len(table_data)} 行数据")
        wandb.log({"comparison_table": table})
        
        # 创建最佳改进帧信息表格（只包含图片）
        if best_frames_per_cam:
            best_frames_table_data = []
            for cam_id in sorted(best_frames_per_cam.keys()):
                best_frame = best_frames_per_cam[cam_id]
                
                # 加载图片用于表格显示
                img_old_cell = None
                img_new_cell = None
                
                old_path = best_frame.get('image_path_old', '')
                new_path = best_frame.get('image_path_new', '')
                
                if old_path and os.path.exists(old_path):
                    try:
                        img_old = imageio.imread(old_path)
                        img_old_cell = wandb.Image(img_old)
                    except Exception as e:
                        logger.warning(f"无法加载旧图像用于表格 (Cam {cam_id}): {e}")
                        img_old_cell = os.path.basename(old_path) if old_path else 'N/A'
                
                if new_path and os.path.exists(new_path):
                    try:
                        img_new = imageio.imread(new_path)
                        img_new_cell = wandb.Image(img_new)
                    except Exception as e:
                        logger.warning(f"无法加载新图像用于表格 (Cam {cam_id}): {e}")
                        img_new_cell = os.path.basename(new_path) if new_path else 'N/A'
                
                # 如果图片加载失败，使用文件名作为后备
                if img_old_cell is None:
                    img_old_cell = os.path.basename(old_path) if old_path else 'N/A'
                if img_new_cell is None:
                    img_new_cell = os.path.basename(new_path) if new_path else 'N/A'
                
                best_frames_table_data.append([
                    f"Cam {cam_id}",
                    best_frame.get('frame_idx', 'N/A'),
                    f"{best_frame.get('improvement_score', 0):.6f}",
                    img_old_cell,
                    img_new_cell
                ])
            
            best_frames_table = wandb.Table(
                columns=["Camera ID", "Frame ID", "Improvement Score", dir1_name, dir2_name],
                data=best_frames_table_data
            )
            logger.info(f"  上传最佳改进帧表格（包含图片）: {len(best_frames_table_data)} 行数据")
            wandb.log({"best_improvement_frames_table": best_frames_table})

        logger.info("所有数据已上传到 wandb")
    
    # 保存表格到本地文件（无论是否启用wandb都要保存）
    if best_frames_per_cam or full_comparison:
        try:
            # 提取模型名称与保存目录
            old_model_name = os.path.basename(args.log_dir1.rstrip('/'))
            new_model_name = os.path.basename(args.log_dir2.rstrip('/'))
            log_dir1_abs = os.path.abspath(args.log_dir1)
            base_output_dir = os.path.dirname(log_dir1_abs)
            
            # 生成时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存路径：compare_model/{dir1_name}_{dir2_name}_{timestamp}
            save_base_dir = os.path.join(
                base_output_dir, 
                "compare_model", 
                f"{old_model_name}_{new_model_name}_{timestamp}"
            )
            os.makedirs(save_base_dir, exist_ok=True)

            if not OPENPYXL_AVAILABLE:
                logger.warning("openpyxl未安装，无法生成Excel文件。请运行: pip install openpyxl")
                logger.info("将使用CSV格式保存...")
                # 回退到CSV格式
                if full_comparison:
                    comparison_csv = os.path.join(save_base_dir, "comparison_table.csv")
                    with open(comparison_csv, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(["Metric", dir1_name, dir2_name, "Difference", "Improvement %", "Improved"])
                        for metric_name, metric_data in full_comparison.items():
                            writer.writerow([
                                metric_name,
                                f"{metric_data['old']:.6f}",
                                f"{metric_data['new']:.6f}",
                                f"{metric_data['diff']:+.6f}",
                                f"{metric_data['improvement_pct']:+.2f}%",
                                "✓" if metric_data['improved'] else "✗"
                            ])
                        # 添加总体分数行
                        score_diff = score2 - score1
                        score_improvement_pct = (score_diff / score1 * 100) if score1 != 0 else 0.0
                        writer.writerow([
                            "overall_score",
                            f"{score1:.6f}",
                            f"{score2:.6f}",
                            f"{score_diff:+.6f}",
                            f"{score_improvement_pct:+.2f}%",
                            "✓" if score2 > score1 else "✗"
                        ])
                    logger.info(f"已保存 comparison_table 到本地: {comparison_csv}")
                
                if best_frames_per_cam:
                    best_frames_csv = os.path.join(save_base_dir, "best_improvement_frames_table.csv")
                    with open(best_frames_csv, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            "Camera ID", "Frame ID", "Improvement Score", dir1_name, dir2_name
                        ])
                        for cam_id in sorted(best_frames_per_cam.keys()):
                            best_frame = best_frames_per_cam[cam_id]
                            old_path = best_frame.get('image_path_old', '')
                            new_path = best_frame.get('image_path_new', '')
                            writer.writerow([
                                f"Cam {cam_id}",
                                best_frame.get('frame_idx', 'N/A'),
                                f"{best_frame.get('improvement_score', 0):.6f}",
                                old_path if old_path and os.path.exists(old_path) else "N/A",
                                new_path if new_path and os.path.exists(new_path) else "N/A"
                            ])
                    logger.info(f"已保存最佳改进帧表格到本地: {best_frames_csv}")
            else:
                # 使用Excel格式保存
                # comparison_table 保存为 Excel（全图指标）
                if full_comparison:
                    comparison_xlsx = os.path.join(save_base_dir, "comparison_table.xlsx")
                    wb = Workbook()
                    ws = wb.active
                    ws.title = "Comparison"
                    
                    # 写入表头
                    headers = ["Metric", dir1_name, dir2_name, "Difference", "Improvement %", "Improved"]
                    ws.append(headers)
                    
                    # 写入数据
                    for metric_name, metric_data in full_comparison.items():
                        ws.append([
                            metric_name,
                            f"{metric_data['old']:.6f}",
                            f"{metric_data['new']:.6f}",
                            f"{metric_data['diff']:+.6f}",
                            f"{metric_data['improvement_pct']:+.2f}%",
                            "✓" if metric_data['improved'] else "✗"
                        ])
                    
                    # 添加总体分数行
                    score_diff = score2 - score1
                    score_improvement_pct = (score_diff / score1 * 100) if score1 != 0 else 0.0
                    ws.append([
                        "overall_score",
                        f"{score1:.6f}",
                        f"{score2:.6f}",
                        f"{score_diff:+.6f}",
                        f"{score_improvement_pct:+.2f}%",
                        "✓" if score2 > score1 else "✗"
                    ])
                    
                    # 调整列宽
                    for col in range(1, len(headers) + 1):
                        ws.column_dimensions[get_column_letter(col)].width = 15
                    
                    wb.save(comparison_xlsx)
                    logger.info(f"已保存 comparison_table 到本地: {comparison_xlsx}")

                # 保存最佳改进帧表格（Excel格式，嵌入图片，字段与wandb表格保持一致）
                if best_frames_per_cam:
                    best_frames_xlsx = os.path.join(save_base_dir, "best_improvement_frames_table.xlsx")
                    wb = Workbook()
                    ws = wb.active
                    ws.title = "Best Improvement Frames"
                    
                    # 字段：["Camera ID", "Frame ID", "Improvement Score", dir1_name, dir2_name, "GT"]
                    headers = ["Camera ID", "Frame ID", "Improvement Score", dir1_name, dir2_name, "GT"]
                    ws.append(headers)
                    
                    # 先扫描所有图片，找到最大尺寸（用于设置统一的列宽和行高）
                    max_img_width = 0
                    max_img_height = 0
                    for cam_id in sorted(best_frames_per_cam.keys()):
                        best_frame = best_frames_per_cam[cam_id]
                        for img_path in [best_frame.get('image_path_old', ''), 
                                        best_frame.get('image_path_new', ''),
                                        best_frame.get('image_path_gt', '')]:
                            if img_path and os.path.exists(img_path):
                                try:
                                    img = imageio.imread(img_path)
                                    if img is not None:
                                        h, w = img.shape[:2]
                                        max_img_width = max(max_img_width, w)
                                        max_img_height = max(max_img_height, h)
                                except:
                                    pass
                    
                    # 图片尺寸为原始的一半
                    target_img_width = max_img_width // 2 if max_img_width > 0 else 256
                    target_img_height = max_img_height // 2 if max_img_height > 0 else 256
                    
                    # 设置列宽（Excel列宽单位：1单位 ≈ 7像素，图片列宽需要匹配图片宽度的一半）
                    # 图片列宽 = 图片宽度(像素) / 7，但需要加上一些边距
                    img_col_width = max(target_img_width / 7 + 2, 15)  # 至少15，加上2的边距
                    
                    ws.column_dimensions['A'].width = 12  # Camera ID
                    ws.column_dimensions['B'].width = 12  # Frame ID
                    ws.column_dimensions['C'].width = 18  # Improvement Score
                    ws.column_dimensions['D'].width = img_col_width  # dir1_name (图片)
                    ws.column_dimensions['E'].width = img_col_width  # dir2_name (图片)
                    ws.column_dimensions['F'].width = img_col_width  # GT (图片)
                    
                    # 设置行高（Excel行高单位：1单位 = 1/72英寸，大约1像素=0.75点）
                    # 行高需要匹配图片高度的一半
                    row_height = target_img_height * 0.75 if target_img_height > 0 else 100
                    
                    row_idx = 2  # 从第2行开始（第1行是表头）
                    for cam_id in sorted(best_frames_per_cam.keys()):
                        best_frame = best_frames_per_cam[cam_id]
                        old_path = best_frame.get('image_path_old', '')
                        new_path = best_frame.get('image_path_new', '')
                        gt_path = best_frame.get('image_path_gt', '')
                        frame_idx = best_frame.get('frame_idx', 'N/A')
                        
                        # 写入基本信息
                        ws.cell(row=row_idx, column=1, value=f"Cam {cam_id}")
                        ws.cell(row=row_idx, column=2, value=frame_idx)
                        ws.cell(row=row_idx, column=3, value=f"{best_frame.get('improvement_score', 0):.6f}")
                        
                        # 设置行高
                        ws.row_dimensions[row_idx].height = row_height
                        
                        # 嵌入旧模型图片（尺寸为原始的一半）
                        if old_path and os.path.exists(old_path):
                            try:
                                img = OpenpyxlImage(old_path)
                                # 设置图片尺寸为原始的一半
                                img.width = target_img_width
                                img.height = target_img_height
                                # 插入到D列
                                ws.add_image(img, f'D{row_idx}')
                            except Exception as e:
                                logger.warning(f"无法嵌入 {dir1_name} 图像 (Cam {cam_id}): {e}")
                                ws.cell(row=row_idx, column=4, value="图片加载失败")
                        
                        # 嵌入新模型图片（尺寸为原始的一半）
                        if new_path and os.path.exists(new_path):
                            try:
                                img = OpenpyxlImage(new_path)
                                # 设置图片尺寸为原始的一半
                                img.width = target_img_width
                                img.height = target_img_height
                                # 插入到E列
                                ws.add_image(img, f'E{row_idx}')
                            except Exception as e:
                                logger.warning(f"无法嵌入 {dir2_name} 图像 (Cam {cam_id}): {e}")
                                ws.cell(row=row_idx, column=5, value="图片加载失败")
                        
                        # 嵌入GT图片（尺寸为原始的一半）
                        if gt_path and os.path.exists(gt_path):
                            try:
                                img = OpenpyxlImage(gt_path)
                                # 设置图片尺寸为原始的一半
                                img.width = target_img_width
                                img.height = target_img_height
                                # 插入到F列
                                ws.add_image(img, f'F{row_idx}')
                            except Exception as e:
                                logger.warning(f"无法嵌入 GT 图像 (Cam {cam_id}): {e}")
                                ws.cell(row=row_idx, column=6, value="图片加载失败")
                        else:
                            ws.cell(row=row_idx, column=6, value="N/A")
                        
                        row_idx += 1
                    
                    wb.save(best_frames_xlsx)
                    logger.info(f"已保存最佳改进帧表格到本地（Excel格式，图片尺寸为原始的一半，列宽匹配图片宽度，字段与wandb一致）: {best_frames_xlsx}")

        except Exception as e:
            logger.error(f"保存本地结果时出错: {e}", exc_info=True)
    
    logger.info(f"\n{'='*80}")
    logger.info("对比完成！")
    if args.enable_wandb:
        logger.info(f"结果已上传到 wandb: {wandb.run.url}")
    
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compare two training evaluation results")
    
    # 输入路径
    parser.add_argument("--log_dir1", type=str, required=True, help="First training log directory")
    parser.add_argument("--log_dir2", type=str, required=True, help="Second training log directory")
    
    # wandb 配置
    parser.add_argument("--enable_wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default=None, help="wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")
    
    # 其他参数
    parser.add_argument("--num_cams", type=int, default=11, help="Number of cameras")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main(args)

