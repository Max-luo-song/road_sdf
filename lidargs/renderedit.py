"""
renderedit.py — LiDAR 360° 渲染 + 静态障碍物编辑

基于 render360.py，新增对 edit_config.yaml 中 Nodes.Background.add
的支持，实现沿自车轨迹按帧或按距离批量插入静态障碍物（如锥桶、水马）
并输出编辑后点云。

用法：
    python lidargs/renderedit.py \\
        --model_path <scene_path> \\
        --source_path <data_path> \\
        --output_path <out_dir> \\
        --edit_config edit_config.yaml \\
        [--edit_config edit_config.yaml]

edit_config.yaml 格式（支持 Background.add + time_window + distance_interval_m）：
    Nodes:
      Background:
        add:
          add_obj:
          - "path/to/barrier.ply"
          offset:
          - [5, -2.1, -1.2]   # [x前后, y左右, z上下] 相对自车位置偏移
          time_window:
          - [0, -1, 10]        # [起始帧, 结束帧(-1=末尾), 间隔帧]
          distance_interval_m:
          - 5.0                # 可选，按轨迹累计距离采样；设置后优先于 time_window[2]
      RigidNodes:              # 可选
        remove:
          instance_id: [1]
        trajectory:
          instance_id: [1]
          offset: [-10, -0.7, 0]

输出：
    <output_path>/renders_edit/           — 编辑后点云 TXT/PCD
    <output_path>/renders_edit/barrier_instance_bboxes.json — 障碍物 bbox 信息
"""

import os
import json
import torch
import numpy as np
import sys
import subprocess
import shutil
import cv2
from plyfile import PlyData
from typing import List, Dict, Optional

cmd = "nvidia-smi -q -d Memory |grep -A4 GPU|grep Used"
result = (
    subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split("\n")
)
os.environ["CUDA_VISIBLE_DEVICES"] = str(
    np.argmin([int(x.split()[2]) for x in result[:-1]])
)
os.system("echo $CUDA_VISIBLE_DEVICES")

from scene import Scene
import time
import yaml
from gaussian_renderer import renderComposite
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams
from scene.gaussian_model import GaussianModel
from scene import Scene
from scene.unet import UNet

from utils.lidar_utils import (
    pano_to_lidar_with_intensities,
    pano_to_lidar_with_intensities_torch,
    filter_pcd,
    find_closest_labels,
)
from scene.cameras import Camera
import copy
import logging
from scipy.spatial.transform import Rotation

logger = logging.getLogger("render360edit")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

cpu_count = os.cpu_count()
torch.set_num_threads(cpu_count)

from typing import NamedTuple


class GaussianView(NamedTuple):
    gaussians: GaussianModel
    scene: Scene
    time_poses: dict


class TimePose(NamedTuple):
    camera_pose: np.array
    view: Camera
    valid_mask: np.array
    gt_mask: torch.Tensor


class ValidModeInfo(NamedTuple):
    model_id: int
    model_pose: np.array
    model_view: Camera
    model_gaussians: GaussianModel
    need_train: bool
    model_world_rotation: torch.Tensor = None
    model_world_translation: torch.Tensor = None
    model_world_quaternion: torch.Tensor = None


class EditObjInfo(NamedTuple):
    delete_obj_ids: list
    obj_id_offset_pairs: list
    add_id_path_pairs: list
    novel_poses: list


def remap_instance_id_to_model_id(gt_dynamic_model, instance_id):
    if hasattr(gt_dynamic_model, "instance_id_to_model_id"):
        return gt_dynamic_model.instance_id_to_model_id(instance_id)
    return int(instance_id)


def remap_edit_obj_info_ids(gt_dynamic_model, edit_obj_info):
    if edit_obj_info is None:
        return edit_obj_info
    return EditObjInfo(
        delete_obj_ids=[remap_instance_id_to_model_id(gt_dynamic_model, x) for x in edit_obj_info.delete_obj_ids],
        obj_id_offset_pairs=[
            [remap_instance_id_to_model_id(gt_dynamic_model, pair[0]), pair[1]]
            for pair in edit_obj_info.obj_id_offset_pairs
        ],
        add_id_path_pairs=edit_obj_info.add_id_path_pairs,
        novel_poses=edit_obj_info.novel_poses,
    )


def resolve_unet_ckpt_path(model_path: str) -> Optional[str]:
    ckpt_dir = os.path.join(model_path, "ckpt")
    for ckpt_name in ("unet_refine.pth", "refine.pth"):
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        if os.path.exists(ckpt_path):
            return ckpt_path
    return None


def build_world_transform_tensors(model_to_world: np.ndarray):
    model_world_rotation = torch.tensor(
        model_to_world[:3, :3], dtype=torch.float32, device="cuda"
    )
    model_world_translation = torch.tensor(
        model_to_world[:3, 3], dtype=torch.float32, device="cuda"
    )
    model_world_quaternion_np = Rotation.from_matrix(model_to_world[:3, :3]).as_quat()
    model_world_quaternion = torch.tensor(
        [
            model_world_quaternion_np[3],
            model_world_quaternion_np[0],
            model_world_quaternion_np[1],
            model_world_quaternion_np[2],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    return model_world_rotation, model_world_translation, model_world_quaternion


def attach_render_postprocess_args(model_args, args):
    model_args.enable_raydrop_postfilter = args.enable_raydrop_postfilter
    model_args.enable_depth_postfilter = args.enable_depth_postfilter
    model_args.raydrop_postfilter_threshold = args.raydrop_postfilter_threshold
    model_args.depth_postfilter_ratio_threshold = (
        args.depth_postfilter_ratio_threshold
    )
    model_args.near_preserve_depth = args.near_preserve_depth
    model_args.fast_render = bool(getattr(args, "fast_render", False))
    model_args.no_txt = bool(getattr(args, "no_txt", False))
    model_args.no_video = bool(getattr(args, "no_video", False))
    return model_args


def apply_render_postprocess(
    render_pkg,
    render_intensity,
    depth,
    render_raydrop,
    raydrop_unet_available,
    unet,
    dataset,
):
    near_preserve_depth = max(getattr(dataset, "near_preserve_depth", 0.0), 0.0)
    raw_depth = render_pkg["depth"]
    near_preserve_mask = (raw_depth > 0.0) & (raw_depth < near_preserve_depth)

    if getattr(dataset, "enable_raydrop_postfilter", False):
        raydrop_threshold = getattr(dataset, "raydrop_postfilter_threshold", 0.5)
        if raydrop_unet_available:
            # 4 channels: [raydrop, intensity, depth/80.0, mid_depth_diff]
            # 与 refine_ray_drop.py 训练时的通道顺序保持一致
            mid_depth_diff = render_pkg["mid_depth_diff"]
            unet_input = torch.cat(
                [
                    render_raydrop,
                    render_intensity,
                    raw_depth / 80.0,
                    mid_depth_diff,
                ],
                dim=0,
            ).unsqueeze(0)  # (1, 4, H, W)
            with torch.no_grad():
                raydrop_score = unet(unet_input).squeeze(0)  # (1, H, W)
        else:
            raydrop_score = render_raydrop
        render_raydrop_mask = (
            (raydrop_score > raydrop_threshold) | near_preserve_mask
        ).to(render_intensity.dtype)
        render_intensity = render_intensity * render_raydrop_mask
        depth = depth * render_raydrop_mask

    if getattr(dataset, "enable_depth_postfilter", True):
        depth_distortion_aware = render_pkg["mid_depth_diff"]
        depth_val = raw_depth.clamp(min=0.5)
        relative_distortion = depth_distortion_aware / depth_val
        depth_ratio_threshold = getattr(
            dataset, "depth_postfilter_ratio_threshold", 0.10
        )
        depth_distortion_mask = (
            (relative_distortion < depth_ratio_threshold) | near_preserve_mask
        ).to(render_intensity.dtype)
        render_intensity = render_intensity * depth_distortion_mask
        depth = depth * depth_distortion_mask

    return render_intensity, depth


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(path, log_path):
    logger = logging.getLogger("render360edit")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_file = os.path.join(log_path, "outputs_edit.log")
    has_file = any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", None) == os.path.abspath(log_file)
        for h in logger.handlers
    )
    if not has_file:
        fileinfo = logging.FileHandler(log_file)
        fileinfo.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
        fileinfo.setFormatter(formatter)
        logger.addHandler(fileinfo)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        controlshow = logging.StreamHandler()
        controlshow.setLevel(logging.INFO)
        controlshow.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
        )
        logger.addHandler(controlshow)

    return logger


# ---------------------------------------------------------------------------
# PCD I/O
# ---------------------------------------------------------------------------

def save_pcd_binary(file_path: str, points: np.ndarray):
    """Save Nx>=4 array to PCD binary (fields x y z intensity)."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 4:
        raise ValueError("points must be a (N, >=4) array with x,y,z,intensity columns")

    N = pts.shape[0]
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {N}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {N}\n"
        "DATA binary\n"
    )
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(header.encode("ascii"))
        data = pts[:, :4].astype(np.float32)
        f.write(data.tobytes())


def _make_bev_video(
    render_outputs: Dict,
    out_path: str,
    bev_xlim=(-60.0, 60.0),
    bev_ylim=(-50.0, 50.0),
    panel_w: int = 1200,
    panel_h: int = 900,
    fps: int = 10,
    pt_size: float = 2.0,
):
    """
    从 render_outputs 内存数据直接生成 GT vs Rendered 并排 BEV 视频。

    - 自车中心跟随模式：每帧坐标轴以 ego_pos 为中心，bev_xlim/bev_ylim 为相对偏移（米）。
      例如 bev_xlim=(-60, 60) 表示视野为自车前后各 60 m。
    - 不写任何临时文件，canvas.buffer_rgba() 直接内存取帧
    - 白色背景，viridis 强度着色
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timestamps = sorted(render_outputs.keys(), key=lambda k: int(k))
    if not timestamps:
        logger.warning("[BEV] render_outputs 为空，跳过视频生成")
        return

    logger.info(
        f"[BEV] 自车中心跟随模式  xlim_rel={bev_xlim}  ylim_rel={bev_ylim}  "
        f"{len(timestamps)} 帧  fps={fps}"
    )

    # 诊断：打印第一帧渲染/GT 点云数量，确认数据是否到达 BEV 函数
    _ts0 = timestamps[0]
    _r = render_outputs[_ts0].get("rendered")
    _g = render_outputs[_ts0].get("gt")
    _n_ren = len(_r) if _r is not None else 0
    _n_gt  = len(_g) if _g is not None else 0
    _n_cone = int(np.sum(_r[:, 3] >= 0.99)) if _r is not None and _n_ren > 0 else 0
    logger.info(
        f"[BEV 诊断] frame={_ts0}: GT 点数={_n_gt}, Rendered 点数={_n_ren}, "
        f"其中高强度(>=0.99)点={_n_cone} (锥桶注入点约估)"
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (panel_w * 2, panel_h))

    for ts in timestamps:
        gt_pts  = render_outputs[ts].get("gt")
        ren_pts = render_outputs[ts].get("rendered")

        # 以 ego_pos 为中心的绝对坐标轴范围
        ego_pos = render_outputs[ts].get("ego_pos", np.zeros(3))
        ego_x, ego_y = float(ego_pos[0]), float(ego_pos[1])
        # 横轴=横向Y：Y 正方向=车辆左侧，需反转使左侧在图像左边
        frame_xlim = (ego_y + bev_ylim[1], ego_y + bev_ylim[0])
        frame_ylim = (ego_x + bev_xlim[0], ego_x + bev_xlim[1])   # 纵轴=行驶X

        fig, axs = plt.subplots(
            1, 2,
            figsize=(panel_w * 2 / 100, panel_h / 100),
            dpi=100,
        )
        fig.patch.set_facecolor("white")

        for ax, pts, title in zip(
            axs,
            [gt_pts, ren_pts],
            ["GT", "Rendered (edit)"],
        ):
            ax.set_facecolor("white")
            ax.set_xlim(frame_xlim)
            ax.set_ylim(frame_ylim)
            #ax.set_aspect("equal", adjustable="box")
            ax.set_aspect("auto") # aspect="auto"：撑满面板，不强制等比例缩放
            ax.axis("off")
            ax.set_title(title, fontsize=13, pad=6, color="#222222")
            if pts is not None and len(pts) > 0:
                ax.scatter(
                    #pts[:, 0], pts[:, 1],
                    pts[:, 1], pts[:, 0],   # 横=Y(左右)，纵=X(前进方向)
                    c=np.clip(pts[:, 3], 0.0, 1.0),
                    s=pt_size,
                    cmap="jet",
                    vmin=0.0, vmax=1.0,
                    linewidths=0,
                    rasterized=True,
                )
            # 两个面板都标注自车位置
            # ax.plot(ego_x, ego_y, "r+", markersize=10, markeredgewidth=1.5)
            # 两个面板都标注自车位置（横=ego_y，纵=ego_x）
            ax.plot(ego_y, ego_x, "r+", markersize=10, markeredgewidth=1.5)



        fig.tight_layout(pad=0.4)
        fig.canvas.draw()

        # 内存直取 RGBA buffer，无需写磁盘临时文件
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        rgba = rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        bgr  = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
        bgr  = cv2.resize(bgr, (panel_w * 2, panel_h), interpolation=cv2.INTER_LINEAR)
        vw.write(bgr)
        plt.close(fig)

    vw.release()
    logger.info(f"[BEV] 视频已保存: {out_path}")


# ---------------------------------------------------------------------------
# Trajectory helper
# ---------------------------------------------------------------------------

def _inject_into_pano(
    depth_pano: np.ndarray,
    intens_pano: np.ndarray,
    world_xyz: np.ndarray,
    world_to_cam: np.ndarray,
    beam_inclinations: np.ndarray,
    W: int,
    lidar_hfov: float,
    max_depth: float,
    intensity_val: float = 1.0,
    sample_ratio: float = 1.0,
) -> int:
    """
    将 PLY 世界坐标点注入深度全景图（in-place），返回实际影响的像素数。

    在全景图层操作而非 3D 点云层，因此：
      - 遮挡自动正确：锥桶深度 < 背景深度 → 锥桶覆盖该 beam slot
      - 背后扫描缺失自动正确：被锥桶占据的 beam×azimuth 格子不再有背景深度
      - 后续 pano_to_lidar_with_intensities 产生的点云天然包含两种效果
    """
    if len(world_xyz) == 0:
        return 0

    H = depth_pano.shape[0]
    M = len(world_xyz)

    # 可选降采样（在投影前做，减少计算量）
    if sample_ratio < 1.0:
        n_keep = max(1, int(M * sample_ratio))
        idx_s = np.random.choice(M, n_keep, replace=False)
        world_xyz = world_xyz[idx_s]
        M = len(world_xyz)

    hom = np.concatenate([world_xyz, np.ones((M, 1), dtype=np.float32)], axis=1)
    local_xyz = (hom @ world_to_cam.T)[:, :3]

    x, y, z = local_xyz[:, 0], local_xyz[:, 1], local_xyz[:, 2]
    dists = np.linalg.norm(local_xyz, axis=1).astype(np.float64)

    valid = dists < max_depth
    if not np.any(valid):
        return 0
    x, y, z, dists = x[valid], y[valid], z[valid], dists[valid]

    # 方位角列
    beta = lidar_hfov / 2 - np.arctan2(y, x)
    c = np.rint(beta / (lidar_hfov / W)).astype(np.int32)

    # 仰角 → 最近 beam
    alpha = np.arctan2(z, np.sqrt(x ** 2 + y ** 2 + z ** 2))
    beam_idx = find_closest_labels(beam_inclinations, alpha)
    r = H - 1 - beam_idx

    in_bounds = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    r, c, dists = r[in_bounds], c[in_bounds], dists[in_bounds]
    if len(r) == 0:
        return 0

    # 每个 flat cell 取最近 PLY 点距离（向量化）
    flat = r * W + c
    min_ply = np.full(H * W, np.inf, dtype=np.float64)
    np.minimum.at(min_ply, flat, dists)

    # 更新条件：有 PLY 点，且（背景无回波 OR PLY 比背景近）
    has_ply     = np.isfinite(min_ply)
    depth_flat  = depth_pano.reshape(-1)
    intens_flat = intens_pano.reshape(-1)
    update      = has_ply & ((depth_flat == 0.0) | (min_ply < depth_flat))

    depth_flat[update]  = min_ply[update].astype(np.float32)
    intens_flat[update] = intensity_val

    return int(update.sum())


def compute_ego_trajectory(l2ws) -> np.ndarray:
    """
    从 gt_dynamic_model.l2ws（sensor_lidar2world 矩阵列表）中提取自车世界坐标轨迹。

    直接使用 l2ws（原始、未经 novel_pose 偏移修改的 sl2w 变换）提取平移分量，
    而不依赖 view.R/view.T（后者可能含有 novel_poses_setting 引入的平移偏移）。

    坐标系说明：
        lidargs 世界坐标系 = inv(lidar_to_world_start) @ lidar_to_world_current，
        即以第一帧 baselidar 位姿为原点、轴向与第一帧车辆前/左/上方向对齐的相对世界系。
        offset: [x, y, z] 近似对应 [前后, 左右, 上下]（在车辆初始朝向下有效）。

    Args:
        l2ws: gt_dynamic_model.l2ws，每帧的 sensor_lidar2world (sl2w) 矩阵列表，
              形状 (N,) of (4,4) numpy arrays。

    Returns:
        (N, 3) numpy 数组，每行为对应帧传感器 LiDAR 在世界坐标系中的位置。
    """
    return np.array([m[:3, 3] for m in l2ws], dtype=np.float64)  # (N, 3)

def compute_cam_trajectory_from_source(
    scene_l2ws: List[np.ndarray],
    source_path: str,
    front_cam_id: int = 0,
) -> Optional[np.ndarray]:
    """
    从外参文件和 LiDAR poses 计算前置相机世界轨迹。

    与相机编辑侧 front_camera_trajectory()（= cam_to_worlds[:, :3, 3]）对齐，
    使障碍物以相机光心位置为参考点，消除 LiDAR 传感器与相机光心之间的空间偏差
    （通常为前后 0.x～2 m、高度 0.x～0.5 m），从而使 LiDAR 点云中的锥桶位置
    与相机编辑视频中的锥桶位置在世界坐标系中完全一致。

    cam_to_world[t] = scene_l2ws[t] @ cam_to_lidar
    cam_position[t] = cam_to_world[t][:3, 3]

    Args:
        scene_l2ws:   每帧的 lidar-to-world 4×4 矩阵列表（相对于 block 0 起始帧）
        source_path:  数据集路径（含 extrinsics/{cam_id}.txt）
        front_cam_id: 前置相机 ID（默认 0）

    Returns:
        (N, 3) 相机世界坐标轨迹，若外参文件不存在则返回 None（调用方回退到 LiDAR 轨迹）。
    """
    extr_path = os.path.join(source_path, "extrinsics", f"{front_cam_id}.txt")
    if not os.path.exists(extr_path):
        logger.warning(
            f"[cam_trajectory] 未找到前置相机外参 {extr_path}，"
            "回退到 LiDAR 轨迹（障碍物摆放参考点将与相机编辑存在偏差）"
        )
        return None
    cam_to_lidar = np.loadtxt(extr_path).astype(np.float64).reshape(4, 4)
    logger.info(
        f"[cam_trajectory] 已加载外参 {extr_path}  "
        f"cam_in_lidar_frame=[{cam_to_lidar[0,3]:.3f}, "
        f"{cam_to_lidar[1,3]:.3f}, {cam_to_lidar[2,3]:.3f}]"
    )
    positions = []
    for l2w in scene_l2ws:
        cam_to_world = np.asarray(l2w, dtype=np.float64) @ cam_to_lidar
        positions.append(cam_to_world[:3, 3])
    return np.array(positions, dtype=np.float64)


def load_relative_lidar_poses(source_path: str, frame_ids: List[int]) -> Dict[int, np.ndarray]:
    """Load per-frame lidar poses and normalize them to the first frame."""
    pose_dir = os.path.join(source_path, "lidar_pose")
    if not os.path.isdir(pose_dir):
        raise FileNotFoundError(f"lidar_pose 目录不存在: {pose_dir}")
    if not frame_ids:
        raise ValueError("frame_ids 不能为空")

    ordered_frame_ids = [int(frame_id) for frame_id in frame_ids]
    base_pose_path = os.path.join(pose_dir, f"{ordered_frame_ids[0]:06d}.txt")
    base_pose = np.loadtxt(base_pose_path).reshape(4, 4)

    relative_poses = {}
    inv_base_pose = np.linalg.inv(base_pose)
    for frame_id in ordered_frame_ids:
        pose_path = os.path.join(pose_dir, f"{int(frame_id):06d}.txt")
        if not os.path.exists(pose_path):
            raise FileNotFoundError(f"缺少 lidar pose: {pose_path}")
        current_pose = np.loadtxt(pose_path).reshape(4, 4)
        relative_poses[int(frame_id)] = inv_base_pose @ current_pose
    return relative_poses


def load_training_cfg_args(model_path: str) -> Optional[Namespace]:
    """Read lidargs cfg_args from the training output directory when available."""
    cfg_args_path = os.path.join(model_path, "cfg_args")
    if not os.path.exists(cfg_args_path):
        return None
    with open(cfg_args_path, "r") as f:
        content = f.read().strip()
    if not content:
        return None
    try:
        return eval(content, {"Namespace": Namespace})
    except Exception as exc:
        logger.warning(f"[cfg_args] 解析失败 {cfg_args_path}: {exc}")
        return None


def align_args_with_training_output(args):
    """Back-fill source_path / caseid from cfg_args to keep render aligned with training."""
    cfg_args = load_training_cfg_args(args.model_path)
    if cfg_args is None:
        return

    if not getattr(args, "source_path", None):
        cfg_source_path = getattr(cfg_args, "source_path", None)
        if cfg_source_path:
            args.source_path = cfg_source_path
            logger.info(f"[cfg_args] 自动对齐 source_path={args.source_path}")

    current_caseid = getattr(args, "caseid", None)
    if current_caseid in (None, "", "Nothing"):
        cfg_caseid = getattr(cfg_args, "caseid", None)
        if cfg_caseid:
            args.caseid = cfg_caseid
            logger.info(f"[cfg_args] 自动对齐 caseid={args.caseid}")


def normalize_scene_range(window, num_frames: int):
    """Normalize [start, end, step] against the ordered scene frame list."""
    assert num_frames >= 0
    if not isinstance(window, (list, tuple)) or len(window) == 0:
        return 0, num_frames, 1

    start = int(window[0])
    end = int(window[1]) if len(window) > 1 else num_frames
    step = int(window[2]) if len(window) > 2 else 1

    if end == -1:
        end = num_frames

    start = max(0, min(start, num_frames))
    end = max(start, min(end, num_frames))
    step = max(step, 1)
    return start, end, step


def sample_scene_indices_by_distance(
    ego_trajectory: np.ndarray,
    start_idx: int,
    end_idx: int,
    spacing_m: float,
) -> List[int]:
    """Sample trajectory positions by accumulated travel distance within [start_idx, end_idx)."""
    if end_idx <= start_idx:
        return []
    if spacing_m <= 0:
        return list(range(start_idx, end_idx))

    sampled = [start_idx]
    accumulated = 0.0
    for scene_idx in range(start_idx + 1, end_idx):
        accumulated += float(
            np.linalg.norm(ego_trajectory[scene_idx] - ego_trajectory[scene_idx - 1])
        )
        if accumulated + 1e-6 >= spacing_m:
            sampled.append(scene_idx)
            accumulated = 0.0
    return sampled


# ---------------------------------------------------------------------------
# PLY → insert_obj
# ---------------------------------------------------------------------------

def read_single_ply_to_obj(
    ply_file_path: str,
    obj_class: str = None,
    sim_pose: Optional[torch.Tensor] = None,
    scale_factor: float = 1.0,
) -> Optional[Dict[str, torch.Tensor]]:
    """读取单个 PLY 文件，转换为 renderComposite 所需的 obj 字典。"""
    try:
        ply_data = PlyData.read(ply_file_path)
        vertices = ply_data["vertex"]
        num_vertices = len(vertices)
    except Exception as e:
        logger.warning(f"[read_single_ply_to_obj] 读取失败 {ply_file_path}: {e}")
        return None

    xyz = torch.tensor(
        np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1),
        dtype=torch.float32,
        device="cuda",
    )
    if scale_factor != 1.0:
        xyz = xyz * scale_factor # xyz也会随着scale_factor放大，保持与scaling一致
    opacity = torch.tensor(
        vertices["opacity"], dtype=torch.float32, device="cuda"
    ).unsqueeze(1)
    opacity = torch.clamp(opacity, 0.0, 1.0)
    opacity[:] = 1.0

    scaling = torch.tensor(
        np.stack(
            [vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]], axis=1
        ),
        dtype=torch.float32,
        device="cuda",
    )
    scaling = torch.exp(scaling) * scale_factor

    rot = torch.tensor(
        np.stack(
            [vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"]],
            axis=1,
        ),
        dtype=torch.float32,
        device="cuda",
    )

    # LiDAR 颜色: [intensity=1.0, raydrop=1.0]
    color = torch.ones((num_vertices, 2), dtype=torch.float32, device="cuda")

    if sim_pose is not None:
        if isinstance(sim_pose, (np.ndarray, list)):
            sim_pose = torch.tensor(
                sim_pose, dtype=torch.float32, device="cuda"
            ).reshape(4, 4)
        else:
            sim_pose = sim_pose.to(device="cuda", dtype=torch.float32).reshape(4, 4)

    return {
        "obj_class": obj_class,
        "xyz": xyz,
        "color": color,
        "opacity": opacity,
        "scaling": scaling,
        "rot": rot,
        "sim_pose": sim_pose,
    }


def batch_read_plys_to_insert_objs(
    edit_obj_info: EditObjInfo,
) -> List[Dict[str, torch.Tensor]]:
    """批量读取 EditObjInfo.add_id_path_pairs 中的 PLY，生成 insert_objs 列表。"""
    insert_objs = []
    for obj_class, sim_pose, obj_pcd_filepath in edit_obj_info.add_id_path_pairs:
        single_obj = read_single_ply_to_obj(obj_pcd_filepath, obj_class, sim_pose)
        if single_obj is not None:
            insert_objs.append(single_obj)
    return insert_objs


# ---------------------------------------------------------------------------
# Edit config parsing (新格式: Background.add + time_window)
# ---------------------------------------------------------------------------

def parse_and_apply_edit_yaml(
    args,
    edit_config_path: str = None,
    train_frame_times: list = None,
) -> tuple:
    """
    解析 edit_config.yaml（新格式，支持 Background.add + time_window），
    同时兼容 RigidNodes 的 remove / trajectory / add 操作。

    Returns:
        (insert_objs, edit_obj_info, barrier_placements)
    """
    if not hasattr(args, "novel_poses"):
        args.novel_poses = []

    yaml_path = edit_config_path
    if yaml_path is None or not os.path.exists(yaml_path):
        if yaml_path is not None:
            logger.warning(f"edit_config 文件不存在: {yaml_path}")
        return [], EditObjInfo([], [], [], []), []

    scene_pose_map = load_relative_lidar_poses(args.source_path, train_frame_times)
    scene_l2ws = [scene_pose_map[frame_id] for frame_id in train_frame_times]
    ego_trajectory = compute_ego_trajectory(scene_l2ws)
    cam_trajectory = compute_cam_trajectory_from_source(scene_l2ws, args.source_path)
    if cam_trajectory is not None:
        ref_trajectory = cam_trajectory
        logger.info(
            f"[edit_config] 使用前置相机轨迹摆放障碍物（{len(train_frame_times)} 帧），"
            "与相机编辑侧对齐"
        )
    else:
        ref_trajectory = ego_trajectory
        logger.info(
            f"[edit_config] 使用 LiDAR 轨迹摆放障碍物（{len(train_frame_times)} 帧），"
            "相机外参不可用"
        )

    ego_trajectory = ref_trajectory  # cam_trajectory 优先，否则 lidar ego_trajectory
    edit_frame_ids = train_frame_times

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    nodes = cfg.get("Nodes", {}) or {}
    if len(ego_trajectory) != len(edit_frame_ids):
        raise ValueError(
            "ego_trajectory 与 edit_frame_ids 长度不一致: "
            f"{len(ego_trajectory)} vs {len(edit_frame_ids)}"
        )

    N = len(ego_trajectory)

    # ------------------------------------------------------------------
    # Background.add — time_window 静态障碍物
    # ------------------------------------------------------------------
    insert_objs: List[Dict] = []
    barrier_placements: List[Dict] = []

    bg = nodes.get("Background", {}) or {}
    add_bg = bg.get("add", {}) or {}
    add_objs_bg = add_bg.get("add_obj", []) or []
    offsets_bg = add_bg.get("offset", []) or []
    time_windows = add_bg.get("time_window", []) or []
    scale_factors = add_bg.get("lidar_scale_factor", []) or []
    if isinstance(scale_factors, (int, float)):
        scale_factors = [float(scale_factors)]
    distance_intervals = add_bg.get(
        "distance_interval_m",
        add_bg.get("distance_interval", add_bg.get("spacing_m", [])),
    )
    if isinstance(distance_intervals, (int, float)):
        distance_intervals = [float(distance_intervals)]
    distance_intervals = distance_intervals or []

    inject_direct_flags = add_bg.get("inject_direct_points", []) or []
    if isinstance(inject_direct_flags, bool):
        inject_direct_flags = [inject_direct_flags]
    inject_sample_ratios = add_bg.get("inject_sample_ratio", []) or []
    if isinstance(inject_sample_ratios, (int, float)):
        inject_sample_ratios = [float(inject_sample_ratios)]
    rotations_bg = add_bg.get("rotation", []) or []

    for i, ply_path in enumerate(add_objs_bg):
        offset = offsets_bg[i] if i < len(offsets_bg) else [0.0, 0.0, 0.0]
        tw = time_windows[i] if i < len(time_windows) else [0, -1, 10]
        scale_factor = float(scale_factors[i]) if i < len(scale_factors) else 1.0
        inject_direct = bool(inject_direct_flags[i]) if i < len(inject_direct_flags) else True
        inject_sample_ratio = float(inject_sample_ratios[i]) if i < len(inject_sample_ratios) else 1.0
        rot_deg = rotations_bg[i] if i < len(rotations_bg) else [0.0, 0.0, 0.0]
        rot_matrix = Rotation.from_euler("XYZ", rot_deg, degrees=True).as_matrix().astype(np.float32)
        start_idx, end_idx, step = normalize_scene_range(tw, N)
        spacing_m = (
            float(distance_intervals[i])
            if i < len(distance_intervals) and distance_intervals[i] is not None
            else None
        )

        if spacing_m is not None:
            selected_scene_indices = sample_scene_indices_by_distance(
                ego_trajectory,
                start_idx,
                end_idx,
                spacing_m,
            )
        else:
            selected_scene_indices = list(range(start_idx, end_idx, step))

        offset_z = float(offset[2] if len(offset) > 2 else 0)

        logger.info(f"offset_yaml={offset}  offset_z_final={offset_z:.2f}")
        if selected_scene_indices:
            _first = ego_trajectory[selected_scene_indices[0]]
            logger.info(
                f"[barrier z debug] ego_pos (first frame, block-local world): "
                f"x={_first[0]:.3f}, y={_first[1]:.3f}, z={_first[2]:.3f}  "
                f"final_z={_first[2] + offset_z:.3f}"
            )

        for scene_idx in selected_scene_indices:
            ego_pos = ego_trajectory[scene_idx]
            raw_frame_id = int(edit_frame_ids[scene_idx])

            # sim_pose: 先旋转 PLY 本地坐标，再平移到 ego_pos + offset
            sim_pose = np.eye(4, dtype=np.float32)
            sim_pose[:3, :3] = rot_matrix
            sim_pose[0, 3] = float(ego_pos[0]) + float(offset[0])
            sim_pose[1, 3] = float(ego_pos[1]) + float(offset[1])
            sim_pose[2, 3] = float(ego_pos[2]) + offset_z

            obj = read_single_ply_to_obj(ply_path, obj_class="barrier", sim_pose=sim_pose, scale_factor=scale_factor)
            if obj is None:
                continue

            obj["inject_direct"] = inject_direct
            obj["inject_sample_ratio"] = inject_sample_ratio
            insert_objs.append(obj)

            # 计算障碍物在世界坐标系中的 bbox（应用完整 sim_pose = R + T）
            xyz_local = obj["xyz"].cpu().numpy()       # (M, 3) PLY 本地坐标
            xyz_hom = np.concatenate([xyz_local, np.ones((len(xyz_local), 1), dtype=np.float32)], axis=1)
            xyz_world = (xyz_hom @ sim_pose.T)[:, :3]
            barrier_placements.append(
                {
                    "frame": scene_idx,
                    "frame_id": raw_frame_id,
                    "bbox_min": xyz_world.min(axis=0).tolist(),
                    "bbox_max": xyz_world.max(axis=0).tolist(),
                }
            )

    logger.info(
        f"[edit_config] Background.add: 共创建 {len(insert_objs)} 个静态障碍物副本"
    )

    # ------------------------------------------------------------------
    # RigidNodes — 兼容 remove / trajectory / add
    # ------------------------------------------------------------------
    rigid = nodes.get("RigidNodes", {}) or {}

    remove_section = rigid.get("remove", {}) or {}
    remove_ids = remove_section.get("instance_id", []) or []

    traj_section = rigid.get("trajectory", {}) or {}
    traj_ids     = traj_section.get("instance_id", []) or []
    traj_offsets = traj_section.get("offset", []) or []
    if traj_offsets and isinstance(traj_offsets[0], (int, float)):
        traj_offsets = [traj_offsets]
    if traj_offsets and len(traj_offsets) == 1 and len(traj_ids) > 1:
        traj_offsets = traj_offsets * len(traj_ids)
    move_pairs = [
        [iid, traj_offsets[j] if j < len(traj_offsets) else [0.0, 0.0, 0.0]]
        for j, iid in enumerate(traj_ids)
    ]

    add_section_rigid = rigid.get("add", {}) or {}
    ref_ids_rigid  = add_section_rigid.get("ref_id", []) or []
    add_objs_rigid = add_section_rigid.get("add_obj", []) or []
    add_offs_rigid = add_section_rigid.get("offset", []) or []
    maxlen = max(len(add_objs_rigid), len(ref_ids_rigid), len(add_offs_rigid), 0)
    add_pairs = []
    for j in range(maxlen):
        ref  = ref_ids_rigid[j]  if j < len(ref_ids_rigid)  else None
        path = add_objs_rigid[j] if j < len(add_objs_rigid) else None
        off  = add_offs_rigid[j] if j < len(add_offs_rigid) else None
        sim_pose_list = None
        if isinstance(off, list) and len(off) == 3:
            sim_pose_list = np.eye(4).tolist()
            sim_pose_list[0][3] = float(off[0])
            sim_pose_list[1][3] = float(off[1])
            sim_pose_list[2][3] = float(off[2])
        add_pairs.append((ref, sim_pose_list, path))

    edit_obj_info = EditObjInfo(
        delete_obj_ids=remove_ids,
        obj_id_offset_pairs=move_pairs,
        add_id_path_pairs=add_pairs,
        novel_poses=[],
    )

    return insert_objs, edit_obj_info, barrier_placements



# ---------------------------------------------------------------------------
# 动态对象编辑工具函数
# ---------------------------------------------------------------------------

def delete_specified_objs(model_id_list, edit_obj_info):
    if edit_obj_info is not None:
        for delete_id in edit_obj_info.delete_obj_ids:
            if delete_id in model_id_list:
                model_id_list.remove(delete_id)
    logger.info(f"model_id_list after delete: {model_id_list}")
    return model_id_list


def move_specified_objs(model_id, model_gaussians, edit_obj_info):
    transform_R = torch.eye(
        3,
        device=model_gaussians._anchor.device,
        dtype=model_gaussians._anchor.dtype,
    )
    if edit_obj_info is not None:
        for move_pair in edit_obj_info.obj_id_offset_pairs:
            if move_pair[0] == model_id:
                transform_T = torch.tensor(
                    move_pair[1],
                    device=model_gaussians._anchor.device,
                    dtype=model_gaussians._anchor.dtype,
                )
                new_anchor = (
                    model_gaussians._anchor @ transform_R.T
                ) + transform_T.reshape(1, 3)
                model_gaussians._anchor.data.copy_(new_anchor)
    return model_gaussians


# ---------------------------------------------------------------------------
# 核心渲染函数 (编辑版，输出到 renders_edit/)
# ---------------------------------------------------------------------------

def render_set_edit(
    gt_dynamic_model,
    dataset,
    name,
    iteration,
    valid_timestamp_model,
    model_id_scene_info,
    views,
    pipeline,
    background,
    insert_objs,
    inject_objs=None,
    block_coord_transform=None,
):
    """与 render360.py 中的 render_set 相同，但输出目录改为 renders_edit/。"""
    render_outputs = {}
    render_path = os.path.join(dataset.output_path, "renders_edit")
    gt_path     = os.path.join(dataset.output_path, "gt")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gt_path, exist_ok=True)

    t_list = []
    total_frames = len(views)
    original_l2ws = gt_dynamic_model.l2ws

    raydrop_unet_available = False
    unet = None
    if not getattr(dataset, "enable_raydrop_postfilter", False):
        logger.info("UNet raydrop postfilter 已禁用（未传 --enable_raydrop_postfilter），跳过加载 checkpoint。")
    else:
        ckpt_path = resolve_unet_ckpt_path(dataset.model_path)
        if ckpt_path is None:
            logger.warning(
                f"UNet checkpoint 不存在，跳过 refine。已检查: "
                f"{os.path.join(dataset.model_path, 'ckpt', 'unet_refine.pth')} / "
                f"{os.path.join(dataset.model_path, 'ckpt', 'refine.pth')}"
            )
        else:
            raydrop_unet_available = True
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            unet = UNet(in_channels=4, out_channels=1)
            state = torch.load(ckpt_path, map_location="cpu")
            unet.load_state_dict(state)
            unet.to(device)
            unet.eval()

    logger.info(
        "Render postprocess: raydrop=%s(th=%.3f), depth=%s(ratio=%.3f), near_preserve=%.2fm",
        getattr(dataset, "enable_raydrop_postfilter", False),
        getattr(dataset, "raydrop_postfilter_threshold", 0.5),
        getattr(dataset, "enable_depth_postfilter", True),
        getattr(dataset, "depth_postfilter_ratio_threshold", 0.10),
        getattr(dataset, "near_preserve_depth", 0.0),
    )

    # ------------------------------------------------------------------
    # fast_render: pre-cache + CUDA warmup（--fast_render 时启用）
    # ------------------------------------------------------------------
    use_fast_render = bool(getattr(dataset, "fast_render", False))
    skip_txt = bool(getattr(dataset, "no_txt", False))
    skip_gt = bool(getattr(dataset, "no_video", False))

    # hfov 在每帧不变，提前缓存
    lidar_hfov_cached = gt_dynamic_model.get_fov_horizontal()

    beam_incl_gpu_cache = None
    c2w_gpu_cache = {}
    bct_gpu = None

    if use_fast_render and views:
        # 预搬 beam_inclinations 到 GPU（所有帧共用同一组倾角）
        _beam_np = views[0].beam_inclinations.detach().cpu().numpy()
        beam_incl_gpu_cache = torch.from_numpy(_beam_np.astype("float32")).cuda()

        # 预搬每帧 c2w 矩阵到 GPU
        for _v in views:
            _wcp = np.eye(4)
            _wcp[:3, :3] = np.transpose(_v.R)
            _wcp[:3, 3] = _v.T
            c2w_gpu_cache[_v.image_name] = torch.from_numpy(
                np.linalg.inv(_wcp).astype("float32")
            ).cuda()

        # 预搬 block_coord_transform 到 GPU（如有）
        if block_coord_transform is not None:
            bct_gpu = torch.from_numpy(block_coord_transform.astype("float32")).cuda()

        logger.info(
            "[fast_render] 初始化完成: beam_gpu=%s, c2w缓存%d帧, bct_gpu=%s",
            beam_incl_gpu_cache is not None,
            len(c2w_gpu_cache),
            bct_gpu is not None,
        )

    total_render_start = time.time()
    for idx, view in enumerate(tqdm(views, desc="Rendering (edit) progress")):
        render_timestamp = view.image_name
        world_to_camera_pose = np.eye(4)
        world_to_camera_pose[:3, :3] = np.transpose(view.R)
        world_to_camera_pose[:3, 3] = view.T
        camera_to_world_pose = np.linalg.inv(world_to_camera_pose)

        valid_model_info = []
        for model_id in valid_timestamp_model[render_timestamp]:
            time_pose = model_id_scene_info[model_id].time_poses[render_timestamp]
            model_to_camera_pose = time_pose.camera_pose
            object_view = time_pose.view

            model_gaussian = model_id_scene_info[model_id].gaussians
            model_gaussian.eval()
            model_to_world = camera_to_world_pose @ model_to_camera_pose
            (
                model_world_rotation,
                model_world_translation,
                model_world_quaternion,
            ) = build_world_transform_tensors(model_to_world)
            valid_model_info.append(
                ValidModeInfo(
                    model_id=model_id,
                    model_pose=model_to_world,
                    model_view=object_view,
                    model_gaussians=model_gaussian,
                    need_train=False,
                    model_world_rotation=model_world_rotation,
                    model_world_translation=model_world_translation,
                    model_world_quaternion=model_world_quaternion,
                )
            )

        torch.cuda.synchronize()
        t0 = time.time()
        render_pkg = renderComposite(
            view,
            background,
            pipeline,
            valid_model_info,
            dataset.max_depth,
            insert_objs=insert_objs,
            retain_grad=False,
            composite_insert_render=True,
            use_multi_stream=True,
        )
        torch.cuda.synchronize()
        t1 = time.time()

        mem_allocated = torch.cuda.memory_allocated() / 1024 / 1024
        mem_reserved  = torch.cuda.memory_reserved()  / 1024 / 1024
        logger.info(
            f"[GPU] 当前帧显存: allocated={mem_allocated:.2f} MB, reserved={mem_reserved:.2f} MB"
        )
        t_list.append(t1 - t0)

        rendering = render_pkg["render"]
        render_intensity = rendering[0:1, ...]
        depth = render_pkg["depth"]

        gt = view.original_image.cuda()
        ray_drop = gt[0:1, ...]
        gt_intensity = (gt[1:2, ...] * ray_drop).detach().cpu().numpy()
        gt_depth     = (gt[2:3, ...] * ray_drop).detach().cpu().numpy()
        render_raydrop = rendering[1:2, ...]
        render_intensity, depth = apply_render_postprocess(
            render_pkg,
            render_intensity,
            depth,
            render_raydrop,
            raydrop_unet_available,
            unet if raydrop_unet_available else None,
            dataset,
        )

        depth_numpy     = depth.detach().cpu().numpy()
        intensity_numpy = render_intensity.detach().cpu().numpy()

        # ------------------------------------------------------------------
        # 全景图层注入（inject_direct_points=true 的对象）
        # 在 pano_to_lidar_with_intensities 之前直接修改 depth/intensity 全景图：
        #   - 锥桶深度 < 背景深度 → 覆盖该 beam×azimuth slot（正确遮挡）
        #   - 被锥桶占据的格子后方不再有背景深度（自然产生"扫描阴影"）
        # ------------------------------------------------------------------
        if inject_objs:
            _beam_incl = view.beam_inclinations.detach().cpu().numpy()
            _W         = view.image_width
            _hfov      = gt_dynamic_model.get_fov_horizontal()
            _d = depth_numpy[0].copy()
            _i = intensity_numpy[0].copy()
            total_px = 0
            for each_obj in inject_objs:
                if "sim_pose" not in each_obj:
                    continue
                obj_xyz = each_obj["xyz"]
                M_obj   = obj_xyz.shape[0]
                xyz_hom = torch.cat(
                    [obj_xyz, torch.ones(M_obj, 1, device=obj_xyz.device, dtype=obj_xyz.dtype)],
                    dim=1,
                )
                world_xyz = (xyz_hom @ each_obj["sim_pose"].T)[:, :3].cpu().numpy()
                total_px += _inject_into_pano(
                    _d, _i, world_xyz, world_to_camera_pose,
                    _beam_incl, _W, _hfov, dataset.max_depth,
                    intensity_val=1.0,
                    sample_ratio=float(each_obj.get("inject_sample_ratio", 1.0)),
                )
            depth_numpy[0]     = _d
            intensity_numpy[0] = _i
            logger.info(
                f"[inject_direct] frame={render_timestamp}: 影响 {total_px} 个全景像素"
            )

        # ------------------------------------------------------------------
        # pano→LiDAR + 世界坐标变换
        # fast_render=True 且无直接注入对象时：全程在 GPU 完成，末尾一次传输
        # 否则：原始 CPU numpy 路径（行为与原代码完全相同）
        # ------------------------------------------------------------------
        if use_fast_render and not inject_objs:
            _pts_local = pano_to_lidar_with_intensities_torch(
                depth[0],
                render_intensity[0],
                beam_inclinations_gpu=beam_incl_gpu_cache,
                lidar_hfov=lidar_hfov_cached,
            )
            _n_pts = _pts_local.shape[0]
            if _n_pts > 0:
                _c2w_mat = c2w_gpu_cache[render_timestamp]
                _hom = torch.cat(
                    [_pts_local[:, :3], _pts_local.new_ones(_n_pts, 1)], dim=1
                )
                _xyz_world = (_hom @ _c2w_mat.T)[:, :3]
                if bct_gpu is not None:
                    _hom_bct = torch.cat(
                        [_xyz_world, _pts_local.new_ones(_n_pts, 1)], dim=1
                    )
                    _xyz_world = (_hom_bct @ bct_gpu.T)[:, :3]
                _pts_world = torch.cat([_xyz_world, _pts_local[:, 3:4]], dim=1)
            else:
                _pts_world = torch.zeros((0, 4), dtype=torch.float32, device="cuda")
            point_with_intensity_world = _pts_world.detach().cpu().numpy()

            # GT：no_video 时跳过（节省一次 pano2lidar + matmul）
            if skip_gt:
                gt_point_with_intensity_world = np.zeros((0, 4), dtype=np.float32)
            else:
                original_l2w = original_l2ws[idx]
                gt_point_with_intensity_world = pano_to_lidar_with_intensities(
                    gt_depth[0, :, :],
                    gt_intensity[0],
                    lidar_K=None,
                    beam_inclinations=view.beam_inclinations.detach().cpu().numpy(),
                    lidar_hfov=lidar_hfov_cached,
                )
                gt_point_with_intensity_world[:, :3] = (
                    np.pad(
                        gt_point_with_intensity_world[:, :3],
                        ((0, 0), (0, 1)),
                        constant_values=1,
                    )
                    @ original_l2w.T
                )[:, :3]
                if block_coord_transform is not None:
                    gt_point_with_intensity_world[:, :3] = (
                        np.pad(
                            gt_point_with_intensity_world[:, :3],
                            ((0, 0), (0, 1)),
                            constant_values=1,
                        )
                        @ block_coord_transform.T
                    )[:, :3]
        else:
            # 原始 CPU 路径（fast_render=False，或有 inject_objs 时）
            lidar_hfov = gt_dynamic_model.get_fov_horizontal()
            logger.info(
                f"Using lidar_hfov = {lidar_hfov} rad ({lidar_hfov / np.pi * 180:.1f} deg)"
            )
            point_with_intensity = pano_to_lidar_with_intensities(
                depth_numpy[0, :, :],
                intensity_numpy[0],
                lidar_K=None,
                beam_inclinations=view.beam_inclinations.detach().cpu().numpy(),
                lidar_hfov=lidar_hfov,
            )
            gt_point_with_intensity = pano_to_lidar_with_intensities(
                gt_depth[0, :, :],
                gt_intensity[0],
                lidar_K=None,
                beam_inclinations=view.beam_inclinations.detach().cpu().numpy(),
                lidar_hfov=lidar_hfov,
            )

            # 转换到世界坐标系（block-local）
            point_with_intensity_world    = point_with_intensity
            gt_point_with_intensity_world = gt_point_with_intensity
            point_with_intensity_world[:, :3] = (
                np.pad(point_with_intensity[:, :3], ((0, 0), (0, 1)), constant_values=1)
                @ camera_to_world_pose.T
            )[:, :3]
            original_l2w = original_l2ws[idx]
            gt_point_with_intensity_world[:, :3] = (
                np.pad(gt_point_with_intensity[:, :3], ((0, 0), (0, 1)), constant_values=1)
                @ original_l2w.T
            )[:, :3]

            # 若存在跨 block 坐标对齐变换，转到全局坐标系
            if block_coord_transform is not None:
                point_with_intensity_world[:, :3] = (
                    np.pad(
                        point_with_intensity_world[:, :3], ((0, 0), (0, 1)), constant_values=1
                    )
                    @ block_coord_transform.T
                )[:, :3]
                gt_point_with_intensity_world[:, :3] = (
                    np.pad(
                        gt_point_with_intensity_world[:, :3],
                        ((0, 0), (0, 1)),
                        constant_values=1,
                    )
                    @ block_coord_transform.T
                )[:, :3]

        # TXT 写出（--no_txt 时跳过，仅保留 PCD）
        if not skip_txt:
            np.savetxt(
                os.path.join(render_path, "{}.txt".format(render_timestamp)),
                point_with_intensity_world,
            )
            np.savetxt(
                os.path.join(gt_path, "{}.txt".format(render_timestamp)),
                gt_point_with_intensity_world,
            )

        # 计算 ego 在全局坐标系中的位置（与点云坐标系一致，用于 BEV 视频自车定心）
        ego_world = camera_to_world_pose[:3, 3].copy()
        if block_coord_transform is not None:
            ego_hom = np.concatenate([ego_world, [1.0]], axis=0)
            ego_world = (ego_hom @ block_coord_transform.T)[:3]

        render_outputs[str(render_timestamp)] = {
            "rendered": point_with_intensity_world,
            "gt":       gt_point_with_intensity_world,
            "ego_pos":  ego_world,
        }

    total_render_time = time.time() - total_render_start
    fps = total_frames / total_render_time if total_render_time > 0 else 0
    logger.info(f"\n编辑渲染完成:")
    logger.info(f"总渲染帧数: {total_frames}")
    logger.info(f"总渲染时间: {total_render_time:.4f} 秒")
    logger.info(f"平均帧率 (FPS): {fps:.4f}")
    if t_list:
        logger.info(f"单帧平均渲染时间: {sum(t_list) / len(t_list):.4f} 秒")

    return render_outputs


# ---------------------------------------------------------------------------
# render_sets_edit — 加载场景、计算轨迹、解析 edit_config、执行渲染
# ---------------------------------------------------------------------------

def render_sets_edit(
    gt_dynamic_model,
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    edit_obj_info_config,   # 来自 --edit_config 的 RigidNodes 配置
    insert_objs_config,     # 来自 --edit_config 的 Background.add 预加载 PLY
    logger,
    block_coord_transform=None,
):
    """
    加载场景模型，解析 edit_config，执行编辑并渲染。

    Returns:
        render_outputs: 渲染结果字典
    """
    model_id_list = [0]
    if gt_dynamic_model.get_dynamic_obj_id_list() is not None:
        model_id_list.extend(gt_dynamic_model.get_dynamic_obj_id_list())
    model_id_list = list(dict.fromkeys(model_id_list))
    edit_obj_info_config = remap_edit_obj_info_ids(gt_dynamic_model, edit_obj_info_config)

    # ------------------------------------------------------------------
    # iteration 自动检测
    # Scene.__init__ 在 iteration=-1 时会调用 searchForMaxIteration(
    #   {model_path}/point_cloud/)，但 lidargs 实际保存路径是
    #   {model_path}/{model_id}/{block_id}/iteration_*/，两者不同。
    # 在这里预先找到正确路径的最大迭代号，以显式值传入 Scene，绕开该问题。
    # ------------------------------------------------------------------
    block_id = getattr(dataset, "block_id", 0)
    if iteration == -1:
        detected = find_max_iteration_for_block(dataset.model_path, 0, block_id)
        if detected is None:
            raise RuntimeError(
                f"无法自动检测迭代号: 目录不存在或为空。\n"
                f"  请检查: {dataset.model_path}/0/{block_id}/\n"
                f"  或在 render360edit.sh 中手动设置 iteration=<训练迭代数>"
            )
        iteration = detected
        logger.info(f"[iteration 自动检测] block_id={block_id}, 使用 iteration={iteration}")

    with torch.no_grad():
        static_views = None
        model_id_scene_info = {}

        for model_id in model_id_list:
            model_gaussians = GaussianModel(
                dataset.feat_dim,
                dataset.n_offsets,
                dataset.voxel_size,
                dataset.update_depth,
                dataset.update_init_factor,
                dataset.update_hierachy_factor,
                dataset.use_feat_bank,
                dataset.appearance_dim,
                dataset.ratio,
                dataset.add_opacity_dist,
                dataset.add_cov_dist,
                dataset.add_color_dist,
                dataset.color_channel,
            )
            try:
                model_scene = Scene(
                    dataset,
                    model_id,
                    gt_dynamic_model,
                    model_gaussians,
                    load_iteration=iteration,
                    shuffle=False,
                )
            except FileNotFoundError as e:
                if model_id == 0:
                    # 背景模型缺失是致命错误，直接抛出
                    logger.error(f"[Fatal] 背景模型文件不存在: {e}")
                    logger.error(f"  model_path = {dataset.model_path}")
                    logger.error(f"  block_id   = {getattr(dataset, 'block_id', 'N/A')}")
                    logger.error(f"  iteration  = {iteration}")
                    raise
                logger.warning(f"[Skip] model_id {model_id} 缺少模型文件: {e}")
                continue
            except Exception as e:
                if model_id == 0:
                    logger.error(f"[Fatal] 背景模型初始化失败: {e}")
                    raise
                logger.exception(f"[Skip] model_id {model_id} 初始化失败: {e}")
                continue

            if not getattr(model_scene, "init_status", False):
                if model_id == 0:
                    logger.error(
                        f"[Fatal] 背景模型 init_status=False。"
                        f" model_path={dataset.model_path}, block_id={getattr(dataset, 'block_id', 'N/A')}"
                    )
                    raise RuntimeError("背景模型 (model_id=0) init_status=False，无法继续渲染。")
                logger.info(f"model_id {model_id} init_status is False，跳过")
                continue

            time_poses = {}
            total_views = model_scene.getTotalCameras()
            if model_id == 0:
                static_views = total_views
            for view in total_views:
                timestamp = view.image_name
                camera_pose = np.eye(4)
                camera_pose[:3, :3] = np.transpose(view.R)
                camera_pose[:3, 3] = view.T
                time_poses[timestamp] = TimePose(
                    camera_pose=camera_pose,
                    view=view,
                    valid_mask=view.img_mask,
                    gt_mask=view.original_image,
                )
            model_id_scene_info[model_id] = GaussianView(
                gaussians=model_gaussians,
                scene=model_scene,
                time_poses=time_poses,
            )
            model_gaussians.eval()

        if static_views is None:
            raise RuntimeError(
                "背景模型 (model_id=0) 未能加载，static_views 为 None。\n"
                f"  请检查: model_path={dataset.model_path}\n"
                f"           block_id={getattr(dataset, 'block_id', 'N/A')}\n"
                f"           iteration={iteration}\n"
                "  以上目录下应存在 0/<iter>/point_cloud.ply 和对应的 MLP checkpoint。"
            )

        # 如果 edit_config 中也有 RigidNodes 删除，追加处理
        # （注意：此时模型已加载，后删比 delete_specified_objs 效率低，但语义正确）
        if edit_obj_info_config.delete_obj_ids:
            for did in edit_obj_info_config.delete_obj_ids:
                if did in model_id_scene_info:
                    del model_id_scene_info[did]
                    logger.info(f"[edit_config RigidNodes] 已删除 model_id={did}")

        # 如果 edit_config 中有 RigidNodes trajectory 偏移，逐一应用到已加载的模型高斯
        if edit_obj_info_config.obj_id_offset_pairs:
            for model_id, gs_view in model_id_scene_info.items():
                updated = move_specified_objs(
                    model_id, gs_view.gaussians, edit_obj_info_config
                )
                # move_specified_objs in-place modifies _anchor; no need to rebind
            logger.info(
                f"[edit_config RigidNodes] 已应用 trajectory 偏移: "
                f"{edit_obj_info_config.obj_id_offset_pairs}"
            )

        # edit_config 中的 RigidNodes add 操作（非 Background.add）不通过 insert_objs 处理，
        # 因为它需要 ref_id 轨迹而当前尚无支持。
        if edit_obj_info_config.add_id_path_pairs:
            logger.warning(
                "[edit_config RigidNodes.add] 当前不支持通过 edit_config.yaml 添加 RigidNodes 实例。"
            )

        # 合并所有 insert_objs，并按 inject_direct 旗标分流：
        #   render_objs → 走高斯光栅化（原有流程）
        #   inject_objs → 帧后直接注入 PLY 坐标（绕开全景图分辨率瓶颈）
        all_insert_objs = list(insert_objs_config)
        render_objs = [o for o in all_insert_objs if not o.get("inject_direct", False)]
        inject_objs = [o for o in all_insert_objs if o.get("inject_direct", False)]
        if inject_objs:
            logger.info(
                f"[inject_direct] 共 {len(inject_objs)} 个注入实例（inject_direct_points=true）"
            )

        # ------------------------------------------------------------------
        # 构建测试视角和时间戳映射
        # ------------------------------------------------------------------
        test_views = []
        test_timestamp = []
        for idx, scene_view in enumerate(static_views):
            render_timestamp = scene_view.image_name
            test_timestamp.append(int(render_timestamp))
            test_views.append(scene_view)

        valid_timestamp_model = {}
        for view in static_views:
            timestamp = view.image_name
            valid_timestamp_model[timestamp] = []
            for model_id in model_id_scene_info.keys():
                if timestamp in model_id_scene_info[model_id].time_poses:
                    valid_timestamp_model[timestamp].append(model_id)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        render_outputs = render_set_edit(
            gt_dynamic_model,
            dataset,
            "test",
            iteration,
            valid_timestamp_model,
            model_id_scene_info,
            test_views,
            pipeline,
            background,
            render_objs,
            inject_objs=inject_objs,
            block_coord_transform=block_coord_transform,
        )

    return render_outputs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def find_max_iteration_for_block(model_path: str, model_id: int, block_id: int) -> Optional[int]:
    """
    在 {model_path}/{model_id}/{block_id}/iteration_*/ 中寻找最大迭代号。

    这是 lidargs 实际的模型保存路径，与 scene/__init__.py 中 searchForMaxIteration
    使用的 {model_path}/point_cloud/ 不同（后者仅适用于 vanilla 3DGS）。
    """
    search_dir = os.path.join(model_path, str(model_id), str(block_id))
    if not os.path.isdir(search_dir):
        return None
    iters = []
    for name in os.listdir(search_dir):
        if name.startswith("iteration_"):
            try:
                iters.append(int(name.split("_")[-1]))
            except ValueError:
                pass
    return max(iters) if iters else None


def get_trained_frames_from_block_info(model_path: str) -> Optional[list]:
    """
    读取 block_info.json，返回实际训练过的帧列表（int）。

    兼容两种格式：
    1. 简化格式：{"0": [36,37,...,49], "1": [50,51,...,56]}
    2. 训练脚本实际输出格式：
       {
         "block_time_with_extend": {"0": [...], "1": [...]},
         "block_time_without_extend": {"0": [...], "1": [...]},
         ...
       }

    返回值：所有 block 的帧合并并排序后的列表，如 [36,37,...,56]
    若文件不存在则返回 None（调用方回退到 bin 文件枚举）。
    """
    block_info_path = os.path.join(model_path, "block_info.json")
    if not os.path.exists(block_info_path):
        return None
    with open(block_info_path) as f:
        data = json.load(f)

    if isinstance(data.get("block_time_without_extend"), dict):
        frame_groups = data["block_time_without_extend"].values()
    elif isinstance(data.get("block_time_with_extend"), dict):
        frame_groups = data["block_time_with_extend"].values()
    else:
        frame_groups = data.values()

    frames = []
    for frame_list in frame_groups:
        if isinstance(frame_list, list):
            frames.extend(frame_list)
        else:
            frames.append(frame_list)

    normalized_frames = []
    for frame_id in frames:
        try:
            normalized_frames.append(int(frame_id))
        except (TypeError, ValueError):
            logger.warning(f"[block_info.json] 跳过无法解析的 frame_id: {frame_id}")

    normalized_frames = sorted(set(normalized_frames))
    return normalized_frames if normalized_frames else None


if __name__ == "__main__":
    parser = ArgumentParser(description="LiDAR 360 render with scene editing")
    model    = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration",   default=-1, type=int)
    parser.add_argument("--quiet",       action="store_true")
    parser.add_argument("--test_frames", nargs="+", type=int, default=[])
    parser.add_argument(
        "--edit_config",
        type=str,
        default=None,
        help="新格式编辑配置文件，支持 Background.add + time_window (edit_config.yaml)",
    )
    parser.add_argument("--dataset",    type=str, default="chery")
    parser.add_argument("--block_size", type=int, default=50)
    parser.add_argument("--video_fps",      type=int,   default=10,   help="可视化视频帧率")
    parser.add_argument("--bev_width",      type=int,   default=1200, help="BEV 视频宽度（像素，单侧）")
    parser.add_argument("--bev_height",     type=int,   default=900,  help="BEV 视频高度（像素）")
    parser.add_argument("--bev_xmin",       type=float, default=None, help="BEV x 方向相对自车的偏移下界（米），默认 -60")
    parser.add_argument("--bev_xmax",       type=float, default=None, help="BEV x 方向相对自车的偏移上界（米），默认  60")
    parser.add_argument("--bev_ymin",       type=float, default=-50.0, help="BEV y 轴最小值（米）")
    parser.add_argument("--bev_ymax",       type=float, default=50.0,  help="BEV y 轴最大值（米）")
    parser.add_argument("--bev_point_size", type=float, default=0.1,   help="BEV 散点大小（matplotlib s 参数）")
    parser.add_argument("--no_video",       action="store_true",        help="跳过 BEV 视频生成，只保存点云 TXT/PCD")
    parser.add_argument("--no_txt",         action="store_true",        help="跳过 TXT 点云写入（np.savetxt），只保留 PCD，大幅加速")
    parser.add_argument("--fast_render",    action="store_true",        help="快速渲染：GPU pano2lidar + 帧变换预缓存 + CUDA warmup（需配合 fast_rasterizer_path 扩展）")
    parser.add_argument("--enable_raydrop_postfilter", action="store_true")
    parser.add_argument("--enable_depth_postfilter", action="store_true", default=True)
    parser.add_argument("--raydrop_postfilter_threshold", type=float, default=0.5)
    parser.add_argument(
        "--depth_postfilter_ratio_threshold", type=float, default=0.10
    )
    parser.add_argument("--near_preserve_depth", type=float, default=0.0)
    args = parser.parse_args(sys.argv[1:])

    out_dir = getattr(args, "output_path", None) or "."
    os.makedirs(out_dir, exist_ok=True)
    logger = get_logger(out_dir, args.model_path)

    align_args_with_training_output(args)
    if not getattr(args, "source_path", None):
        raise ValueError(
            "source_path 为空，且未能从 model_path/cfg_args 自动回填。"
            " 请在 render360edit.sh 中显式设置 source_path，"
            "或确保训练输出目录下存在 cfg_args。"
        )

    args.novel_poses = []

    # 自动检测帧范围：test_frames 为空，且 novel_poses 也为空时自动推断。
    # 枚举数据集全部 bin 文件，过滤掉没有训练好模型的 block 对应帧。
    # 判断依据是 block 目录是否存在（model_path/0/<block_id>/），
    # 而非帧是否出现在训练集中——test frame 同样可以被有效渲染。
    if len(args.test_frames) == 0 and not args.novel_poses:
        bin_files = sorted(
            [f for f in os.listdir(os.path.join(args.source_path, "lidar", "bin"))
             if f.endswith(".bin")],
            key=lambda x: int(x.split(".")[0]),
        )
        all_frames = [int(f.split(".")[0]) for f in bin_files]
        filtered = [
            fid for fid in all_frames
            if os.path.isdir(os.path.join(args.model_path, "0", str(fid // args.block_size)))
        ]
        args.test_frames = filtered if filtered else all_frames
        logger.info(
            f"[bin文件枚举] 检测到可渲染帧 {len(args.test_frames)} 个: "
            f"{args.test_frames[0]} ~ {args.test_frames[-1]}"
        )
    args.test_frames = sorted(set(int(frame_id) for frame_id in args.test_frames))
    logger.info("Rendering (edit) " + args.model_path)

    safe_state(args.quiet)
    model_args = model.extract(args)
    model_args = attach_render_postprocess_args(model_args, args)

    if args.dataset == "chery":
        from scene.chery_dataloader import Chery_Dataloader as GT_Dataloader
    elif args.dataset == "zdrive":
        from scene.zdrive_dataloader import ZDrive_Dataloader as GT_Dataloader
    elif args.dataset == "chery_lidar_360":
        from scene.adaptive_dataloader import AdaptiveDataloader as GT_Dataloader
    else:
        logger.info(f"不支持的数据集格式: {args.dataset}")
        sys.exit(1)

    train_frame_times = []
    if args.test_frames:
        train_frame_times = [int(frame_id) for frame_id in args.test_frames]
    else:
        for item in args.novel_poses:
            train_frame_times.append(int(item["frame_id"]))
    train_frame_times = sorted(set(train_frame_times))

    block_info = {}
    for frame_id in train_frame_times:
        block_id = frame_id // args.block_size
        if block_id not in block_info:
            block_info[block_id] = []
        block_info[block_id].append(frame_id)

    insert_objs_config, edit_obj_info_config, all_barrier_placements = (
        parse_and_apply_edit_yaml(
            args,
            edit_config_path=getattr(args, "edit_config", None),
            train_frame_times=train_frame_times,
        )
    )

    output_paths = []
    lidar_id = 0
    pose_first_frame = None  # 全局第0帧的绝对 pose，所有 block 共用同一坐标系原点
    all_render_outputs: Dict[str, dict] = {}   # 跨 block 累积，供 BEV 使用

    for block_id, block_frame_times in block_info.items():
        model_args.block_id = block_id
        gt_dynamic_model = GT_Dataloader(
            model_args, train=False, train_frame_times=block_frame_times
        )
        if hasattr(args, "novel_poses"):
            gt_dynamic_model.set_novel_poses_setting(args.novel_poses)

        # 计算跨 block 坐标对齐变换：inv(pose_first_frame) @ pose_current_block
        pose_current_block = gt_dynamic_model.get_lidar_to_world_start()
        if pose_first_frame is None:
            pose_first_frame = pose_current_block
        block_coord_transform = np.linalg.inv(pose_first_frame) @ pose_current_block

        # 将 insert_objs 的 sim_pose 从全局坐标系转换到当前 block-local 坐标系。
        # insert_objs_config 里的 sim_pose 是基于全局坐标（relative to frame 0）计算的，
        # 但 adaptive_dataloader 每个 block 的渲染坐标系以该 block 第一帧为原点。
        # 变换：sim_pose_local = inv(block_coord_transform) @ sim_pose_global
        # block 0 时 block_coord_transform = I，无影响。
        to_block_local = np.linalg.inv(block_coord_transform)  # inv(inv(pose_first_frame)@P_N) = inv(P_N)@pose_first_frame

        insert_objs_config_block = []
        for obj in insert_objs_config:
            obj_copy = dict(obj)
            if "sim_pose" in obj_copy and obj_copy["sim_pose"] is not None:
                sp = obj_copy["sim_pose"]
                if isinstance(sp, torch.Tensor):
                    sp_np = sp.cpu().numpy().reshape(4, 4)
                    sp_np = to_block_local.astype(np.float32) @ sp_np
                    obj_copy["sim_pose"] = torch.tensor(sp_np, dtype=torch.float32, device="cuda")
                else:
                    obj_copy["sim_pose"] = (to_block_local.astype(np.float32) @ np.array(sp).reshape(4, 4))
            insert_objs_config_block.append(obj_copy)

        render_outputs = render_sets_edit(
            gt_dynamic_model,
            model_args,
            args.iteration,
            pipeline.extract(args),
            edit_obj_info_config,
            insert_objs_config_block,
            logger,
            block_coord_transform=block_coord_transform,
        )
        all_render_outputs.update(render_outputs)   # 跨 block 累积

        # 保存编辑后点云 PCD
        edit_pcd_dir = os.path.join(out_dir, "renders_edit")
        os.makedirs(edit_pcd_dir, exist_ok=True)
        for frame_key, pcd in render_outputs.items():
            logger.info(
                f"[Lidar Edit Renderer] 保存编辑后点云，frame={frame_key}"
            )
            render_pcd = pcd["rendered"]
            render_output_filename = os.path.join(
                edit_pcd_dir, f"ldr{lidar_id}_{frame_key}.pcd"
            )
            save_pcd_binary(render_output_filename, render_pcd)
            output_paths.append(render_output_filename)

    # 保存 barrier_instance_bboxes.json
    if all_barrier_placements:
        bbox_json_path = os.path.join(out_dir, "renders_edit", "barrier_instance_bboxes.json")
        with open(bbox_json_path, "w") as f:
            json.dump(all_barrier_placements, f, indent=2)
        logger.info(
            f"[Lidar Edit Renderer] barrier_instance_bboxes.json 已保存至 {bbox_json_path}"
        )

    logger.info(f"所有编辑后点云已保存至 {os.path.join(out_dir, 'renders_edit')}")

    # 生成可视化视频（GT vs Rendered 并排 BEV）
    if getattr(args, "no_video", False):
        logger.info("跳过 BEV 视频生成（--no_video）")
    else:
        if args.bev_xmin is not None and args.bev_xmax is not None:
            ego_xlim = (args.bev_xmin, args.bev_xmax)
        elif args.bev_xmin is not None or args.bev_xmax is not None:
            raise ValueError("--bev_xmin 和 --bev_xmax 需要同时设置")
        else:
            ego_xlim = (-30.0, 30.0)  # 默认：自车前后各 30 m
        render_video_dir = os.path.join(out_dir, "rendervideo")
        os.makedirs(render_video_dir, exist_ok=True)
        _make_bev_video(
            all_render_outputs,
            out_path=os.path.join(render_video_dir, "lidar_edit_bev.mp4"),
            bev_xlim=ego_xlim,
            bev_ylim=(getattr(args, "bev_ymin", -50.0), getattr(args, "bev_ymax", 50.0)),
            panel_w=getattr(args, "bev_width", 1200),
            panel_h=getattr(args, "bev_height", 900),
            fps=getattr(args, "video_fps", 10),
            pt_size=getattr(args, "bev_point_size", 2.0),
        )


    sys.exit(0)
