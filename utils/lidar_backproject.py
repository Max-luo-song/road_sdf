#!/usr/bin/env python3
# =============================================================================
# lidar_backproject.py — LiDAR 点云反投影到相机图像
#
# 适用场景：
#   render360edit（AdaptiveDataloader 训练）输出的世界坐标系点云 → 投影到相机图像
#
# 坐标系完整链路（与 AdaptiveDataloader / render360edit 一致）：
#
#   [AdaptiveDataloader 中]
#     lidar_to_world_start  ← lidar_pose/{start_frame:06d}.txt  （绝对世界 pose，block 起始帧）
#     l2w[t]                = inv(lidar_to_world_start) @ L_t    （block-local 归一化 pose）
#     extrinsic             = I                                   （单雷达，无偏移）
#
#   [render360edit 中，跨 block 全局对齐后]
#     P_0                   = 第一个 block 的 lidar_to_world_start（绝对世界 pose）
#     block_coord_transform = inv(P_0) @ P_N
#     最终世界坐标           = block_coord_transform @ l2w[t] @ pts_sensor
#                           = inv(P_0) @ L_t @ pts_sensor          ← TXT 文件中的坐标
#
#   [反投影时]
#     L0                    = P_0 = lidar_pose/{first_block0_frame:06d}.txt
#     lidar_to_world        = inv(L0) @ L_t
#     cam_to_world[t]       = lidar_to_world @ cam_to_lidar
#                           = inv(P_0) @ L_t @ cam_to_lidar
#     world_to_cam[t]       = inv(cam_to_world[t])
#     投影：pts_cam = world_to_cam[:3,:3] @ pts_world + world_to_cam[:3,3]
#           pts_2d  = K @ pts_cam / pts_cam.z  （+ 畸变校正）
#
# 文件格式（AdaptiveDataloader 约定）：
#   renders_edit/{frame_id}.txt  → N×4  (x, y, z, intensity)  无零填充文件名
#   lidar_pose/{t:06d}.txt       → 4×4  绝对 LiDAR-to-world  六位零填充
#   extrinsics/{cam_id}.txt      → 4×4  cam_to_lidar
#   intrinsics/{cam_id}.txt      → [fx, fy, cx, cy, k1, k2, p1, p2, ...]
#   images/{t:03d}_{cam_id}.jpg  → 原始相机图像（Chery/ZDrive 格式）
#
# 输出目录结构：
#   <output_dir>/cam00/000005.jpg        — 帧图像（每相机一个子目录）
#   <output_dir>/cam00_backproject.mp4   — 对应相机的视频
# =============================================================================
# python3 sim_render/lidar/lidar_backproject.py \
#     --source_path "/nas/users/liuzy/data/scene_reconstruction_data/100clips/processed/training/20251203_132047_Q3719-56_76" \
#     --render_dir  "/nas/oldbak/yx/M5finalfix/cheryoutput/3M360/20251203_132047_Q3719-56_76/2026-410-clips/render_edit/renders_edit/" \
#     --img_dir     "/nas_thoru/users/liuzy/data/scene_reconstruction_data/100clips/output/qcraft_20251203_132047_Q3719-56_76/20260411_lidar+cam0_1_2_5_6_7_9_10_11_12/videos_edit/full_set_40000_rgbs/" \
#     --output_dir  "/nas/oldbak/yx/M5finalfix/cheryoutput/3M360/20251203_132047_Q3719-56_76/render_edit/backproject/" \
#     --cam_ids 0 1 2 5 6 7 9 10 11 12 \
#     --point_radius 2 \
#     --depth_min 2.0 \
#     --depth_max 60.0 \
#     --video_fps 10

import os
import sys
import argparse
import glob
import logging

import numpy as np
import cv2
import imageio

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数：文件加载
# ──────────────────────────────────────────────────────────────────────────────

def load_lidar_pose(pose_dir: str, frame_id: int) -> np.ndarray:
    """加载单帧 LiDAR pose（4×4），自动尝试 6 位和 3 位文件名。"""
    for fmt in (f"{frame_id:06d}.txt", f"{frame_id:03d}.txt"):
        p = os.path.join(pose_dir, fmt)
        if os.path.exists(p):
            return np.loadtxt(p)
    raise FileNotFoundError(
        f"LiDAR pose not found for frame {frame_id} in {pose_dir}"
    )


def load_intrinsics(path: str):
    """
    加载内参文件：[fx, fy, cx, cy, k1, k2, p1, p2, k3, ...]

    返回:
        K     : (3, 3) float64 内参矩阵
        dist  : (n,)   float64 畸变系数（k1,k2,p1,p2[,k3,...]）
    """
    raw = np.loadtxt(path)
    fx, fy, cx, cy = float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])
    dist = raw[4:].astype(np.float64) if len(raw) > 4 else np.zeros(5, np.float64)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return K, dist


def load_extrinsics(path: str) -> np.ndarray:
    """加载外参文件（4×4 cam_to_lidar 矩阵）。"""
    return np.loadtxt(path).astype(np.float64)


def find_image(img_dir: str, frame_id: int, cam_id: int):
    """在 images/ 目录中查找指定帧和相机的原始图像。"""
    patterns = [
        f"{frame_id:03d}_{cam_id}.jpg",       # Chery/zdrive 标准格式
        f"{frame_id:06d}_{cam_id}.jpg",       # qcraft 1位cam格式
        f"{frame_id:03d}_{cam_id}.png",
        f"{frame_id:06d}_{cam_id}.png",
        f"{frame_id:06d}_{cam_id:03d}.png",   # qcraft 相机编辑输出格式（3位cam，如 000056_000.png）
        f"{frame_id:06d}_{cam_id:03d}.jpg",
    ]
    for pat in patterns:
        path = os.path.join(img_dir, pat)
        if os.path.exists(path):
            return path
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 投影与可视化
# ──────────────────────────────────────────────────────────────────────────────

def project_points(
    world_xyz: np.ndarray,        # (N, 3) 世界坐标
    K: np.ndarray,                # (3, 3) 内参
    dist: np.ndarray,             # (n,)   畸变系数
    world_to_cam: np.ndarray,     # (4, 4) world → camera 变换
    intensities: np.ndarray = None,  # (N,) 强度值，可选
) -> tuple:
    """
    将世界坐标点投影到图像平面（含畸变）。

    返回:
        pts_2d     : (M, 2) 浮点像素坐标
        depths     : (M,)   正深度值（用于遮挡排序）
        valid_idx  : (M,)   有效点在原始 world_xyz 中的索引
        intens_out : (M,)   对应强度值（若未传入则全为 0.5）
    """
    R = world_to_cam[:3, :3]
    t = world_to_cam[:3, 3]
    pts_cam = (R @ world_xyz.T).T + t        # (N, 3)

    # 只保留正深度（z > 0.1m）
    valid = pts_cam[:, 2] > 0.1
    valid_idx = np.where(valid)[0]
    pts_cam_v = pts_cam[valid]               # (M, 3)

    if len(pts_cam_v) == 0:
        return np.zeros((0, 2)), np.zeros(0), valid_idx, np.zeros(0)

    # cv2.projectPoints：rvec=0, tvec=0 表示点已在相机坐标系中，只需 K+distortion
    rvec = np.zeros(3, dtype=np.float64)
    tvec = np.zeros(3, dtype=np.float64)
    pts_2d, _ = cv2.projectPoints(
        pts_cam_v[:, np.newaxis, :].astype(np.float64),
        rvec, tvec, K, dist,
    )
    pts_2d = pts_2d.reshape(-1, 2)

    intens_out = (
        intensities[valid_idx].astype(np.float64)
        if intensities is not None
        else np.full(len(valid_idx), 0.5)
    )
    return pts_2d, pts_cam_v[:, 2], valid_idx, intens_out


def intensity_colormap(intensities: np.ndarray) -> np.ndarray:
    """强度 [0,1] → jet 伪彩色 BGR（与 BEV 一致：蓝→绿→黄→红），输出 (N, 3) uint8。"""
    val = (np.clip(intensities, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.applyColorMap(val.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)


def overlay_points(
    img: np.ndarray,
    pts_2d: np.ndarray,        # (M, 2)
    depths: np.ndarray,        # (M,)  用于遮挡排序（近覆盖远）
    intensities: np.ndarray,   # (M,)  用于着色，与 BEV jet 对齐
    radius: int,
) -> np.ndarray:
    """
    在图像上叠加强度着色点云，近处点覆盖远处点。
    着色方案：jet colormap，低强度=蓝，高强度=红（与 BEV 一致）。
    """
    H, W = img.shape[:2]
    out = img.copy()

    in_frame = (
        (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < W) &
        (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < H)
    )
    if not np.any(in_frame):
        return out

    pts   = pts_2d[in_frame]
    dep   = depths[in_frame]
    intens = intensities[in_frame]
    cols  = intensity_colormap(intens)   # (M, 3) BGR

    # 从远到近排序，近处覆盖远处
    order = np.argsort(-dep)
    px = np.clip(np.round(pts[order, 0]).astype(np.int32), 0, W - 1)
    py = np.clip(np.round(pts[order, 1]).astype(np.int32), 0, H - 1)
    col = cols[order]

    if radius <= 1:
        out[py, px] = col
    else:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    ny = np.clip(py + dy, 0, H - 1)
                    nx = np.clip(px + dx, 0, W - 1)
                    out[ny, nx] = col

    return out


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    """在左上角写文字标签。"""
    out = img.copy()
    cv2.putText(out, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="将 render360edit 点云反投影到相机图像并生成视频"
    )
    parser.add_argument(
        "--source_path", "-s", required=True,
        help="原始数据集路径（含 intrinsics/ extrinsics/ lidar_pose/ images/）",
    )
    parser.add_argument(
        "--render_dir", required=True,
        help="render360edit 输出目录（含 {frame}.txt），"
             "例如 <model_path>/render_edit/renders_edit/",
    )
    parser.add_argument(
        "--output_dir", "-o", required=True,
        help="反投影结果保存目录",
    )
    parser.add_argument(
        "--cam_ids", nargs="+", type=int, default=None,
        help="要处理的相机 ID 列表（默认自动从 intrinsics/ 检测）",
    )
    parser.add_argument(
        "--frame_ids", nargs="+", type=int, default=None,
        help="要处理的帧 ID 列表（默认自动从 render_dir/*.txt 检测）",
    )
    # ── 可视化参数 ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--point_radius", type=int, default=2,
        help="点圆半径（像素），默认 2；增大可提高可见性",
    )
    parser.add_argument(
        "--depth_min", type=float, default=2.0,
        help="深度着色最小值（米），对应 TURBO 色谱冷色端",
    )
    parser.add_argument(
        "--depth_max", type=float, default=80.0,
        help="深度着色最大值（米），对应 TURBO 色谱暖色端",
    )
    parser.add_argument(
        "--video_fps", type=int, default=10,
        help="输出视频帧率",
    )
    parser.add_argument(
        "--no_video", action="store_true",
        help="只保存单帧图像，不生成视频",
    )
    parser.add_argument(
        "--use_gt", action="store_true",
        help="使用 render_dir/gt/{frame}.txt（GT 点云）而非渲染点云",
    )
    # ── 坐标系参数 ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--pose_dir", default=None,
        help="LiDAR pose 文件目录（默认 source_path/lidar_pose/）",
    )
    parser.add_argument(
        "--img_dir", default=None,
        help="原始图像目录（默认 source_path/images/）",
    )
    parser.add_argument(
        "--no_distortion", action="store_true",
        help="投影时忽略畸变系数（鱼眼相机不适用此脚本，普通相机建议保留畸变）",
    )
    parser.add_argument(
        "--l0_frame_id", type=int, default=None,
        help="世界坐标系原点帧号（默认 = frame_ids 的第一帧）。"
             "必须与 render360edit 训练时 block 0 的 start_timestep 一致，"
             "通常为数据集的第一帧（frame 0），无需手动指定。",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 目录路径 ─────────────────────────────────────────────────────────────
    intr_dir  = os.path.join(args.source_path, "intrinsics")
    extr_dir  = os.path.join(args.source_path, "extrinsics")
    pose_dir  = args.pose_dir or os.path.join(args.source_path, "lidar_pose")
    img_dir   = args.img_dir  or os.path.join(args.source_path, "images")

    # 若 --use_gt，从 gt/ 子目录读取
    if args.use_gt:
        pcd_dir = os.path.join(args.render_dir, "gt")
        logger.info("模式: GT 点云")
    else:
        pcd_dir = args.render_dir
        logger.info("模式: 渲染点云")

    # ── 检测相机 ID ──────────────────────────────────────────────────────────
    if args.cam_ids is None:
        cam_txts = sorted(glob.glob(os.path.join(intr_dir, "*.txt")))
        args.cam_ids = []
        for p in cam_txts:
            stem = os.path.splitext(os.path.basename(p))[0]
            try:
                args.cam_ids.append(int(stem))
            except ValueError:
                pass
    if not args.cam_ids:
        logger.error(f"未在 {intr_dir} 找到内参文件，请检查 --source_path")
        sys.exit(1)
    logger.info(f"相机列表: {args.cam_ids}")

    # ── 检测帧 ID ────────────────────────────────────────────────────────────
    if args.frame_ids is None:
        pcd_txts = sorted(glob.glob(os.path.join(pcd_dir, "*.txt")))
        args.frame_ids = []
        for p in pcd_txts:
            stem = os.path.splitext(os.path.basename(p))[0]
            try:
                args.frame_ids.append(int(stem))
            except ValueError:
                logger.debug(f"跳过非帧文件: {p}")
        args.frame_ids.sort()
    if not args.frame_ids:
        logger.error(f"未在 {pcd_dir} 找到点云 TXT 文件")
        sys.exit(1)
    logger.info(
        f"共 {len(args.frame_ids)} 帧，范围 {args.frame_ids[0]}~{args.frame_ids[-1]}"
    )

    # ── 加载 L0（世界坐标系原点 pose）───────────────────────────────────────
    # AdaptiveDataloader 中：
    #   lidar_to_world_start = lidar_pose/{train_frame_times[0]:06d}.txt  （block 起始帧）
    #   l2w[t]               = inv(lidar_to_world_start) @ L_t
    #
    # render360edit 中：
    #   P_0                  = lidar_to_world_start of block 0            （全局原点）
    #   block_coord_transform = inv(P_0) @ P_N
    #   TXT 文件坐标           = inv(P_0) @ L_t @ pts_sensor
    #
    # → L0 必须 = P_0 = lidar_pose/{block0_start_frame:06d}.txt
    #
    # 默认取 frame_ids 最小帧（= block 0 第一帧），与 render360edit 保持一致。
    # 若起始帧不是 0，用 --l0_frame_id 显式指定。
    l0_frame_id = args.l0_frame_id if args.l0_frame_id is not None else min(args.frame_ids)
    L0     = load_lidar_pose(pose_dir, l0_frame_id)
    L0_inv = np.linalg.inv(L0)
    logger.info(f"L0 (全局原点) = 帧 {l0_frame_id}  ← lidar_pose/{l0_frame_id:06d}.txt")

    # ── 加载各相机标定 ───────────────────────────────────────────────────────
    cameras = {}  # cam_id → {"K", "dist", "cam_to_lidar"}
    for cam_id in args.cam_ids:
        intr_path = os.path.join(intr_dir, f"{cam_id}.txt")
        extr_path = os.path.join(extr_dir, f"{cam_id}.txt")

        if not os.path.exists(intr_path):
            logger.warning(f"cam {cam_id}: 内参文件不存在，跳过 → {intr_path}")
            continue
        if not os.path.exists(extr_path):
            logger.warning(f"cam {cam_id}: 外参文件不存在，跳过 → {extr_path}")
            continue

        K, dist = load_intrinsics(intr_path)
        if args.no_distortion:
            dist = np.zeros_like(dist)

        cam_to_lidar = load_extrinsics(extr_path)
        cameras[cam_id] = {"K": K, "dist": dist, "cam_to_lidar": cam_to_lidar}

        logger.info(
            f"cam {cam_id}: fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
            f"cx={K[0,2]:.1f} cy={K[1,2]:.1f} | "
            f"dist={dist[:4]}"
        )

    if not cameras:
        logger.error("没有找到有效相机标定，退出")
        sys.exit(1)

    # 为每个相机创建输出目录
    for cam_id in cameras:
        os.makedirs(os.path.join(args.output_dir, f"cam{cam_id:02d}"), exist_ok=True)

    # 收集视频帧
    video_frames = {cam_id: [] for cam_id in cameras}

    # ── 逐帧处理 ─────────────────────────────────────────────────────────────
    processed = 0
    for frame_id in args.frame_ids:
        pcd_path = os.path.join(pcd_dir, f"{frame_id}.txt")
        if not os.path.exists(pcd_path):
            logger.warning(f"帧 {frame_id}: 点云文件不存在，跳过 → {pcd_path}")
            continue

        # 加载世界坐标点云 (N, 4): x y z intensity
        pts_raw = np.loadtxt(pcd_path)
        if pts_raw.ndim == 1:
            pts_raw = pts_raw[np.newaxis, :]
        if len(pts_raw) == 0:
            logger.warning(f"帧 {frame_id}: 空点云，跳过")
            continue
        world_xyz   = pts_raw[:, :3].astype(np.float64)
        world_intens = pts_raw[:, 3].astype(np.float64) if pts_raw.shape[1] >= 4 else None

        # 加载当前帧 LiDAR pose，计算 lidar_to_world（相对 L0 归一化）
        try:
            L_t = load_lidar_pose(pose_dir, frame_id)
        except FileNotFoundError as e:
            logger.warning(str(e))
            continue
        lidar_to_world = L0_inv @ L_t   # inv(L0) @ L_t

        for cam_id, cam in cameras.items():
            # 坐标链（与 AdaptiveDataloader / chery_sourceloader 完全对齐）：
            #   cam_to_world  = (inv(P_0) @ L_t) @ cam_to_lidar
            #   world_to_cam  = inv(cam_to_world)
            #   pts_cam       = world_to_cam[:3,:3] @ pts_world + world_to_cam[:3,3]
            cam_to_world = lidar_to_world @ cam["cam_to_lidar"]
            world_to_cam = np.linalg.inv(cam_to_world)

            # 投影
            pts_2d, depths, valid_idx, intens = project_points(
                world_xyz, cam["K"], cam["dist"], world_to_cam, world_intens
            )
            if len(pts_2d) == 0:
                continue

            # 读取原始图像
            img_path = find_image(img_dir, frame_id, cam_id)
            if img_path is not None:
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    logger.warning(f"  cam {cam_id}: 无法读取图像 {img_path}")
                    img_bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
            else:
                # 找不到原图时使用黑色画布（720p）
                img_bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
                if frame_id == args.frame_ids[0]:
                    logger.warning(
                        f"cam {cam_id}: 在 {img_dir} 中未找到帧 {frame_id} 图像，"
                        "将在黑色背景上绘制点云"
                    )

            # 叠加点云（强度着色，与 BEV jet colormap 一致）
            out_bgr = overlay_points(
                img_bgr, pts_2d, depths, intens,
                radius=args.point_radius,
            )
            # 添加帧号标签
            out_bgr = add_label(out_bgr, f"frame {frame_id:04d}  cam {cam_id}")

            # 保存图像
            out_path = os.path.join(args.output_dir, f"cam{cam_id:02d}", f"{frame_id:06d}.jpg")
            cv2.imwrite(out_path, out_bgr)

            # 收集视频帧（BGR → RGB）
            if not args.no_video:
                out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
                video_frames[cam_id].append(out_rgb)

        processed += 1
        if processed % 20 == 0 or processed == len(args.frame_ids):
            logger.info(f"  已处理 {processed}/{len(args.frame_ids)} 帧")

    # ── 写入视频 ──────────────────────────────────────────────────────────────
    if not args.no_video:
        for cam_id, frames in video_frames.items():
            if not frames:
                continue
            vpath = os.path.join(args.output_dir, f"cam{cam_id:02d}_backproject.mp4")
            logger.info(f"写入视频 cam{cam_id:02d}: {vpath}  ({len(frames)} 帧)")
            with imageio.get_writer(vpath, fps=args.video_fps, macro_block_size=1) as writer:
                for f in frames:
                    writer.append_data(f)
            logger.info(f"视频已保存: {vpath}")

    logger.info(f"\n反投影完成！共处理 {processed} 帧，结果在: {args.output_dir}")


if __name__ == "__main__":
    main()