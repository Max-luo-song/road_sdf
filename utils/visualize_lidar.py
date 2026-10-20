import os
import shutil
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import open3d as o3d

# 视频参数
fps = 5
width, height = 1200, 900
point_size = 0.1
use_intensity = True
# =========================================


def get_sorted_timestamps(folder):
    """按时间戳排序所有 .txt 文件"""
    files = [f for f in os.listdir(folder) if f.endswith(".txt")]
    timestamps = []
    for f in files:
        try:
            ts = int(f.split(".")[0])  # e.g., "100_render_points.txt" -> 100
            timestamps.append(ts)
        except:
            continue
    return sorted(set(timestamps))


def create_top_view_video(data_dir, video_name, fixed_ylim):
    """生成俯视视频，使用 CloudCompare 风格（Z 高度着色）"""
    timestamps = get_sorted_timestamps(data_dir)
    if not timestamps:
        print(f"No valid files in {data_dir}, skip video.")
        return

    video_path = os.path.join(video_dir, video_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    z_min, z_max = -1.0, 10.0  # 默认范围（可根据场景调整）
    for ts in timestamps:
        file_path = os.path.join(data_dir, f"{ts}.txt")

        if not file_path:
            continue

        try:
            data = np.loadtxt(file_path, skiprows=1)
            if data.size == 0:
                print(f"Skip frame {ts}: empty")
                continue
            if data.ndim == 1:
                data = data.reshape(1, -1)
            xyz = data[:, :3]
            intensity = data[:, 3]
        except Exception as e:
            print(f"Skip frame {ts}: {e}")
            continue

        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        # === 每帧动态计算 xlim（少留白）===
        if len(x) > 0:
            x_min, x_max = x.min(), x.max()
            margin_x = 2.0  # 左右各留 2 米
            xlim = (x_min - margin_x, x_max + margin_x)
        else:
            xlim = (-10, 10)

        plt.figure(figsize=(width / 100, height / 100), dpi=100)
        plt.xlim(xlim)
        plt.ylim(fixed_ylim)
        plt.axis("off")

        # === 用 Z 坐标着色 ===
        # colors = np.clip((z - z_min) / (z_max - z_min + 1e-6), 0, 1)
        colors = np.clip(intensity, 0, 1)
        plt.scatter(x, y, c=colors, s=point_size, cmap="jet", vmin=0, vmax=1)

        plt.tight_layout(pad=0)
        plt.savefig("temp_frame.png", bbox_inches="tight", pad_inches=0)
        plt.close()

        frame = cv2.imread("temp_frame.png")
        if frame is not None:
            frame = cv2.resize(frame, (width, height))
            video_writer.write(frame)

    video_writer.release()
    if os.path.exists("temp_frame.png"):
        os.remove("temp_frame.png")


def create_gt_render_view_video(render_dir, gt_dir, video_name, fixed_ylim):
    """生成gt和render的top view视频，放在一起对比"""
    timestamps = get_sorted_timestamps(render_dir)
    if not timestamps:
        print(f"No valid files in {render_dir}, skip video.")
        return

    video_path = os.path.join(video_dir, video_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width * 2, height))

    z_min, z_max = -1.0, 10.0  # 默认范围（可根据场景调整）
    for ts in timestamps:

        render_file_path = os.path.join(render_dir, f"{ts}.txt")
        gt_file_path = os.path.join(gt_dir, f"{ts}.txt")

        if not render_file_path or not gt_file_path:
            continue

        try:
            render_data = np.loadtxt(render_file_path, skiprows=1)
            gt_data = np.loadtxt(gt_file_path, skiprows=1)
            if render_data.size == 0 or gt_data.size == 0:
                print(f"Skip frame {ts}: empty")
                continue
            if render_data.ndim == 1:
                render_data = render_data.reshape(1, -1)
            if gt_data.ndim == 1:
                gt_data = gt_data.reshape(1, -1)
            render_xyz = render_data[:, :3]
            render_intensity = render_data[:, 3]
            gt_xyz = gt_data[:, :3]
            gt_intensity = gt_data[:, 3]
        except Exception as e:
            print(f"Skip frame {ts}: {e}")
            continue

        # Create side-by-side plots for render and gt
        fig, axs = plt.subplots(1, 2, figsize=(width * 2 / 100, height / 100), dpi=100)

        for ax, xyz, intensity, title in zip(
            axs,
            [gt_xyz, render_xyz],
            [gt_intensity, render_intensity],
            ["before move", "after move(shift right 3m)"],
        ):
            x = xyz[:, 0]
            y = xyz[:, 1]
            z = xyz[:, 2]

            # === 每帧动态计算 xlim（少留白）===
            if len(x) > 0:
                x_min, x_max = x.min(), x.max()
                margin_x = 2.0  # 左右各留 2 米
                xlim = (x_min - margin_x, x_max + margin_x)
            else:
                xlim = (-10, 10)
            ax.set_xlim(xlim)
            ax.set_ylim(fixed_ylim)
            # title 不要贴着边框，字体大小调大
            ax.set_title(title, pad=20, fontsize=16)
            ax.axis("off")
            # === 用 Z 坐标着色 ===
            colors = np.clip(intensity, 0, 1)
            ax.scatter(x, y, c=colors, s=point_size, cmap="jet", vmin=0, vmax=1)
        plt.tight_layout(pad=0)
        plt.savefig("temp_frame.png", bbox_inches="tight", pad_inches=0)
        plt.close()
        frame = cv2.imread("temp_frame.png")
        if frame is not None:
            frame = cv2.resize(frame, (width * 2, height))
            video_writer.write(frame)
    video_writer.release()
    if os.path.exists("temp_frame.png"):
        os.remove("temp_frame.png")


if __name__ == "__main__":
    render_dir = "/home/not0513/data/20250702_133223_Q2517/renders"
    gt_dir = "/home/not0513/data/20250702_133223_Q2517/gt"
    video_dir = "/home/not0513/data/20250702_133223_Q2517/video"

    ylim = [-40, 30]

    print("Generating BEV render video...")
    create_top_view_video(render_dir, "render_bev.mp4", fixed_ylim=ylim)
    print("Generating BEV GT video...")
    create_top_view_video(gt_dir, "gt_bev.mp4", fixed_ylim=ylim)
    print("Generating GT vs Render BEV video...")
    create_gt_render_view_video(
        render_dir, gt_dir, "gt_vs_render_bev.mp4", fixed_ylim=ylim
    )

    print(f"\n✅ All done!")
    print(f"   BEV Videos:  {video_dir}/render_bev.mp4, gt_bev.mp4")
