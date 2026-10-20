#!/usr/bin/env python3
"""
download_snippets.py — 从 OBS snippets 中查找覆盖目标时间段的 STF 文件并下载。

算法:
    1. 对 camera_jpg_img / lidar_data / lite_msg 三种 dtype 各自独立处理
    2. 列出 snippets 目录下该 scene 指定 dtype 的所有 .stf 分段
    3. 优先找单个 snippet 覆盖全范围 (start < clip_start 且 end > clip_end)
    4. 找不到则收集所有与 [clip_start, clip_end] 有重叠的 snippet
       重叠条件: st < clip_end 且 et > clip_start
       → 按 start_time 升序下载
    5. lite_run_info.pb.bin 模糊匹配，取该 scene 下任意一个

用法:
    python3 download_snippets.py \
        --scene_id 20260526_145008_QCJPSD851972 \
        --clip_start_time 350 \
        --clip_end_time 380 \
        --output_dir /path/to/raw/data/ \
        --obsutil tools/obsutil_linux_amd64_5.8.3/obsutil \
        --ak <OBS_AK> \
        --sk <OBS_SK> \
        --endpoint obs.cn-east-4.myhuaweicloud.com
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def run_obsutil(obsutil_path, ak, sk, endpoint, args, check=True):
    """运行 obsutil 命令，返回 stdout 文本。"""
    cmd = [
        obsutil_path,
        *args,
        f"-i={ak}",
        f"-k={sk}",
        f"-e={endpoint}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"[ERROR] obsutil 命令失败: {' '.join(cmd)}", file=sys.stderr)
        print(f"  stderr: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout


def list_snippets(obsutil_path, ak, sk, endpoint, snippet_dir, scene_id, dtype):
    """
    列出 snippets 目录下该 scene 的指定 dtype 分段。

    Args:
        dtype: 数据类型，如 "camera_jpg_img", "lidar_data", "lite_msg"

    Returns:
        list of (start_str, end_str, start_float, end_float)，按 start_time 升序排列
    """
    print(f"[INFO] Listing {snippet_dir} for {dtype} ...")
    stdout = run_obsutil(obsutil_path, ak, sk, endpoint,
                         ["ls", snippet_dir, "-limit=10000"], check=False)

    # 正则匹配: {scene_id}_{start}_{end}_{dtype}.stf
    pattern = re.compile(
        rf"{re.escape(scene_id)}_(?P<start>\d+\.\d+)_(?P<end>\d+\.\d+)_{re.escape(dtype)}\.stf"
    )

    snippets = []
    for line in stdout.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        st_str, et_str = m.group("start"), m.group("end")
        st, et = float(st_str), float(et_str)
        snippets.append((st_str, et_str, st, et))

    snippets.sort(key=lambda x: x[2])  # 按 start 升序
    print(f"[INFO] Found {len(snippets)} {dtype} snippet(s) for {scene_id}")
    return snippets


def find_snippets(snippets, clip_start, clip_end):
    """
    找到覆盖 [clip_start, clip_end] 的 snippet 列表。

    优先找单个 snippet 完全覆盖；找不到则收集所有重叠的。

    Returns:
        list of (start_str, end_str)，可能为空
    """
    # ---- 策略1: 单 snippet 完全覆盖 ----
    # 条件: st < clip_start 且 et > clip_end，取时长最短的
    single_candidates = [(s, e, sf, ef) for s, e, sf, ef in snippets
                         if sf < clip_start and ef > clip_end]
    if single_candidates:
        single_candidates.sort(key=lambda x: x[3] - x[2])  # 按时长升序
        s, e, _, _ = single_candidates[0]
        print(f"[INFO] Single snippet covers full range: {s} - {e}")
        return [(s, e)]

    # ---- 策略2: 跨 snippet 覆盖 ----
    # 条件: st < clip_end 且 et > clip_start
    print("[INFO] No single snippet covers full range, searching overlapping snippets...")
    overlapping = [(s, e, sf, ef) for s, e, sf, ef in snippets
                   if sf < clip_end and ef > clip_start]

    # 去除被其他片段完全包含的冗余片段 (如 70-83.3 完全在 60-120.5 内)
    if len(overlapping) > 1:
        filtered = []
        for i, (s, e, sf, ef) in enumerate(overlapping):
            redundant = any(
                sf >= other_sf and ef <= other_ef
                for j, (_, _, other_sf, other_ef) in enumerate(overlapping)
                if i != j
            )
            if not redundant:
                filtered.append((s, e, sf, ef))
        if len(filtered) < len(overlapping):
            print(f"[INFO] Removed {len(overlapping) - len(filtered)} "
                  f"redundant contained snippet(s)")
        overlapping = filtered

    if overlapping:
        print(f"[INFO] Found {len(overlapping)} overlapping snippet(s): "
              f"{[f'{s}-{e}' for s, e, _, _ in overlapping]}")
    else:
        print(f"[ERROR] No snippet overlaps with [{clip_start}, {clip_end}]")

    return [(s, e) for s, e, _, _ in overlapping]


def download_stf_files(obsutil_path, ak, sk, endpoint, snippet_dir, scene_id,
                       dtype, snippet_list, output_dir, clip_start, clip_end):
    """下载指定 dtype 匹配到的所有 .stf 文件。"""
    dest = Path(output_dir) / f"{scene_id}_{int(clip_start)}_{int(clip_end)}" / "download"
    dest.mkdir(parents=True, exist_ok=True)
    dest_str = str(dest) + "/"

    for s_start, s_end in snippet_list:
        src = f"{snippet_dir}/{scene_id}_{s_start}_{s_end}_{dtype}.stf"
        dst = f"{dest_str}{dtype}_{s_start}_{s_end}.stf"
        print(f"[INFO] Downloading {dtype}.stf ({s_start}-{s_end}) ...")
        run_obsutil(obsutil_path, ak, sk, endpoint,
                    ["cp", src, dst, "-f"], check=False)


def download_lite_run_info(obsutil_path, ak, sk, endpoint, snippet_dir, scene_id,
                           output_dir, clip_start, clip_end):
    """下载 lite_run_info.pb.bin (模糊匹配，取该 scene 下第一个)。"""
    dest = Path(output_dir) / f"{scene_id}_{int(clip_start)}_{int(clip_end)}"
    dest.mkdir(parents=True, exist_ok=True)
    dest_str = str(dest) + "/"

    print("[INFO] Looking for lite_run_info.pb.bin ...")

    # 策略1: 在 snippets 目录中模糊匹配
    downloaded = False
    try:
        stdout = run_obsutil(obsutil_path, ak, sk, endpoint,
                             ["ls", snippet_dir, "-limit=10000"], check=False)
        pb_pattern = re.compile(rf"{re.escape(scene_id)}.*lite_run_info\.pb\.bin")
        for line in stdout.splitlines():
            m = pb_pattern.search(line)
            if m:
                pb_path = line.strip().split()[-1]
                dst = f"{dest_str}lite_run_info.pb.bin"
                print(f"[INFO] Downloading lite_run_info.pb.bin from snippets: {pb_path}")
                run_obsutil(obsutil_path, ak, sk, endpoint,
                            ["cp", pb_path, dst, "-f"], check=False)
                downloaded = True
                break
    except Exception:
        pass

    # 策略2: snippets 中找不到, 尝试场景根目录
    if not downloaded:
        try:
            pb_path = f"obs://hwcn-hd2-ad-roaddata-new/{scene_id}/lite_run_info.pb.bin"
            dst = f"{dest_str}lite_run_info.pb.bin"
            print(f"[INFO] Downloading lite_run_info.pb.bin from scene root: {pb_path}")
            run_obsutil(obsutil_path, ak, sk, endpoint,
                        ["cp", pb_path, dst, "-f"], check=False)
            downloaded = True
        except Exception:
            pass

    if not downloaded:
        print("[WARNING] lite_run_info.pb.bin not found anywhere")


def main():
    parser = argparse.ArgumentParser(
        description="从 OBS snippets 下载覆盖目标时间段的 STF 数据"
    )
    parser.add_argument("--scene_id", type=str, required=True,
                        help="场景 ID，如 20260526_145008_QCJPSD851972")
    parser.add_argument("--clip_start_time", type=float, required=True,
                        help="目标时间段的起始时间 (秒)")
    parser.add_argument("--clip_end_time", type=float, required=True,
                        help="目标时间段的结束时间 (秒)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="下载目标目录 (如 /path/to/raw/)")
    parser.add_argument("--obsutil", type=str, required=True,
                        help="obsutil 可执行文件路径")
    parser.add_argument("--ak", type=str, required=True,
                        help="OBS Access Key")
    parser.add_argument("--sk", type=str, required=True,
                        help="OBS Secret Key")
    parser.add_argument("--endpoint", type=str, required=True,
                        help="OBS Endpoint")
    parser.add_argument("--obs_root", type=str,
                        default="obs://hwcn-hd2-ad-roaddata-new",
                        help="OBS 根路径")
    args = parser.parse_args()

    # 从 scene_id 提取日期 (前 8 位)
    date_str = args.scene_id[:8]
    snippet_list_dir = f"{args.obs_root}/snippets/{date_str}/{args.scene_id}"
    snippets_download_dir = f"{args.obs_root}/snippets/{date_str}"

    print(f"[INFO] Scene: {args.scene_id}")
    print(f"[INFO] Target time range: {args.clip_start_time} - {args.clip_end_time}")

    # 三种数据类型各自独立查找和下载
    dtypes = ["camera_jpg_img", "lidar_data", "lite_msg"]
    total_downloaded = 0

    for dtype in dtypes:
        print(f"\n[INFO] === Processing {dtype} ===")

        # 1. 列出该 dtype 的所有分段
        snippets = list_snippets(
            args.obsutil, args.ak, args.sk, args.endpoint,
            snippet_list_dir, args.scene_id, dtype,
        )

        if not snippets:
            print(f"[WARNING] 找不到 {args.scene_id} 的任何 {dtype} snippet，跳过")
            continue

        # 2. 查找匹配的 snippet(s)
        matched = find_snippets(snippets, args.clip_start_time, args.clip_end_time)

        if not matched:
            print(f"[WARNING] 找不到 {dtype} 覆盖 "
                  f"[{args.clip_start_time}-{args.clip_end_time}] 的 snippet，跳过")
            continue

        # 3. 下载该 dtype 的 .stf 文件
        download_stf_files(
            args.obsutil, args.ak, args.sk, args.endpoint,
            snippets_download_dir, args.scene_id,
            dtype, matched, args.output_dir,
            args.clip_start_time, args.clip_end_time,
        )
        total_downloaded += len(matched)

    if total_downloaded == 0:
        print(f"\n[ERROR] 所有 dtype 均未找到覆盖 "
              f"[{args.clip_start_time}-{args.clip_end_time}] 的 snippet",
              file=sys.stderr)
        sys.exit(1)

    # 4. 下载 lite_run_info.pb.bin (只需一个)
    download_lite_run_info(
        args.obsutil, args.ak, args.sk, args.endpoint,
        snippet_list_dir, args.scene_id, args.output_dir,
        args.clip_start_time, args.clip_end_time,
    )

    print(f"\n[INFO] Download completed ({total_downloaded} snippet(s) total)")


if __name__ == "__main__":
    main()
