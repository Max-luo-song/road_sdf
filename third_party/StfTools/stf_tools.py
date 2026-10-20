#!/usr/bin/env python3
"""
STF 工具统一入口：通过子命令选择 to-json / extract-images / split。
用法:
  stf_tools to-json <stf文件或目录> [输出目录] [-t 线程数] [-w 进程数] [--topics TOPICS]
  stf_tools extract-images <stf文件> [-o 输出目录] [-t 线程数] [-d 解码线程] [-q 质量] [-l 数量]
  stf_tools split <stf文件> [-i 间隔秒] [-o 输出目录] [-c 块大小MB]
  stf_tools <子命令> --help  查看该子命令参数
"""

import sys
import os

# 保证从脚本所在目录解析导入（支持打包后运行）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


def _usage():
    print(__doc__.strip(), file=sys.stderr)
    print("\n子命令: to-json, extract-images, split", file=sys.stderr)


def main():
    if len(sys.argv) < 2:
        _usage()
        sys.exit(1)

    cmd = sys.argv[1].strip().lower()
    rest = sys.argv[2:]

    if cmd in ("to-json", "to_json"):
        if not rest:
            print("用法: stf_tools to-json <stf文件或目录> [输出目录]", file=sys.stderr)
            print("  默认输出目录: output", file=sys.stderr)
            sys.exit(1)
        if rest[0] in ("-h", "--help"):
            print("用法: stf_tools to-json <stf文件或目录> [输出目录] [-t 线程数] [-w 进程数] [--topics TOPICS]")
            print("  默认输出目录: output，默认线程数: 1")
            print("  -w/--workers  多进程数（>1 时多进程并行）")
            print("  --topics      仅输出指定 topic，逗号分隔，如 pose_proto,objects_proto 或 2,33")
            sys.exit(0)
        sys.argv = [sys.argv[0], *rest]
        import stf_to_json_full
        stf_to_json_full.main()
        return
    if cmd in ("extract-images", "extract_images", "extract-h265", "extract_h265"):
        sys.argv = [sys.argv[0], *rest]
        import stf_extract_images
        stf_extract_images.main()
        return
    if cmd == "split":
        sys.argv = [sys.argv[0], *rest]
        import stf_split_fast
        stf_split_fast.main()
        return

    print(f"未知子命令: {cmd!r}", file=sys.stderr)
    _usage()
    sys.exit(1)


if __name__ == "__main__":
    main()
