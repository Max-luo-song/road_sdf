'''
处理覆盖的情况
如: clip的需求片段是40-80s
snippets片段之间有相覆盖的部分,如0-60.5, 60-120.5, 50-76.6
完全包含的部分在下载中已经剔除

'''

import argparse
import os
import re
import sys
from pathlib import Path
from stf_reader_stream import StfReader
from stf_split_stream import StfWriter, StfSplitter

def check_files_sole(download_dir):
    """检查必需文件是否存在且唯一"""
    download_dir = Path(download_dir)
    
    if not download_dir.is_dir():
        print(f"❌ 错误：目录不存在 - {download_dir}")
        return False
    
    file_names = [f.name for f in download_dir.iterdir() if f.is_file()]
    
    required_prefixes = ["camera_jpg_img", "lidar_data", "lite_msg"]
    
    for prefix in required_prefixes:
        matches = [f for f in file_names if f.startswith(prefix)]
        
        if len(matches) == 0:
            print(f"❌ 缺少以 '{prefix}' 开头的文件")
            return False
        elif len(matches) > 1:
            print(f"❌ 以 '{prefix}' 开头的文件有多个: {matches}")
            return False
    
    print(f"✅ 目录 {download_dir} 文件检查通过")
    return True

class StfMerger:
    'Stf的snippets合并器'
    def __init__(self, input_file_list: list, 
                 output_dir: str, 
                 clip_start_time: int = None,
                 clip_end_time: int = None ):
        
        self.input_file_list = input_file_list
        self.output_dir = output_dir
        self.clip_start_time = clip_start_time
        self.clip_end_time = clip_end_time

        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def merge_cut(self):
        # 从文件名提取类型前缀 (如 lidar_data_02640_02700.stf → lidar_data)
        input_filename = Path(self.input_file_list[0]).stem
        match = re.match(r'^(camera_jpg_img|lidar_data|lite_msg)_', input_filename)
        if not match:
            print(f"❌ 无法识别文件类型: {input_filename}")
            return
        prefix = match.group(1)

        slice_start_us = int(self.clip_start_time * 1_000_000)
        slice_end_us = int(self.clip_end_time * 1_000_000)

        # 收集所有在时间范围内的消息，按 timestamp 去重
        seen_timestamps = set()
        all_messages = []

        for file_path in self.input_file_list:
            print(f"[INFO] Reading {file_path} ...")
            reader = StfReader(file_path)
            reader.open()
            file_count = 0
            try:
                for msg in reader.get_messages_streaming():
                    ts = msg['timestamp']
                    if slice_start_us <= ts <= slice_end_us:
                        if ts not in seen_timestamps:
                            seen_timestamps.add(ts)
                            all_messages.append(msg)
                            file_count += 1
            finally:
                if reader._file_handle:
                    reader._file_handle.close()
            print(f"[INFO]   → {file_count} new message(s) from this snippet")

        if not all_messages:
            print(f"[WARNING] No messages in range [{self.clip_start_time}, {self.clip_end_time}]")
            return

        # 按时间戳排序 (不同 snippet 的消息交替出现)
        all_messages.sort(key=lambda m: m['timestamp'])

        output_file = os.path.join(
            self.output_dir,
            f"{prefix}_{int(self.clip_start_time)}_{int(self.clip_end_time)}.stf"
        )
        writer = StfWriter(output_file)
        for msg in all_messages:
            writer.add_message(timestamp=msg['timestamp'], data=msg['data'])
        writer.write(use_compression=True)

        print(f"[INFO] Merged {len(all_messages)} messages → {output_file}")



def main():
 
    # 输入snippets的下载路径
    # 输入clip的start 和 end
    parser = argparse.ArgumentParser(
        description='STF文件时间切分工具 - 按时间间隔切分STF数据包')
    
    parser.add_argument('--download_dir', type=str, help='输入STF文件路径')
    parser.add_argument("--clip_start_time", type=float, required=True,
                        help="目标时间段的起始时间 (秒)")
    parser.add_argument("--clip_end_time", type=float, required=True,
                        help="目标时间段的结束时间 (秒)")
    parser.add_argument('--output', '-o', type=str, default='stf_split_output',
                    help='输出目录（默认: stf_split_output）')
    args = parser.parse_args()

    if not os.path.exists(args.download_dir):
        print(f"❌ 错误：路径不存在 - {args.download_dir}")
        sys.exit(1)  # 退出码 1 表示错误

    download_dir = Path(args.download_dir)
    file_list = [str(f.absolute()) for f in download_dir.iterdir() if f.is_file() and not f.name.endswith('.pb.bin')]
    
    # 如果download_dir各类型文件只有一个,则不需要合并,只需要切分
    if check_files_sole(download_dir):
        print('只需要做切分,不需要合并')
        for file_path in file_list:
            splitter = StfSplitter(input_file=file_path,
                                   output_dir=args.output,
                                   slice_start_time=args.clip_start_time,
                                   slice_end_time=args.clip_end_time)
            splitter.split()
    # 相反如果download_dir中各类型文件有多个,说明需要合并并切分
    else:
        print("需要做snippets的合并")
        # 分别获取camera,lidar,lite_msg这些snippets的路径列表
        for dtype in ["camera_jpg_img", "lidar_data", "lite_msg"]:
            stf_path_list = [path for path in file_list if dtype in path]
            print(stf_path_list)
            print(f"合并{dtype}的snippets")
            merger = StfMerger(input_file_list=stf_path_list,
                            output_dir=args.output,
                            clip_start_time=args.clip_start_time,
                            clip_end_time=args.clip_end_time)
            merger.merge_cut()

            
            print(f"{dtype}合并处理完成")


    #merger 读取列表,然后将这些拼接在一起,返回一个时间上为start-end的stf
    pass

if __name__ == "__main__":
    main()

