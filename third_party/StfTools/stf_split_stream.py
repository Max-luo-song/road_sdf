#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STF Split - STF文件时间切分工具
按照时间间隔将大的STF文件切分成多个小文件
--使用流式读取切分
不解析具体内容，只按时间切分原始数据
"""

import os
import re
import sys
import argparse
import struct
import cramjam
from pathlib import Path
from typing import List, Dict
from stf_reader import StfReader
from stf_reader_stream import StfReader

class StfWriter:
    """STF文件写入器（简化但兼容的格式）"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.messages = []
    
    def add_message(self, timestamp: int, data: bytes):
        """添加一条消息（保持原始数据不变）"""
        self.messages.append({
            'timestamp': timestamp,
            'data': data
        })
    
    def write(self, use_compression: bool = True):
        """写入STF文件（简化但兼容的格式）
        
        文件格式（兼容StfReader读取）:
        - Header: "QSTF" (4字节) + version (4字节) + num_messages (8字节) + compression_flag (1字节)
        - 每条消息:
            - timestamp (8字节, uint64)
            - data_length (4字节, uint32)
            - data (变长，原始protobuf数据)
        """
        with open(self.file_path, 'wb') as f:
            # 写入header
            f.write(b'QSTF')  # Magic number
            f.write(struct.pack('<I', 1))  # Version
            f.write(struct.pack('<Q', len(self.messages)))  # Number of messages
            f.write(struct.pack('B', 1 if use_compression else 0))  # Compression flag
            
            # 准备消息数据
            messages_data = bytearray()
            for msg in self.messages:
                messages_data.extend(struct.pack('<Q', msg['timestamp']))
                messages_data.extend(struct.pack('<I', len(msg['data'])))
                messages_data.extend(msg['data'])
            
            # 压缩（可选）
            if use_compression:
                try:
                    compressed = cramjam.snappy.compress_raw(bytes(messages_data))
                    compressed_data = bytes(compressed)
                    # 写入原始大小和压缩数据
                    f.write(struct.pack('<Q', len(messages_data)))  # Original size
                    f.write(compressed_data)
                except Exception as e:
                    print(f"警告: 压缩失败，使用未压缩格式: {e}", file=sys.stderr)
                    f.write(struct.pack('<Q', len(messages_data)))  # Original size = compressed size
                    f.write(messages_data)
            else:
                f.write(struct.pack('<Q', len(messages_data)))  # Original size = compressed size
                f.write(messages_data)


class StfSplitter:
    """STF文件切分器"""
    
    def __init__(self, input_file: str, 
                 output_dir: str, 
                 interval_seconds: float = None,
                 slice_start_time: int = None,
                 slice_end_time: int = None ):
        
        self.input_file = input_file
        self.output_dir = output_dir
        self.interval_microseconds = None
        self.slice_start_time = None
        self.slice_end_time = None

        if interval_seconds:
            self.interval_microseconds = int(interval_seconds * 1_000_000)  # 转换为微秒
        elif slice_start_time is not None and slice_end_time is not None:
            self.slice_start_time = int(slice_start_time)
            self.slice_end_time = int(slice_end_time)
        
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def split(self):
        """执行流式切分"""
        print(f"\n📖 读取: {Path(self.input_file).name}")
        
        # 读取原始STF文件
        reader = StfReader(self.input_file)
        if not reader.open():
            print(f"无法打开STF文件: {self.input_file}", file=sys.stderr)
            return
        
        if self.interval_microseconds:
            # 流式切分处理
            stats = self._split_streaming(reader)

            if stats['total_messages'] == 0:
                print("没有找到任何消息", file=sys.stderr)
                return
            
            print(f"\n✅ 完成: {self.output_dir} ({stats['total_size'] / 1024 / 1024:.1f} MB)")
            print(f"\n📊 统计: 共{stats['total_messages']} 条消息 → {stats['num_splits']} 个文件 ({stats['duration_seconds']/60:.1f}分钟)")
        
        elif self.slice_start_time is not None and self.slice_end_time is not None:
            # 流式切片
            stats = self._slice_streaming(reader)

            if stats['slice_messages'] == 0:
                print("没有找到任何消息", file=sys.stderr)
                return

            print(f"✅ 完成: {self.output_dir} ({stats['slice_size'] / 1024 / 1024:.1f} MB)\n共{stats['slice_messages']} 条消息")

    def _slice_streaming(self, reader: StfReader):
        '''根据起止时间切片,并写入数据'''
        # ========参数初始化=======
        # stf文件首尾的时间戳
        first_ts = None

        # 切片的起点与持续时间        
        slice_offset_us = int(self.slice_start_time * 1_000_000)
        slice_interval_us = int((self.slice_end_time - self.slice_start_time) * 1_000_000)

        writer = None
        stats = {
            'slice_messages': 0,
            'slice_size': 0,
        }
        input_stem = Path(self.input_file).stem
        match = re.match(r'^(camera_jpg_img|lidar_data|lite_msg)_', input_stem)
        input_filename = match.group(1) if match else input_stem
        # ========参数初始化=======
        for msg in reader.get_messages_streaming():
            # 记录stf第一帧时间戳计算slice的起止
            if first_ts is None:
                # first_ts = msg['timestamp'] 并不是将第一帧视为ts = 0, ts = 0就是等于0
                first_ts = 0
                
                slice_start_us = first_ts + slice_offset_us
                slice_end_us = slice_start_us + slice_interval_us

            if slice_start_us <= msg['timestamp'] <= slice_end_us:
                # 切片内
                if writer == None:
                    output_file_path = os.path.join(self.output_dir, f"{input_filename}_{int(self.slice_start_time)}_{int(self.slice_end_time)}.stf")
                    writer = StfWriter(output_file_path)

                writer.add_message(
                    timestamp=msg['timestamp'],
                    data=msg['data']
                ) 
                stats['slice_messages'] += 1

            elif msg['timestamp'] > slice_end_us and writer is not None:               
                writer.write(use_compression=True)                
                stats['slice_size'] = os.path.getsize(output_file_path)

                writer = None

                break
         
         # 处理切片直到循环结束
        if writer is not None:
            writer.write(use_compression=True)
            stats['slice_size'] = os.path.getsize(output_file_path)

        return stats


    def _split_streaming(self, reader: StfReader):
        """根据interval切分,并写入数据"""
        # ========参数初始化=======
        current_split_end = None

        split_idx = 0
        first_ts = None
        last_ts = None

        writer = None

        stats = {
            'total_messages': 0,
            'num_splits': 0,
            'total_size': 0,
            'duration_seconds': 0
        }

        input_stem = Path(self.input_file).stem
        match = re.match(r'^(camera_jpg_img|lidar_data|lite_msg)_', input_stem)
        input_filename = match.group(1) if match else input_stem
        # ========参数初始化=======

        for msg in reader.get_messages_streaming():

            # 用来记录数据的第一帧与最后一帧时间戳
            if first_ts == None:
                first_ts = msg['timestamp']
            last_ts = msg['timestamp'] 

            # 记录共多少条msg
            stats['total_messages'] += 1

            if writer == None: # 新的切片,创建新的writer
                output_file_path = os.path.join(self.output_dir, f"{input_filename}_part{split_idx:03d}.stf")
                writer = StfWriter(output_file_path)
                current_split_end = msg['timestamp'] + self.interval_microseconds

                # 添加消息(包括创建切片,writer的第一帧),如果用else就不会加入第一帧
                writer.add_message(
                    timestamp=msg['timestamp'],
                    data=msg['data']
                )
            elif msg['timestamp'] < current_split_end:
                 writer.add_message(
                    timestamp=msg['timestamp'],
                    data=msg['data']
                )
            elif msg['timestamp'] >= current_split_end:
                # 写入文件
                writer.write(use_compression=True)
                
                file_size = os.path.getsize(output_file_path)
                stats['total_size'] += file_size
                stats['num_splits'] += 1

                # 开始下一切片
                split_idx += 1
                # 创建新 writer
                output_file_path = os.path.join(self.output_dir, f"{input_filename}_part{split_idx:03d}.stf")
                writer = StfWriter(output_file_path)
                current_split_end = msg['timestamp'] + self.interval_microseconds
                
                # 添加当前消息到新切片
                writer.add_message(
                    timestamp=msg['timestamp'],
                    data=msg['data']
                )

        # 循环结束,处理最后一个切片
        if writer is not None:
            writer.write(use_compression=True)

            file_size = os.path.getsize(output_file_path)
            stats['total_size'] += file_size
            stats['num_splits'] += 1

        if first_ts and last_ts:
                stats['duration_seconds'] = (last_ts - first_ts) / 1_000_000

        return stats


def main():
    parser = argparse.ArgumentParser(
        description='STF文件时间切分工具 - 按时间间隔切分STF数据包',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 将2分钟的数据包切分为20秒的小包（生成6个文件）
  python3 stf_split.py input.stf --interval 20 --output output_dir
  
  # 将10分钟的数据包切分为1分钟的小包
  python3 stf_split.py data/lite_msg.stf --interval 60 --output split_output
  
  # 将1小时的数据包切分为5分钟的小包
  python3 stf_split.py long_run.stf --interval 300 --output chunks
        """
    )
    
    parser.add_argument('input_file', type=str, help='输入STF文件路径')
    parser.add_argument('--interval', '-i', type=float, required=False, help='切分时间间隔（秒），例如: 20, 60, 300')
    parser.add_argument('--time_range', '-t', nargs=2,  type=int, metavar=('START', 'END'), 
                        required=False, help='切片的时间范围, 如 -t 30 60')
    parser.add_argument('--output', '-o', type=str, default='stf_split_output',
                    help='输出目录（默认: stf_split_output）')

    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input_file):
        print(f"错误: 输入文件不存在: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    if not args.input_file.endswith('.stf'):
        print(f"错误: 输入文件必须是.stf格式", file=sys.stderr)
        sys.exit(1)
    
    # 执行切分或切片
    if args.interval:
        # 检查时间间隔
        if args.interval <= 0:
            print(f"错误: 时间间隔必须大于0", file=sys.stderr)
            sys.exit(1)
        splitter = StfSplitter(args.input_file, args.output, interval_seconds=args.interval)
    elif args.time_range:
        start_time, end_time = args.time_range
        if start_time >= end_time:
            print(f"错误: 起始时间必须小于终止时间", file=sys.stderr)
            sys.exit(1)
        splitter = StfSplitter(args.input_file, args.output, slice_start_time=start_time, slice_end_time=end_time)

    splitter.split()

if __name__ == '__main__':
    main()
