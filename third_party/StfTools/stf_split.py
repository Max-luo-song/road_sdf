#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STF Split - STF文件时间切分工具
按照时间间隔将大的STF文件切分成多个小文件
不解析具体内容，只按时间切分原始数据
"""

import os
import sys
import argparse
import struct
import cramjam
from pathlib import Path
from typing import List, Dict
from stf_reader import StfReader

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
    
    def __init__(self, input_file: str, output_dir: str, interval_seconds: float):
        self.input_file = input_file
        self.output_dir = output_dir
        self.interval_microseconds = int(interval_seconds * 1_000_000)  # 转换为微秒
        
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def split(self):
        """执行切分"""
        print(f"📖 读取: {Path(self.input_file).name}")
        
        # 读取原始STF文件
        reader = StfReader(self.input_file)
        if not reader.open():
            print(f"无法打开STF文件: {self.input_file}", file=sys.stderr)
            return
        
        messages = reader.get_messages()
        if not messages:
            print("没有找到任何消息", file=sys.stderr)
            return
        
        # 按时间戳排序
        messages.sort(key=lambda x: x['timestamp'])
        
        # 获取时间范围
        start_time = messages[0]['timestamp']
        end_time = messages[-1]['timestamp']
        duration_seconds = (end_time - start_time) / 1_000_000
        
        # 计算切分数量
        num_splits = int((end_time - start_time) / self.interval_microseconds) + 1
        
        print(f"💾 切分: {len(messages)} 条消息 → {num_splits} 个文件 ({duration_seconds/60:.1f}分钟)")
        
        # 切分消息
        splits = self._split_messages(messages, start_time)
        
        # 写入文件
        self._write_splits(splits)
    
    def _split_messages(self, messages: List[Dict], start_time: int) -> List[List[Dict]]:
        """根据时间间隔切分消息"""
        splits = []
        current_split = []
        current_split_end = start_time + self.interval_microseconds
        
        for msg in messages:
            # 如果消息超出当前切分的时间范围
            while msg['timestamp'] >= current_split_end:
                if current_split:
                    splits.append(current_split)
                current_split = []
                current_split_end += self.interval_microseconds
            
            current_split.append(msg)
        
        # 添加最后一个切分
        if current_split:
            splits.append(current_split)
        
        return splits
    
    def _write_splits(self, splits: List[List[Dict]]):
        """写入切分后的文件"""
        input_filename = Path(self.input_file).stem
        total_size = 0
        
        for i, split_messages in enumerate(splits):
            if not split_messages:
                continue
            
            # 生成输出文件名
            output_filename = f"{input_filename}_part{i:03d}.stf"
            output_path = os.path.join(self.output_dir, output_filename)
            
            # 创建写入器
            writer = StfWriter(output_path)
            
            # 添加消息（保持原始数据）
            for msg in split_messages:
                writer.add_message(
                    timestamp=msg['timestamp'],
                    data=msg['data']
                )
            
            # 写入文件
            writer.write(use_compression=True)
            
            file_size = os.path.getsize(output_path)
            total_size += file_size
            
            # 显示进度（每10个或最后一个）
            if (i + 1) % 10 == 0 or i == len(splits) - 1:
                print(f"  {i + 1}/{len(splits)}", end='\r')
        
        print(f"\n✅ 完成: {self.output_dir} ({total_size / 1024 / 1024:.1f} MB)")

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
    parser.add_argument('--interval', '-i', type=float, required=True,
                       help='切分时间间隔（秒），例如: 20, 60, 300')
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
    
    # 检查时间间隔
    if args.interval <= 0:
        print(f"错误: 时间间隔必须大于0", file=sys.stderr)
        sys.exit(1)
    
    # 执行切分
    splitter = StfSplitter(args.input_file, args.output, args.interval)
    splitter.split()

if __name__ == '__main__':
    main()

