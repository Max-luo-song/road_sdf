#!/usr/bin/env python3
"""STF Extract"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Optional, Dict
from stf_reader import StfReader

class StfExtractor:
    RESERVED_TAG_NUMBERS = {4, 12, 18, 26, 31, 36, 38, 39, 55, 63, 64, 65, 
                            76, 78, 99, 111, 112, 118, 123, 128, 212, 213, 261}
    
    TAG_TO_TOPIC = {
        2: 'pose_proto',
        3: 'trajectory_proto',
        5: 'system_status_proto',
        6: 'shm_message_metadata',
        7: 'imu_raw_reading_proto',
        8: 'canbus_proto',
        11: 'chassis',
        25: 'chassis_detail',
    }
    
    def __init__(self, output_dir: str, topics_filter: Optional[List[str]] = None):
        self.output_dir = output_dir
        self.topics_filter = set(topics_filter) if topics_filter else None
    
    def extract_from_lite_msg_wrapper(self, data: bytes) -> Optional[Dict]:
        if len(data) < 2:
            return None
        
        pos = 0
        tag_number = None
        inner_message = None
        
        while pos < len(data):
            # 读取tag
            if pos >= len(data):
                break
            
            tag_byte = data[pos]
            field_number = tag_byte >> 3
            wire_type = tag_byte & 0x07
            pos += 1
            
            if field_number == 1 and wire_type == 0:
                # tag_number字段
                value, pos = self._read_varint(data, pos)
                tag_number = value
                
            elif field_number > 1 and wire_type == 2:
                # 内层消息（length-delimited）
                length, pos = self._read_varint(data, pos)
                if pos + length <= len(data):
                    inner_message = data[pos:pos + length]
                    pos += length
                else:
                    break
            else:
                # 跳过其他字段
                if wire_type == 0:  # varint
                    _, pos = self._read_varint(data, pos)
                elif wire_type == 2:  # length-delimited
                    length, pos = self._read_varint(data, pos)
                    pos += length
                else:
                    break
        
        if tag_number is not None and inner_message is not None:
            return {
                'tag_number': tag_number,
                'inner_message': inner_message
            }
        
        return None
    
    def _read_varint(self, data: bytes, offset: int) -> tuple:
        """读取varint"""
        result = 0
        shift = 0
        pos = offset
        
        while pos < len(data):
            byte = data[pos]
            result |= (byte & 0x7F) << shift
            pos += 1
            
            if (byte & 0x80) == 0:
                return result, pos
            
            shift += 7
            if shift >= 64:
                break
        
        return result, pos
    
    def convert_stf_file(self, stf_file: str, run_name: str):
        """转换STF文件"""
        reader = StfReader(stf_file)
        if not reader.open():
            return
        
        messages = reader.get_messages()
        
        # 按topic分组
        topic_messages = {}
        converted_count = 0
        skipped_count = 0
        
        for stf_msg in messages:
            tag_number = stf_msg['tag_number']
            
            # 跳过保留字段
            if tag_number in self.RESERVED_TAG_NUMBERS:
                skipped_count += 1
                continue
            
            # 获取topic名称
            topic = self.TAG_TO_TOPIC.get(tag_number)
            if not topic:
                topic = f'unknown_{tag_number}'
                skipped_count += 1
                continue
            
            # 应用topic过滤
            if self.topics_filter and topic not in self.topics_filter:
                skipped_count += 1
                continue
            
            # 提取内层消息
            extracted = self.extract_from_lite_msg_wrapper(stf_msg['data'])
            if not extracted:
                skipped_count += 1
                continue
            
            if topic not in topic_messages:
                topic_messages[topic] = []
            
            # 保存为JSON（inner_message转为hex）
            topic_messages[topic].append({
                'timestamp': stf_msg['timestamp'],
                'topic': topic,
                'tag_number': tag_number,
                'message_hex': extracted['inner_message'].hex(),
                'message_size': len(extracted['inner_message'])
            })
            
            converted_count += 1
        
        # 写入文件
        self._write_files(topic_messages, run_name)
    
    def _write_files(self, topic_messages: Dict, run_name: str):
        """写入文件"""
        for topic, messages in topic_messages.items():
            topic_dir = os.path.join(self.output_dir, run_name, topic)
            os.makedirs(topic_dir, exist_ok=True)
            
            for msg in messages:
                timestamp = msg['timestamp']
                filename = f"{timestamp:016d}.json"
                filepath = os.path.join(topic_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(msg, f, indent=2, ensure_ascii=False)
    
    def convert_run_directory(self, run_dir: str):
        """转换RUN目录"""
        run_name = os.path.basename(run_dir)
        
        lite_msg_path = os.path.join(run_dir, 'lite_msg.stf')
        if not os.path.exists(lite_msg_path):
            return
        
        self.convert_stf_file(lite_msg_path, run_name)

def main():
    parser = argparse.ArgumentParser(description='STF消息提取器')
    parser.add_argument('--stf_file', type=str, help='STF文件路径')
    parser.add_argument('--run_dir', type=str, help='RUN目录路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--topics', type=str, help='要提取的topic列表（逗号分隔）')
    
    args = parser.parse_args()
    
    if not args.stf_file and not args.run_dir:
        parser.error('必须指定 --stf_file 或 --run_dir')
    
    topics_filter = None
    if args.topics:
        topics_filter = [t.strip() for t in args.topics.split(',')]
    
    extractor = StfExtractor(args.output_dir, topics_filter)
    
    if args.stf_file:
        run_name = Path(args.stf_file).stem
        extractor.convert_stf_file(args.stf_file, run_name)
    elif args.run_dir:
        extractor.convert_run_directory(args.run_dir)

if __name__ == '__main__':
    main()
