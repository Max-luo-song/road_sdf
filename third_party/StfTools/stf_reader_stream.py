#!/usr/bin/env python3
"""STF Reader"""

import struct
import cramjam
import zstandard as zstd
from typing import Optional, Tuple, Iterator
import os

class StfReader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_size = 0
        self.data = None
        self.messages = []
        self._zstd_dctx = zstd.ZstdDecompressor()

        self._file_handle = None

    def open(self) -> bool:
        """流式模式：只读取索引，不读取数据"""
        try:
            # 保持文件句柄打开
            self._file_handle = open(self.file_path, 'rb')
            # 替代加载所有数据,通过len()求file_size
            self.file_size = os.path.getsize(self.file_path)

            if self.file_size > 4 and self._file_handle.read(4) == b'QSTF':
                    return self._parse_simplified_format()
            elif self.file_size <= 48:
                    raise ValueError("File too small")
            else:
                self._file_handle.seek(0)  # 回到开头
                self._parse_leveldb_table()
                return True
        except:
            return False
    
    def _parse_timestamp(self, key_str: str) -> int:
        """从key字符串中解析timestamp（微秒）"""
        # 移除前缀"0"
        clean_str = key_str[1:].split('_')[0]
        
        # 清理非数字字符
        timestamp_str = ''.join(c for c in clean_str if c.isdigit())
        
        if timestamp_str:
            return int(timestamp_str)
        return 0
    
    def _parse_tag_number(self, data: bytes) -> Optional[int]:
        """从protobuf数据中解析tag_number（LiteMsgWrapper的field 1）"""
        if len(data) < 2:
            return None
        
        # tag_number是field 1, wire_type 0 (varint)
        # tag = (field_number << 3) | wire_type = 8 (0x08)
        if data[0] == 0x08:
            tag_number, _ = self._read_varint(data, 1)
            return tag_number
        
        return None

    def _read_varint(self, data: bytes, offset: int) -> Tuple[int, int]:
        """读取varint编码的整数"""
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
        
    def _parse_leveldb_table(self):
        footer = self._parse_footer()
        index_block = self._parse_index_block(footer['index_handle'])
        self.data_blocks = self._parse_data_blocks(index_block)
    
    def _parse_footer(self) -> dict:
        footer_offset = self.file_size - 48

        self._file_handle.seek(footer_offset)
        footer_data = self._file_handle.read(48) # footer 固定48字节

        metaindex_offset, pos = self._read_varint(footer_data, 0)
        metaindex_size, pos = self._read_varint(footer_data, pos)
        index_offset, pos = self._read_varint(footer_data, pos)
        index_size, pos = self._read_varint(footer_data, pos)
        return {'index_handle': {'offset': index_offset, 'size': index_size}}
    
    def _parse_index_block(self, index_handle):
        offset = index_handle['offset']
        size = index_handle['size']
        
        # 读取block数据（+5字节：1字节compression type + 4字节CRC）
        # 定位到 block 数据起始位置
        self._file_handle.seek(offset)
        # 读取 block 数据（size 字节）
        index_block = self._file_handle.read(size)

        compression_type_byte = self._file_handle.read(1)  # 返回 bytes，不是 int
        compression_type = compression_type_byte[0] if compression_type_byte else 0
        
        # 解压缩
        if compression_type == 0:
            # 无压缩
            return index_block
        elif compression_type == 1:
            # Snappy压缩
            try:
                decompressed = cramjam.snappy.decompress_raw(index_block)
                return bytes(decompressed)
            except:
                return index_block
        elif compression_type == 2:
            # Zstandard压缩（复用实例，避免每块新建）
            try:
                return self._zstd_dctx.decompress(index_block)
            except Exception:
                return index_block
        else:
            return index_block

    def _parse_data_blocks(self, index_block):
        """解析data blocks（从index block中提取）"""    
        data_blocks = []
        
        # 获取restart points数量（最后4字节）
        if len(index_block) < 4:
            return data_blocks
        
        num_restarts = struct.unpack('<I', index_block[-4:])[0]
        if num_restarts == 0:
            return data_blocks
        
        # 解析index block中的entries
        pos = 0
        restart_offset = len(index_block) - 4 - (num_restarts * 4)
        
        while pos < restart_offset:
            try:
                # 读取entry: shared_key_len, non_shared_key_len, value_len, key_delta, value
                shared_len, pos = self._read_varint(index_block, pos)
                non_shared_len, pos = self._read_varint(index_block, pos)
                value_len, pos = self._read_varint(index_block, pos)
                
                if pos + non_shared_len + value_len > len(index_block):
                    break
                
                key_delta = index_block[pos:pos + non_shared_len]
                pos += non_shared_len
                value = index_block[pos:pos + value_len]
                pos += value_len
                
                # value是BlockHandle（offset + size）
                if len(value) >= 2:
                    block_offset, vpos = self._read_varint(value, 0)
                    block_size, vpos = self._read_varint(value, vpos)
                    
                    data_blocks.append({
                        'offset': block_offset,
                        'size': block_size
                    })
            except:
                break
        
        return data_blocks

# data_blocks = [{"offset":..., "size":...},
#                 {"offset":..., "size":...},
#                 {"offset":..., "size":...},
#                 {"offset":..., "size":...},
#                 {"offset":..., "size":...}
#                 ...,
#                 {"offset":..., "size":...},
#                 {"offset":..., "size":...}]

    def get_messages_streaming(self):
        """流式获取消息，内部自动解析 data_blocks"""
        if not hasattr(self, 'data_blocks'):
            # 如果没有 data_blocks，先解析
            footer = self._parse_footer()
            index_block = self._parse_index_block(footer['index_handle'])
            self.data_blocks = self._parse_data_blocks(index_block)
        
        try:
            for block_handle in self.data_blocks:
                block_data = self._read_data_block(block_handle)
                if block_data:  # 添加 None 检查
                    yield from self._parse_block_entries_streaming(block_data)
        except GeneratorExit:
            return
        
    def _read_data_block(self, block_handle):
        '''读取并解压一个data_block'''
        offset = block_handle['offset']
        size = block_handle['size']  

        if offset + size > self.file_size:
            return None      

        # 读取block数据（+5字节：1字节compression type + 4字节CRC）
        # 定位到 block 数据起始位置
        self._file_handle.seek(offset)
        # 读取 block 数据（size 字节）
        block_data = self._file_handle.read(size)

        compression_type_byte = self._file_handle.read(1)  # 返回 bytes，不是 int
        compression_type = compression_type_byte[0] if compression_type_byte else 0 

        # 解压缩
        if compression_type == 1:
            try:
                decompressed = cramjam.snappy.decompress_raw(block_data)
                return bytes(decompressed)
            except:
                return block_data
        elif compression_type == 2:
            try:
                return self._zstd_dctx.decompress(block_data)
            except Exception:
                return block_data
        else:
            return block_data
        
    def _parse_block_entries_streaming(self, block_data: bytes):
        """流式解析block中的entries，yield每条消息"""
        if len(block_data) < 4:
            return
        
        num_restarts = struct.unpack('<I', block_data[-4:])[0]
        if num_restarts == 0:
            return
        
        pos = 0
        restart_offset = len(block_data) - 4 - (num_restarts * 4)
        prev_key = b''

        while pos < restart_offset:
            try:
                # 读取entry
                shared_len, pos = self._read_varint(block_data, pos)
                non_shared_len, pos = self._read_varint(block_data, pos)
                value_len, pos = self._read_varint(block_data, pos)
                
                if pos + non_shared_len + value_len > len(block_data):
                    break
                
                key_delta = block_data[pos:pos + non_shared_len]
                pos += non_shared_len
                value = block_data[pos:pos + value_len]
                pos += value_len
                
                # 重构完整key
                key = prev_key[:shared_len] + key_delta
                prev_key = key
                
                # 检查是否是数据key（以"0"开头）
                try:
                    key_str = key.decode('utf-8', errors='ignore')
                    if key_str.startswith('0'):
                        # 解析timestamp
                        timestamp = self._parse_timestamp(key_str)
                        
                        # 解析tag_number
                        tag_number = self._parse_tag_number(value)
                        
                        # yield 消息，而不是添加到列表
                        yield {
                            'timestamp': timestamp,
                            'tag_number': tag_number,
                            'data': value
                        }
                except GeneratorExit:
                    return
            except Exception:
                break       

def main():
    from pathlib import Path
    folder_path = Path("/home/chery/Projects/Data/20260325_153656_QCOYSD968166/")
    file_list = ['camera_jpg_img.stf', 'lite_msg.stf', 'lidar_data.stf']
    file_path = folder_path / file_list[0]
    
    reader = StfReader(file_path)
    
    if reader.open():
        print(f"File:{file_path.name} opened successfully. Size: {reader.file_size} Bytes, {reader.file_size / (1024*1024*1024)}GB")
        
        # 流式读取消息
        count = 0
        for msg in reader.get_messages_streaming():
            count += 1
            if count <= 5:  # 只打印前5条
                print(f"Message {count}: timestamp={msg['timestamp']}, tag={msg['tag_number']}")
        
        print(f"Total messages: {count}")
    else:
        print("Failed to open file")

if __name__ == '__main__':
    main()