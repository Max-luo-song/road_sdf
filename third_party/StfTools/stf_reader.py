#!/usr/bin/env python3
"""STF Reader"""

import struct
import cramjam
import zstandard as zstd
from typing import Optional, Tuple, Iterator

class StfReader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_size = 0
        self.data = None
        self.messages = []
        self._zstd_dctx = zstd.ZstdDecompressor()
        
    def open(self) -> bool:
        try:
            with open(self.file_path, 'rb') as f:
                self.data = f.read()
            self.file_size = len(self.data)
            
            if self.file_size > 4 and self.data[:4] == b'QSTF':
                return self._parse_simplified_format()
            
            if self.file_size <= 48:
                raise ValueError("File too small")
            
            self._parse_leveldb_table()
            return True
        except:
            return False
    
    def _parse_leveldb_table(self):
        footer = self._parse_footer()
        index_block_data = self._parse_index_block(footer['index_handle'])
        data_blocks = self._parse_data_blocks(index_block_data)
        self._read_all_messages(data_blocks)
    
    def _parse_footer(self) -> dict:
        footer_offset = self.file_size - 48
        footer_data = self.data[footer_offset:]
        metaindex_offset, pos = self._read_varint(footer_data, 0)
        metaindex_size, pos = self._read_varint(footer_data, pos)
        index_offset, pos = self._read_varint(footer_data, pos)
        index_size, pos = self._read_varint(footer_data, pos)
        return {'index_handle': {'offset': index_offset, 'size': index_size}}
    
    def _parse_index_block(self, index_handle: dict) -> bytes:
        offset = index_handle['offset']
        size = index_handle['size']
        
        # 读取block数据（+5字节：1字节compression type + 4字节CRC）
        block_data = self.data[offset:offset + size]
        compression_type = self.data[offset + size]
        
        # 解压缩
        if compression_type == 0:
            # 无压缩
            return block_data
        elif compression_type == 1:
            # Snappy压缩
            try:
                decompressed = cramjam.snappy.decompress_raw(block_data)
                return bytes(decompressed)
            except:
                return block_data
        elif compression_type == 2:
            # Zstandard压缩（复用实例，避免每块新建）
            try:
                return self._zstd_dctx.decompress(block_data)
            except Exception:
                return block_data
        else:
            return block_data

    def _parse_data_blocks(self, index_block: bytes) -> list:
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
    
    def _read_all_messages(self, data_blocks: list):
        """读取所有消息"""
        for block_handle in data_blocks:
            block_data = self._read_data_block(block_handle)
            if block_data:
                self._parse_block_entries(block_data)
    
    def _read_data_block(self, block_handle: dict) -> Optional[bytes]:
        """读取并解压data block"""
        offset = block_handle['offset']
        size = block_handle['size']
        
        if offset + size > self.file_size:
            return None
        
        block_data = self.data[offset:offset + size]
        compression_type = self.data[offset + size] if offset + size < self.file_size else 0
        
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

    def _parse_block_entries(self, block_data: bytes):
        """解析block中的entries"""
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
                        
                        self.messages.append({
                            'timestamp': timestamp,
                            'tag_number': tag_number,
                            'data': value
                        })
                except:
                    pass
            except:
                break
    
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
    
    def get_messages(self) -> list:
        """获取所有消息"""
        return self.messages
    
    def __iter__(self) -> Iterator[dict]:
        """迭代所有消息"""
        return iter(self.messages)
    
    def _parse_simplified_format(self) -> bool:
        """解析简化格式的STF文件（由stf_split.py生成）"""
        try:
            pos = 0
            # 读取header
            magic = self.data[pos:pos+4]
            pos += 4
            if magic != b'QSTF':
                return False
            
            version = struct.unpack('<I', self.data[pos:pos+4])[0]
            pos += 4
            num_messages = struct.unpack('<Q', self.data[pos:pos+8])[0]
            pos += 8
            compression_flag = struct.unpack('B', self.data[pos:pos+1])[0]
            pos += 1
            
            # 读取原始数据大小
            original_size = struct.unpack('<Q', self.data[pos:pos+8])[0]
            pos += 8
            
            # 读取消息数据
            messages_data = self.data[pos:]
            
            # 解压缩
            if compression_flag:
                try:
                    decompressed = cramjam.snappy.decompress_raw(messages_data)
                    messages_data = bytes(decompressed)
                except:
                    pass
            
            # 解析消息
            pos = 0
            for _ in range(num_messages):
                if pos + 12 > len(messages_data):
                    break
                
                timestamp = struct.unpack('<Q', messages_data[pos:pos+8])[0]
                pos += 8
                data_length = struct.unpack('<I', messages_data[pos:pos+4])[0]
                pos += 4
                
                if pos + data_length > len(messages_data):
                    break
                
                data = messages_data[pos:pos+data_length]
                pos += data_length
                
                # 尝试解析tag_number（可选）
                tag_number = None
                if len(data) >= 2 and data[0] == 0x08:
                    tag_number, _ = self._read_varint(data, 1)
                
                self.messages.append({
                    'timestamp': timestamp,
                    'tag_number': tag_number,
                    'data': data
                })
            
            return True
        except Exception:
            return False
