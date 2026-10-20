#!/usr/bin/env python3
"""
从 proto_descriptors.bin（FileDescriptorSet）加载描述符，提供动态消息类，
不依赖 onboard/offboard 等 proto 源码目录。
"""

import sys
from pathlib import Path

from google.protobuf import descriptor_pb2
from google.protobuf import descriptor_pool
from google.protobuf import reflection

# 使用独立 pool，不污染默认 symbol_database
_pool = None
_message_classes = {}  # full_name -> class


def _get_descriptor_path():
    """默认描述符文件：与脚本同目录下的 proto_descriptors.bin；打包运行时从 sys._MEIPASS 查找"""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).resolve().parent
    return base / "proto_descriptors.bin"


def load_descriptor_file(path=None):
    """
    加载 FileDescriptorSet 到独立 DescriptorPool。
    path: 描述符文件路径，默认使用项目根目录下的 proto_descriptors.bin
    """
    global _pool, _message_classes
    if path is None:
        path = _get_descriptor_path()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"描述符文件不存在: {path}")

    with open(path, "rb") as f:
        fds = descriptor_pb2.FileDescriptorSet.FromString(f.read())  # pylint: disable=no-member

    _pool = descriptor_pool.DescriptorPool()
    # FileDescriptorSet 通常已是依赖顺序，按序加入
    for fp in fds.file:
        _pool.Add(fp)
    _message_classes = {}
    return _pool


def get_pool(path=None):
    """获取或初始化 DescriptorPool。"""
    global _pool
    if _pool is None:
        load_descriptor_file(path)
    return _pool


def get_message_class(full_name: str, path=None):
    """
    按完整类型名获取动态消息类，例如 'qcraft.LiteMsgWrapper'、'qcraft.EncodedImageMetadata'。
    返回可用于 ParseFromString / MessageToJson 的 Message 子类，若不存在则返回 None。
    """
    global _message_classes
    pool = get_pool(path)
    if full_name in _message_classes:
        return _message_classes[full_name]
    try:
        desc = pool.FindMessageTypeByName(full_name)
    except KeyError:
        return None
    if desc is None:
        return None
    
    # 为同一 descriptor 只生成一次类并缓存
    klass = reflection.MakeClass(desc)

    _message_classes[full_name] = klass
    return klass


def has_message(full_name: str, path=None) -> bool:
    """检查描述符集中是否包含该消息类型。"""
    return get_message_class(full_name, path) is not None
