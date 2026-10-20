# 用于模拟客户端发送Protobuf消息到server，不部署在我们的系统中
# -*- coding: utf-8 -*-
import time

import Pose_pb2
import socket
import random
import select
import re
import os
from infer_utils import QCRAFT_CAMERA_DICT, QCRAFT_LIDAR_DICT

# def data():
#     # 1. 创建Protobuf消息对象并填充数据
#     my_msg = Pose_pb2.MainCarInfo()
#     # x=7969.7074556988155, y=-2922.589950684925, yaw=1.9608628455457437,camera_id=CAM_PBQ_FRONT_RIGHT_RESET_OPTICAL_H99
#     my_msg.x = 7969.7074556988155
#     my_msg.y = -2922.589950684925
#     my_msg.z = -396.939
#     my_msg.yaw = 1.9608628455457437
#     my_msg.roll = 0
#     my_msg.pitch = 0
#     my_msg.timestamp = 1761382841.901
#     my_msg.camera_id = "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H110"
#     print(my_msg.x, my_msg.y, my_msg.z, my_msg.yaw)
#     return my_msg

def data():
    # 1. 创建Protobuf消息对象并填充数据
    my_msg = Pose_pb2.MainCarInfo()
    # x=7969.7074556988155, y=-2922.589950684925, yaw=1.9608628455457437,camera_id=CAM_PBQ_FRONT_RIGHT_RESET_OPTICAL_H99
    my_msg.x = 7987.08639
    my_msg.y = -2964.6539
    my_msg.z = -396.08292
    my_msg.yaw = 1.9608628455457437
    my_msg.roll = 0.000000
    my_msg.pitch = 0.000000
    my_msg.timestamp = 1761382840.0010262
    # my_msg.sensor_id = "LDR_FRONT"
    my_msg.sensor_id = "CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H99"
    print(my_msg.x, my_msg.y, my_msg.z, my_msg.yaw)
    return my_msg

class TCPClient:
    #def __init__(self, host='0.0.0.0', port=9999):
    def __init__(self, host='0.0.0.0', port=22):
        self.host = host
        self.port = port
        self.socket = None

    def connect(self):
        """连接到服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            print(f"已连接到服务器 {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False

    def send_message_pb(self):
        """发送消息到服务器"""
        try:
            if not self.socket:
                print("未连接到服务器")
                return False

            # 2. 序列化为字节串
            my_msg = data()
            serialized_data = my_msg.SerializeToString()
            print(serialized_data)
            # 3. 计算长度并打包为4字节的消息头 (使用大端字节序)
            data_length = len(serialized_data)
            length_header = data_length.to_bytes(4, byteorder='big')  # 4字节头
            # 先发送长度头，再发送序列化数据
            self.socket.sendall(length_header + serialized_data)
            #self.socket.send(message.encode('utf-8'))

            #time.sleep(100)
            # 接收响应
            #response = self.socket.recv(1024)
            #print(f"服务器响应: {response.decode('utf-8')}")
            print(f"服务器响应 OK")
            return True

        except Exception as e:
            print(f"发送消息失败: {e}")
            return False

    def recv_all(self, n: int) -> bytes:
        """从 socket 中严格读取 n 字节，若连接提前关闭则抛出异常。"""
        data = b""
        while len(data) < n:
            packet = self.socket.recv(n - len(data))
            if not packet:
                raise ConnectionError("连接在读取时被关闭")
            data += packet
        return data

    def _save_image(self, data_bytes: bytes, name: str, index: int, img_dir: str) -> str:
        ext = ''
        if '.' in name:
            ext = os.path.splitext(name)[1]
        else:
            if data_bytes[:4] == b'\x89PNG':
                ext = '.png'
            elif data_bytes[:3] == b'\xff\xd8\xff':
                ext = '.jpg'
            else:
                ext = '.bin'
        timestamp = time.time()
        t = int(timestamp)
        ts = int((timestamp - t) * 1000)
        fname = os.path.join(img_dir, f"cam_{index}_{t}_{ts}{ext}")
        with open(fname, 'wb') as wf:
            wf.write(data_bytes)
        return fname

    def _save_lidar(self, data_bytes: bytes, name: str, lidar_dir: str) -> str:
        fname = os.path.join(lidar_dir, name)
        with open(fname, 'wb') as wf:
            wf.write(data_bytes)
        return fname

    def receive_items(self) -> bool:
        """接收服务器按 4 字节长度前缀 + protobuf 消息发送的响应。

        阻塞读取：只要服务器继续发送数据就一直等待并处理，直到连接被对端关闭或出现严重错误。
        """
        try:
            out_dir = './realtime_output/received'
            img_dir = os.path.join(out_dir, 'images')
            lidar_dir = os.path.join(out_dir, 'lidar')
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lidar_dir, exist_ok=True)
            
            try:
                header = self.recv_all(4)
            except ConnectionError:
                # 对端已关闭连接，结束读取
                print("连接在读取时被关闭，接收结束")
                return False

            length = int.from_bytes(header, 'big')
            print(f"准备接收长度为 {length} 的数据")
            if length <= 0:
                print(f"收到无效长度 {length}, 放弃本次解析")
                return False

            # 防御性保护，防止异常的大长度导致 OOM
            if length > 500 * 1024 * 1024:
                print(f"收到过大长度 {length}, 放弃本次解析")
                return False

            data = self.recv_all(length)
            proto = Pose_pb2.MainCarInfo()
            try:
                proto.ParseFromString(data)
            except Exception:
                print('Failed to parse protobuf response')
                return False

            sensor = getattr(proto, 'sensor_id', '') or ''
            print(f"收到传感器ID: {sensor}")
            
            prefix = sensor[:3].lower() if sensor else ''

            # build filename from timestamp if available, format seconds_milliseconds
            tsf = time.time()
            s = int(tsf)
            ms = int((tsf - s) * 1000)

            if prefix == 'cam':
                cam_id = QCRAFT_CAMERA_DICT[sensor]
                print(f"处理相机ID: {cam_id}, 传感器名称: {sensor}")
                fname = self._save_image(getattr(proto, 'sensor_data', b''), f'img_{cam_id}_{s}_{ms:03d}.png', cam_id, img_dir)
                print(f"收到图片并保存为: {fname}")
            elif prefix == 'ldr':
                lidar_id = QCRAFT_LIDAR_DICT[sensor]
                print(f"处理Lidar ID: {lidar_id}, 传感器名称: {sensor}")
                fname = self._save_lidar(getattr(proto, 'sensor_data', b''), f'ldr_{lidar_id}_{s}_{ms:03d}.pcd', lidar_dir)
                print(f"收到Lidar文件并保存为: {fname}")
            else:
                # unknown: write raw bytes
                fname = os.path.join(out_dir, f"{s}_{ms:03d}.bin")
                with open(fname, 'wb') as wf:
                    wf.write(getattr(proto, 'sensor_data', b''))
                print(f"未知类型响应，保存为: {fname}")

            return True
        except Exception as e:
            print(f"接收项目时出错: {e}")
            return False

    def send_message(self, message):
        """发送消息到服务器"""
        try:
            if not self.socket:
                print("未连接到服务器")
                return False

            self.socket.send(message.encode('utf-8'))

            # 接收响应
            response = self.socket.recv(1024)
            print(f"服务器响应: {response.decode('utf-8')}")
            return True

        except Exception as e:
            print(f"发送消息失败: {e}")
            return False

    def close(self):
        """关闭连接"""
        if self.socket:
            self.socket.close()
            print("连接已关闭")

if __name__ == "__main__":
    client = TCPClient()

    if client.connect():
        # 发送测试消息
        messages = ["Hello, Server!", "测试消息2", "测试消息3"]

        for msg in range(1,11):
            # 发送 protobuf 消息
            ok = client.send_message_pb()
            if not ok:
                print("发送失败，跳出循环")
                break

            # receive
            client.receive_items()
            time.sleep(1.0)

        client.close()
