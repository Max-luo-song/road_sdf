#!/usr/bin/env python3
"""STF图像提取工具（支持JPEG/H.265/H.264自动识别）"""

import os
import sys
import argparse
from pathlib import Path
import queue
import threading
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stf_reader import StfReader
from proto_from_descriptors import get_message_class

EncodedImageMetadata = get_message_class("qcraft.EncodedImageMetadata")
if EncodedImageMetadata is None:
    print("错误: proto_descriptors.bin 中缺少 qcraft.EncodedImageMetadata", file=sys.stderr)
    sys.exit(1)

# av / Pillow 为可选依赖，仅 H.265/H.264 解码时需要
_AV_AVAILABLE = False
try:
    import av
    _AV_AVAILABLE = True
except ImportError:
    pass

_PIL_AVAILABLE = False
try:
    from PIL import Image, ImageStat
    _PIL_AVAILABLE = True
except ImportError:
    pass

# 灰噪帧检测阈值：P帧缺少参考时解码出的残差图像几乎是均匀灰色
# 正常相机图像 max_stddev 通常 > 20，灰噪帧通常 < 5
_GRAY_STDDEV_THRESHOLD = 8.0
_GRAY_MEAN_CENTER = 128.0   # 灰噪帧均值聚集在 ~128（YUV 中性灰映射到 RGB）
_GRAY_MEAN_TOLERANCE = 30.0

ENCODE_JPEG = 0
ENCODE_H265 = 1
ENCODE_H264 = 2

CAMERA_ID_TO_NAME = {
    75: "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H110",
    76: "CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H60",
    77: "CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H30",
    97: "CAM_PBQ_FRONT_TELE_RESET_OPTICAL_H15",
    78: "CAM_PBQ_FRONT_LEFT_RESET_OPTICAL_H99",
    83: "CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H99",
    84: "CAM_PBQ_REAR_LEFT_RESET_OPTICAL_H30",
    79: "CAM_PBQ_FRONT_RIGHT_RESET_OPTICAL_H99",
    85: "CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H99",
    86: "CAM_PBQ_REAR_RIGHT_RESET_OPTICAL_H30",
    82: "CAM_PBQ_REAR_RESET_OPTICAL_H50",
}


class ImageExtractor:
    def __init__(self, output_dir: str, num_threads: int = 4,
                 decode_threads: int = 4, jpeg_quality: int = 95):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_threads = num_threads
        self.decode_threads = decode_threads
        self.jpeg_quality = jpeg_quality

        self.write_queue = queue.Queue(maxsize=200)

        self.timestamp_mapping = {}  # relative_timestamp -> cam_timestamp (去重)

        self.extracted_count = 0
        self.error_count = 0
        self.skipped_count = 0
        self.total_bytes = 0
        self.lock = threading.Lock()

    def _parse_qimgmeta(self, data: bytes, index: int) -> dict:
        """解析 QIMGMETA 帧头，返回元数据 dict，失败返回 None。"""
        if len(data) < 12 or data[:8] != b'QIMGMETA':
            return None
        try:
            meta_len = int.from_bytes(data[8:12], byteorder='little', signed=True)
            if meta_len <= 0 or meta_len > len(data) - 12:
                return None
            
            metadata = EncodedImageMetadata()
            metadata.ParseFromString(data[12:12 + meta_len])

            if metadata.camera_id in [k for k in CAMERA_ID_TO_NAME]:
                encoded_data = data[12 + meta_len:]
                has_info = metadata.HasField('image_info')
                encode_type = metadata.image_info.encode_type if has_info else ENCODE_JPEG

                # 用实际字节内容修正 encode_type：
                # proto 默认值为 0（JPEG），但实际可能是 H.265/H.264
                if encode_type == ENCODE_JPEG and len(encoded_data) >= 2:
                    if encoded_data[:2] != b'\xff\xd8':
                        if encoded_data[:4] == b'\x00\x00\x00\x01' or encoded_data[:3] == b'\x00\x00\x01':
                            encode_type = ENCODE_H265
                        else:
                            return None

                return {
                    'index': index,
                    'camera_id': metadata.camera_id or 'unknown',
                    'timestamp': f"{metadata.image_info.middle_row_exposure_timestamp:.4f}",
                    'encode_type': encode_type,
                    'encoded_data': encoded_data,
            }
        except Exception:
            return None

    # H.265 SPS 起始码特征（type=33 → first_byte=0x42）
    _HEVC_SPS_MARKERS = (b'\x00\x00\x01\x42', b'\x00\x00\x00\x01\x42')
    # H.265 IDR NALU 起始码特征（type=19→0x26, type=20→0x28）
    _HEVC_IDR_MARKERS = (
        b'\x00\x00\x01\x26', b'\x00\x00\x00\x01\x26',
        b'\x00\x00\x01\x28', b'\x00\x00\x00\x01\x28',
    )
    # H.264 SPS 起始码特征（type=7，覆盖常见 nal_ref_idc 变体）
    _H264_SPS_MARKERS = (
        b'\x00\x00\x01\x67', b'\x00\x00\x00\x01\x67',
        b'\x00\x00\x01\x27', b'\x00\x00\x00\x01\x27',
        b'\x00\x00\x01\x47', b'\x00\x00\x00\x01\x47',
        b'\x00\x00\x01\x07', b'\x00\x00\x00\x01\x07',
    )
    # H.264 IDR NALU 起始码特征（type=5，覆盖常见 nal_ref_idc 变体）
    _H264_IDR_MARKERS = (
        b'\x00\x00\x01\x65', b'\x00\x00\x00\x01\x65',
        b'\x00\x00\x01\x45', b'\x00\x00\x00\x01\x45',
        b'\x00\x00\x01\x25', b'\x00\x00\x00\x01\x25',
        b'\x00\x00\x01\x05', b'\x00\x00\x00\x01\x05',
    )

    def _extract_param_prefix(self, data: bytes, enc_type: int) -> bytes:
        """从含 SPS 的帧里提取参数集部分（IDR 切片之前的 VPS/SPS/PPS NALUs）。"""
        idr_markers = self._HEVC_IDR_MARKERS if enc_type != ENCODE_H264 else self._H264_IDR_MARKERS
        idr_pos = len(data)
        for m in idr_markers:
            pos = data.find(m)
            if 0 <= pos < idr_pos:
                idr_pos = pos
        return data[:idr_pos]

    def _decode_camera_stream(self, camera_id: str, frames: list) -> list:
        """将同一相机的所有帧拼成完整裸流后整体解码。
        
        frames: [(cam_timestamp, encoded_data, encode_type, relative_timestamp), ...]
        返回: [(cam_timestamp, jpeg_bytes, relative_timestamp), ...]
        """
        if not (_AV_AVAILABLE and _PIL_AVAILABLE) or not frames:
            return []

        # 提取基本信息（兼容3或4个元素）
        if len(frames[0]) == 4:
            enc_type = frames[0][2]
            relative_timestamps = [f[3] for f in frames]  # 保存所有 relative_timestamp
        else:
            enc_type = frames[0][2]
            relative_timestamps = [None] * len(frames)
        
        orig_indices = [f[0] for f in frames]  # cam_timestamp
        encoded_data_list = [f[1] for f in frames]
        
        fmts = ['h264', 'hevc'] if enc_type == ENCODE_H264 else ['hevc', 'h264']
        sps_markers = self._HEVC_SPS_MARKERS if enc_type != ENCODE_H264 else self._H264_SPS_MARKERS

        # Pass 1: 预扫描，找 SPS/PPS 所在帧，提取参数集前缀
        param_prefix = b''
        for data in encoded_data_list:
            if any(m in data for m in sps_markers):
                param_prefix = self._extract_param_prefix(data, enc_type)
                break

        # Pass 2: param_prefix + 所有帧拼成完整流，一次性解码
        full_stream = param_prefix + b''.join(encoded_data_list)

        for fmt in fmts:
            try:
                container = av.open(BytesIO(full_stream), format=fmt)
                decoded = []
                gray_frames = 0
                for frame in container.decode(video=0):
                    pil_img = frame.to_image()
                    # 检测灰噪帧（仅统计，不过滤：保留帧以维持时间戳对齐）
                    stat = ImageStat.Stat(pil_img)
                    mean_rgb = sum(stat.mean) / len(stat.mean)
                    if max(stat.stddev) < _GRAY_STDDEV_THRESHOLD and abs(mean_rgb - _GRAY_MEAN_CENTER) < _GRAY_MEAN_TOLERANCE:
                        gray_frames += 1
                    buf = BytesIO()
                    pil_img.save(buf, 'JPEG', quality=self.jpeg_quality)
                    decoded.append(buf.getvalue())
                container.close()
                
                if gray_frames:
                    with self.lock:
                        self.skipped_count += gray_frames

                # 返回三个值：cam_timestamp, jpeg_bytes, relative_timestamp
                results = []
                for i in range(min(len(frames), len(decoded))):
                    results.append((
                        orig_indices[i],           # cam_timestamp
                        decoded[i],                # jpeg_bytes
                        relative_timestamps[i]     # relative_timestamp
                    ))
                
                with self.lock:
                    self.error_count += len(frames) - len(results)
                return results
            except Exception:
                continue

        with self.lock:
            self.error_count += len(frames)
        return []

    def _decode_camera_stream_(self, camera_id: str, frames: list) -> list:
        """将同一相机的所有帧拼成完整裸流后整体解码。

        frames: [(orig_idx, encoded_data, encode_type), ...]
        返回: [(orig_idx, jpeg_bytes), ...]

        两遍处理策略：
        1. 预扫描整个帧列表，找到 SPS/PPS（可能出现在中间任意帧）
        2. 将找到的 SPS/PPS 作为前缀注入到流最前面
        3. 拼接所有帧整体解码，不跳过任何一帧

        这样即便相机使用"带外参数集"（SPS/PPS 只发一次，后续 IDR 不重复携带），
        所有 IDR 帧和 P/B 帧都能正确解码。
        """
        if not (_AV_AVAILABLE and _PIL_AVAILABLE) or not frames:
            return []

        enc_type = frames[0][2]

        fmts = ['h264', 'hevc'] if enc_type == ENCODE_H264 else ['hevc', 'h264']
        sps_markers = self._HEVC_SPS_MARKERS if enc_type != ENCODE_H264 else self._H264_SPS_MARKERS

        # Pass 1: 预扫描，找 SPS/PPS 所在帧，提取参数集前缀
        param_prefix = b''
        for _, data, _ in frames:
            if any(m in data for m in sps_markers):
                param_prefix = self._extract_param_prefix(data, enc_type)
                break

        # Pass 2: param_prefix + 所有帧拼成完整流，一次性解码
        full_stream = param_prefix + b''.join(data for _, data, _ in frames)

        for fmt in fmts:
            try:
                container = av.open(BytesIO(full_stream), format=fmt)
                decoded = []
                gray_frames = 0
                for frame in container.decode(video=0):
                    pil_img = frame.to_image()
                    # 检测灰噪帧（仅统计，不过滤：保留帧以维持时间戳对齐）
                    stat = ImageStat.Stat(pil_img)
                    mean_rgb = sum(stat.mean) / len(stat.mean)
                    if max(stat.stddev) < _GRAY_STDDEV_THRESHOLD and abs(mean_rgb - _GRAY_MEAN_CENTER) < _GRAY_MEAN_TOLERANCE:
                        gray_frames += 1
                    buf = BytesIO()
                    pil_img.save(buf, 'JPEG', quality=self.jpeg_quality)
                    decoded.append(buf.getvalue())
                container.close()
                if gray_frames:
                    with self.lock:
                        self.skipped_count += gray_frames

                # param_prefix 本身不产生输出帧，decoded[i] 对应 frames[i]
                results = [(frames[i][0], decoded[i]) for i in range(min(len(frames), len(decoded)))]
                with self.lock:
                    self.error_count += len(frames) - len(results)
                return results
            except Exception:
                continue

        with self.lock:
            self.error_count += len(frames)
        return []

    def _writer_worker(self):
        while True:
            item = self.write_queue.get()
            if item is None:
                break
            try:
                with open(item['filepath'], 'wb') as f:
                    f.write(item['data'])
                with self.lock:
                    self.extracted_count += 1
                    self.total_bytes += len(item['data'])
                    if self.extracted_count % 100 == 0:
                        print(f"  {self.extracted_count}", end='\r')
            except Exception:
                pass
            finally:
                self.write_queue.task_done()
    
    # TODO
    # 研究parse_camera_img 和 extract_stf_to_json是如何解析图片和写为json
    '''
    .stf的数据通过
        reader = StfReader(str(stf_path))
        messages = reader.get_messages()
    读取,然后遍历messages,里面存储了图片以及对应的meta
    调用parse_camera_img将messages中的每一条解析,
        for idx, msg in enumerate(messages):
            data = msg.get('data', b'')
            if len(data) >= 8 and data[:8] == b'QIMGMETA':
                parsed = parse_camera_image(data, output_dir, idx) 
        ......
            
            output_obj = {
                'frame': frame_num,
                'timestamp': ts,
                'topic': parsed['topic'],
                'schema_name': parsed['schema_name'],
                'message': parsed['message']
            }
            write_json_file(topic_subdir / filename, output_obj)
    以上就是写json的主要代码,从里面的parsed['message']可以获得camera_id, time_stamp等有用的信息

    但是解析图片应该是通过parse_camera_img实现的
        里面读取图片数据,保存为h.265,然后用ffmpeg转为.jpg图片,
        这套流程非常慢,同等水平要六七分钟,大概是当前这个代码的四五倍时间不止.
    '''
    # data_frame.pb.txt是怎么用的,里面有vehicle_pose和gnss_pose
    '''
    以下数据为必要,main_timestamp怎么来的呢,每个时间戳文件夹下面的data_frame.pb.txt
    主要包括main_timestamp, image_infos[也就是各相机下的vehicle pose], 还有main_camera_id: CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H110
    估计main_timestamp和 main_camera_id对应的camera的timestamp是一个.

        main_timestamp: 1772803616.9270809

        image_infos {
        camera_id: CAM_PBQ_FRONT_WIDE_RESET_OPTICAL_H110
        timestamp: 1772803616.9270809
        vehicle_pose {
            x: -153.5543212890625
            y: 281.94412231445312
            z: -0.85455036163330078
            yaw: 1.5195552110671997
            pitch: 0.0013527508126571774
            roll: 0.0087936501950025558
        }

    class QcraftProcessor(object):
        ...
        def convert_one(self, clip_name):
            scene_data = self._load_scene_data(clip_name)
            ...
            sensor_info_path = os.path.join(clip_save_dir, "sensor_info.json")
            with open(sensor_info_path, "w") as f:
                sensor_info = {
                    "camera_specs": {
                        cam_id: {
                            "key": cam_spec.key,
                            "name": cam_spec.name,
                            "width": cam_spec.width,
                            "height": cam_spec.height,
                        }
                        for cam_id, cam_spec in scene_data.camera_specs.items()
                    },
                    "main_lidar_name": scene_data.main_lidar_name,
                }
                json.dump(sensor_info, f, indent=4)
    
    >>>
    def _load_scene_data(self, clip_name) -> SceneData:
        frame_timestamps = self._read_frame_timestamps(clip_dir)        _read_frame_timestamp 通过data_frame_seq.json读取      
        ...
        ego_pose_data = self._read_ego_pose_data(                       _read_ego_pose_data 通过 _read_ego_pose_data_from_pbtxt-->_read_ego_pose_data 读取data_frame.pb.txt
            clip_dir, frame_timestamps, camera_specs, original_cam_name_to_cam_id
        )
    
        
    '''
    # data_frame_seq.json是怎么用的,里面主要是时间戳
    '''
    一个内部函数把这些data_frame_path 放到列表中,供加载数据集使用,所以主要是*data_frame_path*

        "data_frame_seq_items": [
            {
            "main_timestamp": 1772803602.2270839,
            "data_frame_path": "1772803602.227",
            "distance_to_previous_frame": 221.99421125213553,
            "train_context_seq_path": [],
            "neighbor_timestamps": []
            },
            {
            "main_timestamp": 1772803602.327081,
            "data_frame_path": "1772803602.327",
            "distance_to_previous_frame": 0.523062173102942,
            "train_context_seq_path": [],
            "neighbor_timestamps": []
            },
            ....
    ]
    主要是从这里获取一个加载数据集的文件夹列表,所以目前的逻辑是将图片按时间戳归文件夹的时候,自动设给你成一个包含文件夹名的这个.json
    '''
    # 点云和图片要放在一个文件夹中,也就是同一时间帧

    def extract_from_stf(self, stf_file: str, limit: int = None):
        print(f"读取: {Path(stf_file).name}")

        reader = StfReader(stf_file)
        if not reader.open():
            print(f"无法打开: {stf_file}", file=sys.stderr)
            return
        messages = reader.get_messages()
        if not messages:
            print("无消息", file=sys.stderr)
            return

        print(f"扫描 {len(messages)} 条...")

        has_decode_cap = _AV_AVAILABLE and _PIL_AVAILABLE
        if not has_decode_cap:
            print("提示: 未安装 av/Pillow，H.265/H.264 帧将被跳过（pip install av Pillow）",
                  file=sys.stderr)

        # 启动写入线程
        write_workers = []
        for _ in range(self.num_threads):
            t = threading.Thread(target=self._writer_worker, daemon=True)
            t.start()
            write_workers.append(t)

        # Phase 1: 遍历所有消息，JPEG 直接写，H.265/H.264 按相机分组收集
        camera_stats = {}
        image_count = 0
        # h265_cameras: camera_id -> [(orig_idx, encoded_data, encode_type)]
        h265_cameras = {}


        for idx, msg in enumerate(messages):
            if limit and image_count >= limit:
                break

            data = msg if isinstance(msg, bytes) else msg.get('data', b'')
            if len(data) < 8 or data[:8] != b'QIMGMETA':
                continue

            meta = self._parse_qimgmeta(data, idx)
            if not meta:
                continue
            
            relative_timestamp = msg.get('timestamp', 0)

            enc = meta['encode_type']
            cam_timestamp = meta['timestamp']
            cam = meta['camera_id']
            cam_name = CAMERA_ID_TO_NAME.get(meta['camera_id']) 

            if enc == ENCODE_JPEG:
                filename = f"{idx:08d}_{cam}.jpg"
                self.write_queue.put({
                    'filepath': self.output_dir / filename,
                    'data': meta['encoded_data'],
                })
                camera_stats[cam] = camera_stats.get(cam, 0) + 1
                image_count += 1

            elif enc in (ENCODE_H265, ENCODE_H264):
                if has_decode_cap:
                    if cam not in h265_cameras:
                        h265_cameras[cam] = []
                    # h265_cameras[cam].append((idx, meta['encoded_data'], enc))
                    h265_cameras[cam].append((cam_timestamp, meta['encoded_data'], enc, relative_timestamp))
                    camera_stats[cam] = camera_stats.get(cam, 0) + 1
                    image_count += 1
                else:
                    with self.lock:
                        self.skipped_count += 1

            else:
                with self.lock:
                    self.skipped_count += 1

        # Phase 2: 对每个相机整流解码，decode_threads 控制并行相机数
        if h265_cameras:
            total_h265 = sum(len(v) for v in h265_cameras.values())
            print(f"  解码 {len(h265_cameras)} 个相机共 {total_h265} 帧 H.265/H.264 流...")

            with ThreadPoolExecutor(max_workers=self.decode_threads) as executor:
                future_to_cam = {
                    executor.submit(self._decode_camera_stream, cam, frames): cam
                    for cam, frames in h265_cameras.items()
                }
                for future in as_completed(future_to_cam):
                    cam = future_to_cam[future]
                    try:
                        results = future.result()
                        # for orig_idx, jpeg_bytes in results:
                        for cam_timestamp, jpeg_bytes, relative_timestamp in results:
                            filename = f"{CAMERA_ID_TO_NAME.get(cam, cam)}-{cam_timestamp}.jpg"
                            if any(x in str(cam_timestamp) for x in ['1174424274.5', '1174424274.6']):
                                print("存在247.5和247.6的图片")
                            # filename = f"{orig_idx:08d}_{cam}.jpg"
                            self.write_queue.put({
                                'filepath': self.output_dir / filename,
                                'data': jpeg_bytes,
                            })

                            # 存储去重后的 timestamp 映射
                            # 只保留第一次出现的，后面的忽略（去重）
                            if relative_timestamp not in self.timestamp_mapping:
                                self.timestamp_mapping[relative_timestamp] = cam_timestamp

                    except Exception:
                        with self.lock:
                            self.error_count += len(h265_cameras[cam])

            # 所有图片处理完成后，保存 JSON
            json_file = self.output_dir / 'timestamp_mapping.json'
            with open(json_file, 'w') as f:
                json.dump(self.timestamp_mapping, f, indent=2)

        # 等待所有写入完成
        self.write_queue.join()
        for _ in range(self.num_threads):
            self.write_queue.put(None)
        for t in write_workers:
            t.join()

        print(f"\n完成: {self.extracted_count} 张, {self.total_bytes / 1024 / 1024:.1f} MB")
        print(f"目录: {self.output_dir}")

        if self.skipped_count:
            print(f"警告: {self.skipped_count} 帧疑似灰噪图（P帧缺少参考帧，内容不可恢复，已保留以维持帧序完整）")
        if self.error_count:
            print(f"解码失败: {self.error_count} 帧")

        if camera_stats:
            print("\n相机统计:")
            for cid, cnt in sorted(camera_stats.items(), key=lambda x: str(x[0])):
                print(f"  {cid}: {cnt}")


def main():
    parser = argparse.ArgumentParser(description='STF图像提取工具（支持JPEG/H.265/H.264）')
    parser.add_argument('input_file', help='输入STF文件')
    parser.add_argument('-o', '--output', default='images', help='输出目录（默认: STF文件同级目录下同名文件夹）')
    parser.add_argument('-t', '--threads', type=int, default=4, help='写入线程数（默认: 4）')
    parser.add_argument('-d', '--decode-threads', type=int, default=4,
                        help='并行解码的相机数（默认: 4）')
    parser.add_argument('-q', '--quality', type=int, default=95,
                        help='JPEG 输出质量 1-100（默认: 95）')
    parser.add_argument('-l', '--limit', type=int, help='限制提取数量')

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"文件不存在: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    if not args.input_file.endswith('.stf'):
        print(f"文件必须是.stf格式", file=sys.stderr)
        sys.exit(1)

    if args.quality < 1 or args.quality > 100:
        print(f"JPEG质量必须在1-100之间", file=sys.stderr)
        sys.exit(1)

    # 默认输出到 STF 文件同级目录下，以文件名（去掉.stf）命名
    if args.output == 'images':
        stf_path = Path(args.input_file).resolve()
        args.output = str(stf_path.parent / stf_path.stem)

    extractor = ImageExtractor(
        output_dir=args.output,
        num_threads=args.threads,
        decode_threads=args.decode_threads,
        jpeg_quality=args.quality,
    )
    extractor.extract_from_stf(args.input_file, limit=args.limit)


if __name__ == '__main__':
    main()
