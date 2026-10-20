#!/usr/bin/env python3
"""STF to JSON converter"""

import sys
import os
import json
import base64
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from google.protobuf import json_format

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stf_reader import StfReader
from proto_from_descriptors import get_message_class

# 仅依赖 proto_descriptors.bin，不依赖 onboard/offboard 目录
try:
    LiteMsgWrapper = get_message_class("qcraft.LiteMsgWrapper")
    EncodedImageMetadata = get_message_class("qcraft.EncodedImageMetadata")
    DataFrameImageInfo = get_message_class("qcraft.DataFrameImageInfo")
except FileNotFoundError as e:
    print(f"错误: {e}", file=sys.stderr)
    sys.exit(1)
if LiteMsgWrapper is None or EncodedImageMetadata is None:
    print("错误: proto_descriptors.bin 中缺少 qcraft.LiteMsgWrapper 或 qcraft.EncodedImageMetadata", file=sys.stderr)
    sys.exit(1)

TAG_TO_TOPIC = {
    2: 'pose_proto',
    3: 'trajectory_proto',
    5: 'system_status_proto',
    6: 'shm_message_metadata',
    7: 'imu_raw_reading_proto',
    8: 'pandar_packet_proto',
    9: 'obstacles_proto',
    10: 'gnss_raw_reading_proto',
    11: 'gnss_imu_driver_debug_proto',
    13: 'gnss_pose_proto',
    14: 'can_driver_debug_proto',
    15: 'vehicle_control_status_proto',
    16: 'vehicle_operation_status_proto',
    17: 'pose_estimator_debug_proto',
    19: 'positioning_filter_state_proto',
    20: 'depth_maps_proto',
    22: 'chassis_detail',
    23: 'chassis',
    24: 'control_command',
    25: 'pad_message',
    27: 'execution_issue_proto',
    28: 'autonomy_state_proto',
    29: 'guardian_cmd_proto',
    30: 'driver_action',
    32: 'localization_debug_proto',
    33: 'objects_proto',
    34: 'traffic_light_states_proto',
    35: 'charts_data_proto',
    37: 'objects_prediction_proto',
    40: 'segments_in_image_proto',
    41: 'points_in_image_proto',
    43: 'planner_debug_proto',
    44: 'rerouting_request_proto',
    45: 'remote_assist_to_car_proto',
    46: 'car_to_remote_assist_proto',
    47: 'create_virtual_objects_request_proto',
    48: 'lidar_host_time_diff_proto',
    49: 'localization_pose_proto',
    50: 'audio_record_proto',
    51: 'trace_proto',
    52: 'counter_proto',
    53: 'system_info_proto',
    54: 'label_frame_proto',
    56: 'fen_detections_proto',
    57: 'log_proto',
    58: 'pose_correction_proto',
    59: 'semantic_segmentation_results_proto',
    60: 'routing_result_proto',
    61: 'recorded_route_proto',
    62: 'planner_routing_request_proto',
    68: 'measurements_proto',
    80: 'hmi_content_proto',
    97: 'gnss_ephemeris_proto',
    100: 'planner_state_proto',
    101: 'range_images_proto',
    102: 'traffic_participants_proto',
    103: 'segmentation_objects_proto',
    104: 'v2x_proto',
    105: 'q_events_proto',
    107: 'rsim_stop_proto',
    108: 'route_manager_output_proto',
    109: 'v2x_raw_input_proto',
    110: 'raw_radar_objects_proto',
    113: 'joy_stick_proto',
    114: 'routing_state_proto',
    115: 'node_state_proto',
    116: 'semantic_map_modification_proto',
    117: 'track_classifier_debug_proto',
    119: 'update_run_params_proto',
    120: 'cyber_message_proto',
    121: 'emergency_brake_proto',
    122: 'update_semantic_map_proto',
    124: 'tracker_debug_proto_with_rollback',
    125: 'q_run_events_proto',
    126: 'lite_msg_simple_proto',
    127: 'multi_camera_mono3d_measurements_proto',
    129: 'depth_estimation_results_proto',
    130: 'sensor_fovs_proto',
    131: 'q_run_event_states_proto',
    132: 'q_product_setting_proto',
    133: 'lite_perf_test_msg_proto',
    134: 'lite_run_info_proto',
    135: 'multi_camera_lanes_proto',
    136: 'run_analysis_metadata',
    137: 'measurements_group_proto',
    138: 'radar_debug_proto',
    139: 'multi_camera_road_geometry_proto',
    140: 'obstacle_semantic_grid_proto',
    141: 'vision_bev_results_proto',
    142: 'driver_command_proto',
    143: 'driver_command_response_proto',
    144: 'alc_feedback_proto',
    145: 'gpsd_status_proto',
    146: 'add_modifier_proto',
    147: 'online_semantic_map_proto',
    155: 'localization_viewer_debug_proto',
    156: 'exit_timeout_stack_proto',
    157: 'lon_controller_output_proto',
    158: 'lane_support_proto',
    159: 'human_driving_control_cmd_proto',
    160: 'fusion_objects_proto',
    161: 'vision_bev_road_geometry_proto',
    162: 'lane_support_debug_proto',
    163: 'mpp_sections_proto',
    164: 's_d_route_proto',
    165: 'online_map_proto',
    166: 'human_driving_speed_config_proto',
    167: 'emergency_brake_debug_proto',
    168: 'human_driving_config_proto',
    169: 'omni_net_feature_proto',
    170: 'subscribe_command_proto',
    171: 'history_msg_request_proto',
    172: 'history_msg_response_proto',
    173: 'path_routing_result_proto',
    174: 'route_service_request_proto',
    175: 'work_state_proto',
    176: 'module_state_list_proto',
    177: 'module_startup_command_proto',
    178: 'module_startup_state_proto',
    179: 'vision_freespace_proto',
    180: 'multi_camera_det2d_measurements_proto',
    181: 'traffic_sign_and_intelligent_speed_proto',
    182: 'multi_camera_image_anomalies_proto',
    183: 'vision_result_for_hma_proto',
    184: 'online_calibration_proto',
    185: 'execution_issue_rules_proto',
    186: 'vision_bev_occupancy_proto',
    187: 'blind_spot_assist_proto',
    188: 'hma_debug_proto',
    189: 'launch_plan_modules',
    190: 'calibration_progress_data',
    191: 'calibration_response',
    192: 'blind_spot_assist_debug_proto',
    193: 'multi_camera_traffic_sign_and_intelligent_speed_proto',
    194: 'hobot_bole_raw_proto',
    195: 'positioning_debug_proto',
    196: 'online_map_debug_proto',
    197: 'rule_matcher_result_proto',
    198: 'parking_spot_finder_proto',
    199: 'parking_spot_finder_debug_proto',
    200: 'sd_route_match_requst_proto',
    201: 'sd_map_route_proto',
    202: 'single_camera_fused_road_geometry_proto',
    203: 'fault_table_proto',
    204: 'rear_collision_warning_proto',
    205: 'rear_collision_warning_debug_proto',
    206: 'parking_spots_proto',
    207: 'fused_bev_road_geometry_proto',
    208: 'fusion_parking_spots_proto',
    209: 'gac_lane_boundary_debug_proto',
    210: 'fused_bev_road_geometry_debug_proto',
    211: 'parking_freespace_proto',
    214: 'fusion_parking_freespace_proto',
    215: 'ml_planner_output_proto',
    216: 'ml_planner_debug_proto',
    217: 'q_power_manager_state_proto',
    218: 'q_state_manager_command_proto',
    219: 'uss_proto',
    220: 'lite_test_inner_col_proto',
    221: 'mcu_log_proto',
    222: 'sd_route_match_result_proto',
    223: 'map_match_result',
    224: 'radar_data_proto',
    225: 'visual_localization_state_proto',
    226: 'calibration_evaluation_proto',
    227: 'calibration_detection_frame_proto',
    228: 'post_processing_frame_proto',
    229: 'sd_localization_proto',
    230: 'multi_camera_vision_freespace_proto',
    231: 'calibration_request',
    232: 'system_check_result_proto',
    233: 'sd_route_navi_info_proto',
    234: 'front_camera_tracker_debug_proto',
    235: 'multi_camera_fused_road_geometry_proto',
    236: 'gac_aeb_debug_info_proto',
    237: 'q_trans_phase_proto',
    238: 'gac_vehicle_state_proto',
    239: 'gac_imud_proto',
    240: 'gac_calibration_proto',
    241: 'gac_diag_msg_proto',
    242: 'sd_map_proto',
    243: 'map_fusion_result_proto',
    244: 'vio_pym_buffer_v2_proto',
    245: 'perception_obstacles_proto',
    246: 'dynamic_subscribe_protos',
    247: 'dynamic_node_config_proto',
    248: 'dynamic_node_config_ack_proto',
    249: 'li_laneline_proto',
    250: 'offboard_planner_debug_proto',
    251: 'ehp_raw_messages_proto',
    252: 'turn_by_turn_proto',
    253: 'li_map_tiles_proto',
    254: 'camera_calibration_proto',
    255: 'param_service_ack_proto',
    256: 'multi_camera_intrinsics_proto',
    257: 'hd_fusion_navi_result_proto',
    258: 'hil_system_command_proto',
    259: 'sd_ehp_raw_messages_proto',
    260: 'long_cheng_image_quality_proto',
    262: 'fusion_output_proto',
    263: 'active_safety_a_to_m_proto',
    264: 'active_safety_m_to_a_proto',
    265: 'safety_to_hui_proto',
    266: 'hu_interface_info_proto',
    267: 'park_hmi_state',
    268: 'parking_spot_finder_state_proto',
    269: 'gac_fusion_obstacle_proto',
    270: 'park_hu_dynamic_context',
    271: 'hu_dynamic_context',
    272: 'city_hu_dynamic_context',
    273: 'non_odd_events_proto',
    274: 'park_hmi_command',
    275: 'ehp_speed_limit_ranges_proto',
    276: 'city_noa_info',
    277: 'ready_to_upload_file_proto',
    278: 'ap_hmi_trajectory',
    279: 'perception_speed_limit_proto',
    280: 'road_traffic_events_proto',
    281: 'lane_traffic_events_proto',
    282: 'horizon_ihbc_header_proto',
    283: 'anonymized_box_proto',
    284: 'gnss_frame_proto',
    285: 'active_safety_debug_proto',
    286: 'li_parking_freespace_proto',
    287: 'li_parking_spots_proto',
    288: 'li_parking_limiters_proto',
    289: 'map_road_reminder_proto',
    290: 'x_ap_hmi_state',
    291: 'dds_trigger_proto',
    292: 'imu_can_calibration_proto',
    294: 'pas_odometry_proto',
    295: 'horizon_light_debug_proto',
    296: 'li_mcu_to_soa_failsafe_proto',
    297: 'calibration_result_internal',
    298: 'li_parking_freespace_msg_proto',
    299: 'lc_sd_road_horizon_proto',
    300: 'rpa_state_proto',
    301: 'parking_hmi_limiters',
    302: 'runtime_uploading_proto',
    303: 'rtu_traditional_proto',
    304: 'positioning_state_proto',
    305: 'li_calibration_uds_msg_proto',
    306: 'active_safety_trackers_debug_proto',
    307: 'road_horizon_traffic_events_proto',
    308: 'object_proto',
    309: 'li_vehicle_and_setting_info',
    310: 'li_parking_planning_result_proto',
    311: 'li_sensor_can_fault_status_proto',
    312: 'parking_hmi_rear_mirror_control_proto',
    313: 'data_statistics',
    314: 'soa_ins_proto',
    315: 'li_fisheye_object_proto',
    316: 'occupancy_parking_proto',
    317: 'perception_freespace_proto',
    318: 'grid_map_proto',
    319: 'lsaeb_freespace_proto',
    320: 'suspension_proto',
    321: 'suspension_height_level_sensor_proto',
    322: 'li_suspension_mat4d_proto',
    324: 'hpp_mapping_debug_proto',
    325: 'external_objects_proto',
    326: 'hpp_mapping_request_proto',
    327: 'fusion_occ_objects_proto',
    328: 'hpp_online_map_proto',
    329: 'ad_control_debug_proto',
    330: 'hpp_parking_mode_request_proto',
    331: 'serialized_data_wrapper_proto',
    332: 'remote_drive_states_proto',
    333: 'lidar_fault_message_proto',
    334: 'integrated_navigation_debug_proto',
    335: 'as_function_debug_proto',
    336: 'planning_result_proto',
    337: 'lidar_perception_net_result_proto',
    338: 'aes_result_proto',
    339: 'active_safety_perception_proto',
    340: 'bev_static_perception_ego_trajectory_proto',
    341: 'hpp_mapping_state_proto',
    342: 'emergency_steering_proto',
    343: 'emergency_steering_debug_proto',
    344: 'lidar_tracker_debug_proto',
    345: 'multi_lidar_intrinsics_proto',
    346: 'camera_lidar_calibration_proto',
    347: 'lidar_grids_info_proto',
    348: 'lidar_segment_objects_proto',
    349: 'li_calibration_request_proto',
    350: 'li_calibration_response_proto',
    351: 'short_sd_route_info_proto',
    352: 'nav_info_ni_sdk_proto',
    353: 'ni_sdk_debug_message_proto',
    354: 'traffic_former_states_proto',
    355: 'lidar_point_denoise_result_proto',
    356: 'can_log_proto',
    357: 'sys_log_proto',
    358: 'baidu_nav_info_proto',
    359: 'cockpit_raw_input_proto',
    360: 'cockpit_control_command_proto',
    361: 'cockpit_vehicle_statuses_proto',
    362: 'cockpit_vehicle_list_proto',
    363: 'cockpit_takeover_request_proto',
    364: 'zone_hu_dynamic_context',
    365: 'zone_city_noa_info',
    366: 'cockpit_vehicle_info_proto',
    367: 'park_avp_maps',
    368: 'park_avp_maps_index',
    369: 'sd_raw_messages_proto',
    370: 'park_scene_proto',
    371: 'brightness_proto',
    372: 'cockpit_notify_proto',
    373: 'trajectory_match_result_proto',
    374: 'cockpit_vehicle_perception_proto',
    375: 'adb_objects',
    376: 'external_measurements_proto',
    377: 'ambient_light_intensity',
    378: 'cockpit_camera_statis_proto',
    379: 'cockpit_video_delay_stats_proto',
    380: 'qdem_proto',
    381: 'map_fusion_debug_proto',
    382: 'ft_fusion_loc_data',
    383: 'voxel_grids_proto',
    384: 'someip_huui_info_avm_proto',
    385: 'avm_display_view_proto',
    386: 'q_calibration_request_proto',
    387: 'q_calibration_response_proto',
    388: 'avm_display_state_input_proto',
    389: 'upload_run_info_proto',
    390: 'map_debug_proto',
    391: 'factory_mode_trigger_proto',
    392: 'minimum_risk_maneuvers_trigger_proto',
    393: 'mnoa_online_map_proto',
    394: 'map_fusion_state_proto',
    395: 'publish_command_proto',
    396: 'publish_command_ack_proto',
    397:     'uss_freespace_proto',
}

# topic 名 -> tag_number，用于 --topics 过滤
TOPIC_TO_TAG = {v: k for k, v in TAG_TO_TOPIC.items()}

# 默认写 JSON 时的缓冲大小（字节）
DEFAULT_JSON_WRITE_BUFFER = 256 * 1024


def _topic_to_subdir(topic: str) -> str:
    """将 topic 字符串转为子目录名。
    例：/qcraft/pose_proto -> pose_proto
        /qcraft/camera/1  -> camera_1
        /qcraft/binary_data -> binary_data
    """
    name = topic.strip('/')
    if name.startswith('qcraft/'):
        name = name[len('qcraft/'):]
    return name.replace('/', '_')


def parse_camera_image(data: bytes, output_dir: Path = None, frame_num: int = 0) -> dict:
    if len(data) < 2:
        return None
    try:
        if len(data) < 12 or data[:8] != b'QIMGMETA':
            return None # 验证数据头部
        
        meta_len = int.from_bytes(data[8:12], byteorder='little', signed=True) # 解析元数据长度
        if meta_len <= 0 or meta_len > len(data) - 12: 
            return None # 验证元数据长度有效性
        
        metadata = EncodedImageMetadata()  # 创建protobuf消息对象
        metadata.ParseFromString(data[12:12 + meta_len])    #从数据中提取元数据部分,使用ParseFromString反序列化二进制数据

        meta_obj = json_format.MessageToDict(
            metadata, preserving_proto_field_name=True, including_default_value_fields=True
        ) # protobuf转json字典

        image_bytes = data[12 + meta_len:] # 提取图像数据
        meta_obj['image_size'] = len(image_bytes) # 记录图像数据长度

        # 根据 encode_type 动态确定文件扩展名
        encode_type = metadata.image_info.encode_type if metadata.HasField('image_info') else -1

        if output_dir:
            # 对于 H265/H264，先保存原始文件，然后转换为 JPG
            if encode_type == 1:  # ENCODE_TYPE_H265
                # 检查是否为 P 帧（非关键帧），P 帧通常无法独立解码
                is_p_frame = metadata.image_info.is_p_frame if metadata.HasField('image_info') else False
                # P帧是视频编码中的一种帧类型，它只记录与前一帧的差异，而不是完整的图像数据
                if is_p_frame:
                    # P 帧无法独立解码，直接保存为 h265
                    h265_filename = f"frame_{frame_num:06d}_camera_{metadata.camera_id}.h265"
                    with open(output_dir / h265_filename, 'wb') as f:
                        f.write(image_bytes)
                    meta_obj['image_file'] = h265_filename
                    meta_obj['decode_warning'] = 'P-frame cannot be decoded independently'
                else:
                    # I 帧或其他关键帧，尝试转换为 JPG
                    temp_h265 = output_dir / f"temp_{frame_num:06d}_{metadata.camera_id}.h265"
                    jpg_filename = f"frame_{frame_num:06d}_camera_{metadata.camera_id}.jpg"
                    jpg_path = output_dir / jpg_filename

                    # 保存 H265 文件
                    with open(temp_h265, 'wb') as f:
                        f.write(image_bytes)

                    # 使用 ffmpeg 转换为 JPG（-f hevc 指定 raw HEVC 格式，否则 ffmpeg 无法解析裸流）
                    try:
                        subprocess.run(
                            ['ffmpeg', '-f', 'hevc', '-i', str(temp_h265), 
                             '-frames:v', '1', 
                             '-quality', '95'    # 质量95%
                             '-y', str(jpg_path)],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            check=True
                        )
                        temp_h265.unlink()  # 删除临时文件
                        meta_obj['image_file'] = jpg_filename
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        # ffmpeg 失败，保留 h265 文件
                        temp_h265.rename(output_dir / f"frame_{frame_num:06d}_camera_{metadata.camera_id}.h265")
                        meta_obj['image_file'] = f"frame_{frame_num:06d}_camera_{metadata.camera_id}.h265"
                        meta_obj['decode_warning'] = 'ffmpeg conversion failed'

            elif encode_type == 2:  # ENCODE_TYPE_H264
                temp_h264 = output_dir / f"temp_{frame_num:06d}_{metadata.camera_id}.h264"
                jpg_filename = f"frame_{frame_num:06d}_camera_{metadata.camera_id}.jpg"
                jpg_path = output_dir / jpg_filename

                # 保存 H264 文件
                with open(temp_h264, 'wb') as f:
                    f.write(image_bytes)

                # 使用 ffmpeg 转换为 JPG（-f h264 指定 raw H.264 格式）
                try:
                    subprocess.run(
                        ['ffmpeg', '-f', 'h264', '-i', str(temp_h264), '-frames:v', '1', '-y', str(jpg_path)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=True
                    )
                    temp_h264.unlink()  # 删除临时文件
                    meta_obj['image_file'] = jpg_filename
                except (subprocess.CalledProcessError, FileNotFoundError):
                    # ffmpeg 失败，保留 h264 文件
                    temp_h264.rename(output_dir / f"frame_{frame_num:06d}_camera_{metadata.camera_id}.h264")
                    meta_obj['image_file'] = f"frame_{frame_num:06d}_camera_{metadata.camera_id}.h264"

            else:  # ENCODE_TYPE_JPEG 或其他
                if encode_type == 0:
                    ext = 'jpg'
                else:
                    ext = 'bin'
                img_filename = f"frame_{frame_num:06d}_camera_{metadata.camera_id}.{ext}"
                with open(output_dir / img_filename, 'wb') as f:
                    f.write(image_bytes)
                meta_obj['image_file'] = img_filename
        else:
            meta_obj['image_data'] = base64.b64encode(image_bytes).decode('ascii')
        
        return {
            'topic': f"/qcraft/camera/{metadata.camera_id}",
            'schema_name': 'qcraft.EncodedImageMetadata',
            'message': meta_obj
        }
    except Exception:
        return None


def parse_lite_msg_wrapper(data: bytes) -> dict:
    if len(data) < 2:
        return None
    try:
        wrapper = LiteMsgWrapper()
        wrapper.ParseFromString(data)
        
        for field_desc in wrapper.DESCRIPTOR.fields:
            if field_desc.number >= 2 and wrapper.HasField(field_desc.name):
                field_value = getattr(wrapper, field_desc.name)
                message_dict = json_format.MessageToDict(
                    field_value, preserving_proto_field_name=True, including_default_value_fields=False
                )
                topic_name = TAG_TO_TOPIC.get(field_desc.number, field_desc.name)
                return {
                    'topic': f'/qcraft/{topic_name}',
                    'schema_name': f'qcraft.{field_value.DESCRIPTOR.name}',
                    'message': message_dict
                }
        return None
    except Exception:
        return None


def parse_binary_data(data: bytes, output_dir: Path = None, frame_num: int = 0) -> dict:
    if len(data) > 102400 and output_dir:
        # 按 topic 名称建子目录：binary_data
        bin_subdir = output_dir / 'binary_data'
        bin_subdir.mkdir(parents=True, exist_ok=True)
        bin_filename = f"frame_{frame_num:06d}.bin"
        with open(bin_subdir / bin_filename, 'wb') as f:
            f.write(data)
        return {
            'topic': '/qcraft/binary_data',
            'schema_name': 'raw_binary',
            'message': {
                'data_size': len(data),
                'data_file': bin_filename
            }
        }
    return None


def _parse_chunk_for_process(args: tuple) -> list:
    """多进程用：解析一段 (idx, msg) 列表，返回 [(idx, parsed 或 None, timestamp), ...]。须为模块级以便 pickle。"""
    chunk, output_dir_str, topics_filter = args
    output_dir = Path(output_dir_str)
    out = []
    for idx, msg in chunk:
        data = msg.get('data', b'')
        timestamp = msg.get('timestamp', 0)
        if len(data) >= 8 and data[:8] == b'QIMGMETA':
            parsed = parse_camera_image(data, output_dir, idx)
        elif topics_filter is not None and msg.get('tag_number') is not None and msg.get('tag_number') not in topics_filter:
            out.append((idx, None, timestamp))
            continue
        else:
            parsed = parse_lite_msg_wrapper(data)
            if parsed is None:
                parsed = parse_binary_data(data, output_dir, idx)
        out.append((idx, parsed, timestamp))
    return out


def _parse_one_message(args: tuple) -> tuple:
    """单条消息解析（供线程池调用），返回 (idx, parsed_dict 或 None, timestamp)"""
    idx, msg, output_dir, topics_filter = args
    data = msg.get('data', b'')
    timestamp = msg.get('timestamp', 0)
    if len(data) >= 8 and data[:8] == b'QIMGMETA':
        parsed = parse_camera_image(data, output_dir, idx)
    elif topics_filter is not None and msg.get('tag_number') is not None and msg.get('tag_number') not in topics_filter:
        return (idx, None, timestamp)
    else:
        parsed = parse_lite_msg_wrapper(data)
        if parsed is None:
            parsed = parse_binary_data(data, output_dir, idx)
    return (idx, parsed, timestamp)


def _build_topics_filter(topics_arg: Optional[str]) -> Optional[set]:
    """将 --topics 字符串转为允许的 tag_number 集合，None 表示不过滤。"""
    if not topics_arg or not topics_arg.strip():
        return None
    allowed = set()
    for part in topics_arg.split(','):
        part = part.strip()
        if not part:
            continue
        if part.isdigit():
            allowed.add(int(part))
        else:
            tag = TOPIC_TO_TAG.get(part)
            if tag is not None:
                allowed.add(tag)
    return allowed if allowed else None


def extract_stf_to_json(
    stf_path: str,
    output_dir: str,
    num_threads: int = 1,
    num_workers: int = 1,
    topics_filter: Optional[set] = None,
    json_indent: int = 2,
    write_buffering: int = DEFAULT_JSON_WRITE_BUFFER,
):
    stf_path = Path(stf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for old_json in output_dir.glob("frame_*.json"):
        old_json.unlink()   # 删除原json文件

    reader = StfReader(str(stf_path))
    if not reader.open():
        return

    messages = reader.get_messages()
    if not messages:
        print(f"无消息: {stf_path}", file=sys.stderr)
        return

    success_count = 0
    topic_counters = {}

    def write_json_file(filepath: Path, obj: dict) -> None:
        with open(filepath, 'w', encoding='utf-8', buffering=write_buffering) as f:
            json.dump(obj, f, indent=json_indent, ensure_ascii=False)

    if num_workers > 1:
        # 多进程解析（真正并行，避免 GIL）
        n = len(messages)
        workers = min(num_workers, n)
        chunk_size = (n + workers - 1) // workers
        chunks = []
        for i in range(workers):
            start = i * chunk_size
            end = min(start + chunk_size, n)
            if start >= end:
                break
            chunk = [(j, messages[j]) for j in range(start, end)]
            chunks.append((chunk, str(output_dir), topics_filter))
        results = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for part in executor.map(_parse_chunk_for_process, chunks):
                results.extend(part)
        results.sort(key=lambda x: x[0])
        for idx, parsed, timestamp in results:
            if parsed is None:
                continue
            subdir_name = _topic_to_subdir(parsed['topic'])
            topic_subdir = output_dir / subdir_name
            topic_subdir.mkdir(parents=True, exist_ok=True)
            if subdir_name not in topic_counters:
                topic_counters[subdir_name] = 0
            frame_num = topic_counters[subdir_name]
            filename = f"frame_{frame_num:06d}_{subdir_name}_{timestamp}.json"
            output_obj = {
                'frame': frame_num,
                'timestamp': timestamp,
                'topic': parsed['topic'],
                'schema_name': parsed['schema_name'],
                'message': parsed['message']
            }
            write_json_file(topic_subdir / filename, output_obj)
            topic_counters[subdir_name] += 1
            success_count += 1
        print(f"完成: {success_count} 个文件 -> {output_dir} (进程数: {num_workers})")
        return

    if num_threads <= 1:
        # 单线程：无线程池/排序开销；先判 QIMGMETA 再 lite_msg，减少无效 camera 解析
        for idx, msg in enumerate(messages):
            data = msg.get('data', b'')
            if len(data) >= 8 and data[:8] == b'QIMGMETA':
                parsed = parse_camera_image(data, output_dir, idx)  # 处理相机图像数据,包含元数据+图像

            elif topics_filter is not None and msg.get('tag_number') is not None and msg.get('tag_number') not in topics_filter:
                continue # 跳过不处理某些消息
            else:
                parsed = parse_lite_msg_wrapper(data)

                if parsed is None:
                    parsed = parse_binary_data(data, output_dir, idx)
            if parsed is None:
                continue

            subdir_name = _topic_to_subdir(parsed['topic'])   # topic就是camera id
            topic_subdir = output_dir / subdir_name         # camera_id的文件夹
            topic_subdir.mkdir(parents=True, exist_ok=True)
            
            if subdir_name not in topic_counters:
                topic_counters[subdir_name] = 0  # 初始化计数器,后续统计每个topic处理了多少消息

            frame_num = topic_counters[subdir_name]
            ts = msg.get('timestamp', 0)
            filename = f"frame_{frame_num:06d}_{subdir_name}_{ts}.json"
            output_obj = {
                'frame': frame_num,
                'timestamp': ts,
                'topic': parsed['topic'],
                'schema_name': parsed['schema_name'],
                'message': parsed['message']
            }
            write_json_file(topic_subdir / filename, output_obj)
            topic_counters[subdir_name] += 1
            success_count += 1
        print(f"完成: {success_count} 个文件 -> {output_dir}")
        return

    # 多线程路径
    tasks = [(i, m, output_dir, topics_filter) for i, m in enumerate(messages)]
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(_parse_one_message, tasks))
    results.sort(key=lambda x: x[0])

    for idx, parsed, timestamp in results:
        if parsed is None:
            continue
        subdir_name = _topic_to_subdir(parsed['topic'])
        topic_subdir = output_dir / subdir_name
        topic_subdir.mkdir(parents=True, exist_ok=True)
        if subdir_name not in topic_counters:
            topic_counters[subdir_name] = 0
        frame_num = topic_counters[subdir_name]
        filename = f"frame_{frame_num:06d}_{subdir_name}_{timestamp}.json"
        output_obj = {
            'frame': frame_num,
            'timestamp': timestamp,
            'topic': parsed['topic'],
            'schema_name': parsed['schema_name'],
            'message': parsed['message']
        }
        write_json_file(topic_subdir / filename, output_obj)
        topic_counters[subdir_name] += 1
        success_count += 1

    print(f"完成: {success_count} 个文件 -> {output_dir} (线程数: {num_threads})")

def main():
    parser = argparse.ArgumentParser(description='STF 转 JSON')
    parser.add_argument('input', help='STF 文件或目录')
    parser.add_argument('--output', nargs='?', default='output', help='输出目录（默认: output）')
    parser.add_argument('-t', '--threads', type=int, default=4, help='解析线程数（默认: 4）') 
    parser.add_argument('-w', '--workers', type=int, default=4, help='解析进程数（默认: 1；>1 时多进程并行，可突破 GIL）') #10
    # 2: pose_proto.json, 333: idar_fault_message_proto.json 用于给pointcloud 的timestamp对应现实时间
    parser.add_argument('--topics', type=str, default="2,33", help='仅输出指定 topic，逗号分隔，如 pose_proto,objects_proto 或 2,33')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_base = Path(args.output)
    topics_filter = _build_topics_filter(args.topics)
    # topics_filter = _build_topics_filter(str(TAG_TO_TOPIC.keys())) # 查看全部topic
    json_indent = 2  # 固定缩进，便于阅读

    if not input_path.exists():
        print(f"路径不存在: {input_path}", file=sys.stderr)
        sys.exit(1)

    if input_path.is_dir():
        stf_files = list(input_path.glob("*.stf"))
        if not stf_files:
            print(f"无.stf文件: {input_path}", file=sys.stderr)
            sys.exit(1)
        for stf_file in stf_files:
            extract_stf_to_json(
                stf_file,
                output_base / input_path.name / stf_file.stem,
                num_threads=args.threads,
                num_workers=args.workers,
                topics_filter=topics_filter,
                json_indent=json_indent,
            )
    elif input_path.is_file() and input_path.suffix == '.stf':
        extract_stf_to_json(
            input_path,
            output_base / input_path.stem,
            num_threads=args.threads,
            num_workers=args.workers,
            topics_filter=topics_filter,
            json_indent=json_indent,
        )
    else:
        print(f"输入必须是.stf文件或目录", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
