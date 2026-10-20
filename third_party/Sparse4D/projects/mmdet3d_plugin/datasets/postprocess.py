import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners, rotated_rect_iou

def bev_nms(result, iou_threshold=0.1):
    boxes_corners = box3d_to_corners(result["boxes_3d"])[:, [0, 3, 4, 7]][
        ..., [0, 1]
    ]
    scores = result["scores_3d"].detach().cpu().numpy()
    if len(boxes_corners) == 0:
        return np.array([])

    order = np.argsort(scores)[::-1]
    keep_indices = []
    while order.shape[0] > 0:
        i = order[0]
        keep_indices.append(i)
        current_poly_coords = boxes_corners[i]
        ious = []
        for j_original_idx in order[1:]:
            other_poly_coords = boxes_corners[j_original_idx]
            iou = rotated_rect_iou(current_poly_coords, other_poly_coords)
            ious.append(iou)
        ious = np.array(ious)
        inds = np.where(ious < iou_threshold)[0]
        order = order[1:][inds]
    return np.array(keep_indices)

def process_bbox_results(result, classes, excluded_classes=["traffic_cone", "barrier"]):
    # 适当膨胀 bbox
    if len(result["boxes_3d"]) > 0:
        boxes_3d = result["boxes_3d"].clone()
        boxes_3d[:, 3:6] += 0.15
        result["boxes_3d"] = boxes_3d

    # 过滤无关类别
    if "labels_3d" in result:
        labels = result["labels_3d"]
        keep_mask = ~torch.isin(
            labels,
            torch.tensor(
                [classes.index(cls) for cls in excluded_classes],
                device=labels.device,
            ),
        )
        for key in result:
            if isinstance(result[key], torch.Tensor):
                result[key] = result[key][keep_mask]
            elif isinstance(result[key], list) and len(result[key]) == len(labels):
                result[key] = [
                    item for k, item in enumerate(result[key]) if keep_mask[k]
                ]

    # 过滤低分结果
    # task_threshold = 0.0
    # keep_idx = (result["scores_3d"] > task_threshold) & (result["cls_scores"] > task_threshold)
    # for key in result:
    #     result[key] = result[key][keep_idx]

    # nms
    keep_idx = bev_nms(result)
    for key in result:
        result[key] = result[key][keep_idx]

    return result


# 定义一个Track类来存储每个跟踪目标的当前状态
class Track:
    def __init__(self, track_id, box_3d, score_3d, label_3d, cls_score, frame_idx):
        self.track_id = track_id  # 内部统一的跟踪ID
        self.box_3d = box_3d  # 最新检测到的3D box
        self.score_3d = score_3d  # box评分
        self.label_3d = label_3d  # 类别标签
        self.cls_score = cls_score  # 类别评分
        self.velocity = box_3d[7:10].cpu().numpy()  # 从box中提取速度 [vx, vy, vz]
        self.position = box_3d[0:3].cpu().numpy()  # 从box中提取位置 [x, y, z]
        self.last_seen_frame = frame_idx  # 最后一次检测到的帧索引
        self.first_seen_frame = frame_idx  # 第一次检测到的帧索引
        self.times_not_detected = 0  # 连续未被检测到的帧数
        self.total_detections = 1  # 总共被检测到的次数

    def predict_position(self, current_frame_idx, fps=10):
        """
        根据上次已知速度预测当前帧的位置。
        假设帧之间时间间隔为1。
        """
        time_diff = (current_frame_idx - self.last_seen_frame) / fps
        predicted_pos = self.position + self.velocity * time_diff
        return predicted_pos

    def interpolate_trajectory(self, start_frame, end_frame, start_box, end_box):
        """
        在两个帧之间插值生成中间轨迹
        
        Args:
            start_frame: 起始帧索引
            end_frame: 结束帧索引
            start_box: 起始帧的3D box
            end_box: 结束帧的3D box
            
        Returns:
            interpolated_trajectory: 列表，包含 (frame_idx, interpolated_box) 元组
        """
        interpolated_trajectory = []
        
        if end_frame - start_frame <= 1:
            return interpolated_trajectory
        
        # 提取起始和结束的box参数
        start_params = start_box.cpu().numpy()
        end_params = end_box.cpu().numpy()
        
        # 对中间的每一帧进行线性插值
        for frame_idx in range(start_frame + 1, end_frame):
            alpha = (frame_idx - start_frame) / (end_frame - start_frame)
            interpolated_params = start_params * (1 - alpha) + end_params * alpha
            interpolated_box = torch.from_numpy(interpolated_params).to(start_box.device)
            interpolated_trajectory.append((frame_idx, interpolated_box))
        
        return interpolated_trajectory


def track_and_fix_instance_ids(
    batch_results, max_dist_threshold=7.0, fps=10, max_frames_to_keep_track=5
):
    active_tracks = []  # 存储当前活跃的Track对象
    next_track_id = 0  # 用于分配新的跟踪ID
    processed_batch_results = []  # 存储处理后的结果
    interpolated_detections = {}  # 存储需要插值的检测结果 {frame_idx: [(track_id, box_3d, score_3d, label_3d, cls_score)]}
    all_tracks = {}  # 存储所有track的信息，用于后续过滤 {track_id: Track}
    
    for frame_idx, result in enumerate(batch_results):
        frame_data = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in result.items()
        }
        current_boxes = frame_data["boxes_3d"]
        current_scores = frame_data["scores_3d"]
        current_labels = frame_data["labels_3d"]
        current_cls_scores = frame_data["cls_scores"]
        num_detections = len(current_boxes)
        new_frame_instance_ids = -np.ones(num_detections, dtype=int)
        
        # 预测位置
        predicted_track_positions = []
        for track in active_tracks:
            predicted_track_positions.append(track.predict_position(frame_idx, fps))
        num_tracks = len(active_tracks)
        
        # 二分图匹配
        cost_matrix = np.full((num_detections, num_tracks), max_dist_threshold)
        for i in range(num_detections):
            det_label = current_labels[i].item()
            det_pos = current_boxes[i][0:3].cpu().numpy()
            for j in range(num_tracks):
                track = active_tracks[j]
                if det_label == track.label_3d.item():
                    predicted_pos = predicted_track_positions[j]
                    cost = np.linalg.norm(predicted_pos - det_pos)
                    if cost < max_dist_threshold:
                        cost_matrix[i, j] = cost
        
        detection_indices, track_indices = linear_sum_assignment(cost_matrix)
        matched_detections_set = set()  # 记录已匹配的检测索引
        matched_tracks_set = set()  # 记录已匹配的跟踪目标索引

        # 处理匹配成功的
        for det_idx, track_idx in zip(detection_indices, track_indices):
            if cost_matrix[det_idx, track_idx] != max_dist_threshold:
                track: Track = active_tracks[track_idx]
                
                # 检查是否有缺失帧需要插值（车辆消失后重新出现）
                if frame_idx - track.last_seen_frame > 1:
                    # 生成插值轨迹
                    interpolated = track.interpolate_trajectory(
                        track.last_seen_frame,
                        frame_idx,
                        track.box_3d,
                        current_boxes[det_idx]
                    )
                    
                    # 将插值结果存储到对应的帧中
                    for interp_frame_idx, interp_box in interpolated:
                        if interp_frame_idx not in interpolated_detections:
                            interpolated_detections[interp_frame_idx] = []
                        interpolated_detections[interp_frame_idx].append({
                            'track_id': track.track_id,
                            'box_3d': interp_box,
                            'score_3d': track.score_3d,  # 使用上一次的score
                            'label_3d': track.label_3d,
                            'cls_score': track.cls_score,
                            'is_interpolated': True  # 标记为插值数据
                        })
                
                # 更新track信息
                track.box_3d = current_boxes[det_idx]
                track.score_3d = current_scores[det_idx]
                track.label_3d = current_labels[det_idx]
                track.cls_score = current_cls_scores[det_idx]
                track.velocity = current_boxes[det_idx][7:10].cpu().numpy()
                track.position = current_boxes[det_idx][0:3].cpu().numpy()
                track.last_seen_frame = frame_idx
                track.times_not_detected = 0
                track.total_detections += 1
                
                new_frame_instance_ids[det_idx] = track.track_id
                matched_detections_set.add(det_idx)
                matched_tracks_set.add(track_idx)
        
        # 处理未匹配的检测（新目标）
        for det_idx in range(num_detections):
            if det_idx not in matched_detections_set:
                new_track = Track(
                    track_id=next_track_id,
                    box_3d=current_boxes[det_idx],
                    score_3d=current_scores[det_idx],
                    label_3d=current_labels[det_idx],
                    cls_score=current_cls_scores[det_idx],
                    frame_idx=frame_idx,
                )
                active_tracks.append(new_track)
                all_tracks[next_track_id] = new_track
                new_frame_instance_ids[det_idx] = next_track_id
                next_track_id += 1
        
        # 处理消失的tracks
        tracks_to_remove = []
        for track_idx in range(num_tracks):
            if track_idx not in matched_tracks_set:
                track = active_tracks[track_idx]
                track.times_not_detected += 1
                if track.times_not_detected > max_frames_to_keep_track:
                    tracks_to_remove.append(track)
        
        active_tracks = [
            track for track in active_tracks if track not in tracks_to_remove
        ]

        frame_data["instance_ids"] = torch.from_numpy(
            new_frame_instance_ids
        ).long()  # 确保是long类型
        
        processed_batch_results.append(frame_data)
    
    # 将插值的检测结果添加到对应的帧中
    for frame_idx, interp_list in interpolated_detections.items():
        if frame_idx < len(processed_batch_results):
            frame_data = processed_batch_results[frame_idx]
            
            # 获取当前帧的数据
            current_boxes = frame_data["boxes_3d"]
            current_scores = frame_data["scores_3d"]
            current_labels = frame_data["labels_3d"]
            current_cls_scores = frame_data["cls_scores"]
            current_instance_ids = frame_data["instance_ids"]
            
            # 添加插值的检测
            for interp_data in interp_list:
                current_boxes = torch.cat([current_boxes, interp_data['box_3d'].unsqueeze(0)])
                current_scores = torch.cat([current_scores, interp_data['score_3d'].unsqueeze(0)])
                current_labels = torch.cat([current_labels, interp_data['label_3d'].unsqueeze(0)])
                current_cls_scores = torch.cat([current_cls_scores, interp_data['cls_score'].unsqueeze(0)])
                current_instance_ids = torch.cat([
                    current_instance_ids, 
                    torch.tensor([interp_data['track_id']], dtype=torch.long)
                ])
            
            # 更新帧数据
            frame_data["boxes_3d"] = current_boxes
            frame_data["scores_3d"] = current_scores
            frame_data["labels_3d"] = current_labels
            frame_data["cls_scores"] = current_cls_scores
            frame_data["instance_ids"] = current_instance_ids
    
    # 过滤掉只出现1帧的物体
    # 首先统计每个track_id出现的总帧数（包括插值的帧）
    track_frame_count = {}
    for frame_data in processed_batch_results:
        instance_ids = frame_data["instance_ids"]
        for track_id in instance_ids.tolist():
            if track_id != -1:
                track_frame_count[track_id] = track_frame_count.get(track_id, 0) + 1
    
    # 找出只出现1帧的track_id
    single_frame_tracks = {track_id for track_id, count in track_frame_count.items() if count == 1}
    
    # 从所有帧中删除这些只出现1帧的检测
    filtered_batch_results = []
    for frame_data in processed_batch_results:
        instance_ids = frame_data["instance_ids"]
        
        # 找出需要保留的索引（不是单帧track）
        keep_mask = torch.tensor([track_id not in single_frame_tracks for track_id in instance_ids.tolist()])
        
        if keep_mask.any():
            # 过滤所有相关数据
            filtered_frame_data = {
                "boxes_3d": frame_data["boxes_3d"][keep_mask],
                "scores_3d": frame_data["scores_3d"][keep_mask],
                "labels_3d": frame_data["labels_3d"][keep_mask],
                "cls_scores": frame_data["cls_scores"][keep_mask],
                "instance_ids": frame_data["instance_ids"][keep_mask],
            }
            # 保留其他可能存在的键
            for key in frame_data:
                if key not in filtered_frame_data:
                    filtered_frame_data[key] = frame_data[key]
        else:
            # 如果所有检测都被过滤掉，创建空的tensor
            filtered_frame_data = {
                "boxes_3d": torch.empty((0, frame_data["boxes_3d"].shape[1]), dtype=frame_data["boxes_3d"].dtype, device=frame_data["boxes_3d"].device),
                "scores_3d": torch.empty(0, dtype=frame_data["scores_3d"].dtype, device=frame_data["scores_3d"].device),
                "labels_3d": torch.empty(0, dtype=frame_data["labels_3d"].dtype, device=frame_data["labels_3d"].device),
                "cls_scores": torch.empty(0, dtype=frame_data["cls_scores"].dtype, device=frame_data["cls_scores"].device),
                "instance_ids": torch.empty(0, dtype=torch.long, device=frame_data["instance_ids"].device),
            }
            # 保留其他可能存在的键
            for key in frame_data:
                if key not in filtered_frame_data:
                    filtered_frame_data[key] = frame_data[key]
        
        filtered_batch_results.append(filtered_frame_data)
    
    return filtered_batch_results