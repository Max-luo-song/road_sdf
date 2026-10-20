import os
import argparse 
import mmcv
import numpy as np
from pyquaternion import Quaternion
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmdet.datasets import build_dataset
from mmdet.models import build_detector

from projects.mmdet3d_plugin.datasets.qcraft_inference_dataset import (
    QCraftInferenceDataset,
    euler_to_rotation_matrix,
    lidar_nusc_box_to_global,
    lidar_nusc_box_to_ego,
    output_to_nusc_box 
)

from projects.mmdet3d_plugin.datasets.postprocess import (
    process_bbox_results,
    track_and_fix_instance_ids,
)

PRED_LABEL_DIR_NAME = "label_pred"

def get_annotations(results, dataset: QCraftInferenceDataset, coord="global"):
    annos_list = []
    mapped_class_names = dataset.CLASSES

    for frame_id, result in enumerate(results):
        boxes = output_to_nusc_box(result)

        if coord == "global":
            boxes = lidar_nusc_box_to_global(
                dataset.data_infos[frame_id],
                boxes,
                mapped_class_names,
                dataset.det3d_eval_configs.class_range,
            )
        elif coord == "ego":
            boxes = lidar_nusc_box_to_ego(
                dataset.data_infos[frame_id],
                boxes,
                mapped_class_names,
                dataset.det3d_eval_configs.class_range,
            )

        curr_annos = []
        for box in boxes:
            nusc_anno = dict(
                translation=box.center.tolist(),
                size=box.wlh.tolist(),
                rotation=box.orientation.elements.tolist(),
                velocity=box.velocity[:2].tolist(),
                tracking_name=mapped_class_names[box.label],
                tracking_score=box.score,
                tracking_id=int(box.token.cpu()),
            )
            curr_annos.append(nusc_anno)
        
        annos_list.append(curr_annos)
    
    return annos_list


def export_to_nuscenes_format(annos_list, dataset: QCraftInferenceDataset, label_dir):
    for frame_id, annos in enumerate(annos_list):
        nusc_submissions = {
            "meta": dataset.modality,
            "results": annos,
        }
        label_path = os.path.join(label_dir, f"{frame_id}.json")
        mmcv.dump(nusc_submissions, label_path, indent=4)


def export_to_sustech_format(annos_list, dataset: QCraftInferenceDataset, label_dir):
    timestamps = dataset.raw_scene_data.frame_timestamps
    for frame_timestamp, annos in zip(timestamps, annos_list):
        sustech_annos = []
        for obj in annos:
            obj_id = str(obj["tracking_id"])
            obj_type = dataset.NUSC_TO_SUSTECH_MAP[obj["tracking_name"]]

            box_center = obj["translation"]
            position = dict(
                x=box_center[0],
                y=box_center[1],
                z=box_center[2],
            )

            box_rotation = obj["rotation"]
            rotation = Quaternion(
                box_rotation[0],
                box_rotation[1],
                box_rotation[2],
                box_rotation[3]
            )
            yaw, pitch, roll = rotation.yaw_pitch_roll
            rotation = dict(
                x=roll,
                y=pitch,
                z=yaw,
            )
            
            box_scale = obj["size"]
            scale = dict(
                x=box_scale[1],
                y=box_scale[0],
                z=box_scale[2],
            )

            anno = dict(
                obj_id=obj_id,
                obj_type=obj_type,
                psr=dict(
                    position=position,
                    rotation=rotation,
                    scale=scale,
                )
            )
            sustech_annos.append(anno)
        
        label_path = os.path.join(label_dir, f"{frame_timestamp}.json")
        mmcv.dump(sustech_annos, label_path, indent=4)


def load_gt_labels(label_dir, lidar_to_ego):
    gt_results = []
    for label_file in sorted(os.listdir(label_dir)):
        if not label_file.endswith(".json"):
            continue

        label_path = os.path.join(label_dir, label_file)
        obj_data_list = mmcv.load(label_path)

        boxes_3d = []
        instance_ids = []
        for obj_data in obj_data_list:
            rotation = obj_data["psr"]["rotation"]
            rotation = euler_to_rotation_matrix(rotation["z"], rotation["y"], rotation["x"])

            position = obj_data["psr"]["position"]
            position = np.array([position["x"], position["y"], position["z"]])

            obj2ego = np.eye(4, dtype=np.float64)
            obj2ego[:3, :3] = rotation
            obj2ego[:3, 3] = position

            obj2lidar = np.linalg.inv(lidar_to_ego) @ obj2ego
            yaw, _, _ = Quaternion(matrix=obj2lidar[:3, :3]).yaw_pitch_roll
            x, y, z = obj2lidar[:3, 3]

            scale = obj_data["psr"]["scale"]
            l, w, h = scale["x"], scale["y"], scale["z"]

            box_3d = torch.tensor([x, y, z, l, w, h, yaw, 0.0, 0.0, 0.0], dtype=torch.float32)
            boxes_3d.append(box_3d)

            obj_id = int(obj_data["obj_id"])
            instance_ids.append(obj_id)

        boxes_3d = torch.stack(boxes_3d, dim=0) if boxes_3d else torch.empty((0, 10), dtype=torch.float32)
        instance_ids = torch.tensor(instance_ids, dtype=torch.long) if instance_ids else torch.empty(0, dtype=torch.long)

        gt_result = {
            "boxes_3d": boxes_3d,
            "instance_ids": instance_ids,
        }
        gt_results.append(gt_result)

    return gt_results

def infer(cfg, scene_id, model, dataset: QCraftInferenceDataset, output_dir):
    print(f"Infering scene: {scene_id}")

    gpu_id = 0
    model = model.cuda(gpu_id)
    model.eval()

    # 重置 Instance Bank
    model.head.instance_bank.reset()

    excluded_classes = ["traffic_cone", "barrier"]

    batch_data = []
    pred_results = []
    for frame_idx in range(len(dataset)):
        data = collate([dataset[frame_idx]], samples_per_gpu=1)
        data = scatter(data, [gpu_id])[0]
        batch_data.append(data)

        with torch.no_grad():
            results = model(return_loss=False, rescale=True, **data)

        result = results[0]["img_bbox"]

        # - boxes_3d：预测LiDAR坐标系下的3D box，tensor(N_instances, 10)，格式：[x, y, z, l, w, h, yaw, v_x, v_y, v_z]，其中box坐标为box的几何中心
        # - scores_3d: box评分，tensor(N_instances)
        # - labels_3d：预测的类别标签，tensor(N_instances)
        # - cls_scores：类别评分，tensor(N_instances)
        # - instance_ids：物体ID，tensor(N_instances)

        result = process_bbox_results(result, dataset.CLASSES, excluded_classes)
        pred_results.append(result)

        if frame_idx % 10 == 0:
            print(f"Progress: {frame_idx}/{len(dataset)} frames done.")

    # print(pred_results[2:4])
    # print("--------------------")
    pred_results = track_and_fix_instance_ids(pred_results)
    # print(pred_results[2:4])

    ego_annos_list = get_annotations(pred_results, dataset, coord="ego")

    # 保存自动化标注
    pred_label_dir = os.path.join(output_dir, PRED_LABEL_DIR_NAME)
    os.makedirs(pred_label_dir, exist_ok=True)
    print(f"Saving annotations to: {pred_label_dir}")

    export_to_sustech_format(ego_annos_list, dataset, pred_label_dir)

    # 可视化GT标签（如果存在）
    gt_labels = None
    gt_label_dir = os.path.join(output_dir, "label")
    if os.path.exists(gt_label_dir):
        print(f"GT label directory found: {gt_label_dir}. Loading GT labels for visualization.")

        data_info = dataset.data_infos[0]
        lidar_to_ego_R = Quaternion(data_info["lidar2ego_rotation"])
        lidar_to_ego_T = np.array(data_info["lidar2ego_translation"])
        lidar_to_ego = np.eye(4, dtype=np.float64)
        lidar_to_ego[:3, :3] = lidar_to_ego_R.rotation_matrix
        lidar_to_ego[:3, 3] = lidar_to_ego_T

        gt_labels = load_gt_labels(gt_label_dir, lidar_to_ego)
        if not len(gt_labels) == len(pred_results):
            print(f"GT results length {len(gt_labels)} does not match pred results length {len(pred_results)}. Skipping visualization.")
            gt_labels = None

    dataset.visualize_scene(
        batch_data,
        pred_results,
        pred_label_dir,
        cfg,
        gt_labels,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="QCraft Inference")
    parser.add_argument("--data_root", type=str, help="Path to the root directory of the dataset")
    parser.add_argument("--scene_list_path", type=str, help="Path to the scene list file")
    parser.add_argument("--scene_idx", type=str, help="Path to the scene list file")
    args = parser.parse_args()

    if args.scene_list_path is not None:
        split_file = open(args.scene_list_path, "r").readlines()[1:]
        scene_ids_list = [line.strip().split(",")[0] for line in split_file]
    else:
        scene_idx = args.scene_idx
        scene_ids_list = [scene_idx]
    
    config = "sparse4dv3_qcraft_inference"
    checkpoint = "/data/Sparse4D/ckpt/sparse4dv3_r50.pth"
    data_root = args.data_root

    cfg = Config.fromfile(f"projects/configs/{config}.py")
    cfg.data_root = data_root
    cfg.data_basic_config.data_root = data_root
    cfg.data.val.data_root = data_root

    model = build_detector(cfg.model)
    model.load_state_dict(torch.load(checkpoint)["state_dict"], strict=False)

    for scene_id in scene_ids_list:
        data_cfg = cfg.data.val.copy()
        data_cfg["scene_id"] = scene_id
        dataset = build_dataset(data_cfg)

        infer(
            cfg,
            scene_id,
            model,
            dataset,
            output_dir=os.path.join(cfg.data_root, scene_id),
        )

        del dataset
