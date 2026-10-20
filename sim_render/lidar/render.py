import os
import torch
import numpy as np
import sys
import subprocess

from plyfile import PlyData
from typing import List, Dict, Optional

cmd = "nvidia-smi -q -d Memory |grep -A4 GPU|grep Used"
result = (
    subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split("\n")
)
os.environ["CUDA_VISIBLE_DEVICES"] = str(
    np.argmin([int(x.split()[2]) for x in result[:-1]])
)

os.system("echo $CUDA_VISIBLE_DEVICES")

from scene import Scene
import time
import yaml
from gaussian_renderer import renderComposite
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from scene.gaussian_model import GaussianModel
from scene import Scene
from scene.unet import UNet

from utils.lidar_utils import pano_to_lidar_with_intensities, filter_pcd
from scene.cameras import Camera
from utils.graphics_utils import getWorld2View2
import logging
logger = logging.getLogger("render")
logger.setLevel(logging.INFO)
# avoid propagating to root logger (prevents duplicate printing)
logger.propagate = False
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

cpu_count = os.cpu_count()
torch.set_num_threads(cpu_count)

from typing import NamedTuple

ALIGN_SPIN_GRID_TO_ORIGINAL_SCAN = True
SPIN_GRID_REORDER_ROWS_TO_INTRINSIC = True
SPIN_GRID_FLIP_COLUMNS = True
SPIN_GRID_COLUMN_ROLL = 0

CHERY_ELEVATION_DEG = np.array(
    [
        6.8984375, 6.5, 6.09765625, 5.6953125, 5.27734375, 4.86328125,
        4.45703125, 4.05859375, 3.6328125, 3.234375, 2.828125, 2.41796875,
        2.21484375, 2.01953125, 1.81640625, 1.60546875, 1.42578125,
        1.21484375, 1.01953125, 0.80859375, 0.62109375, 0.41015625,
        0.21484375, 0.00390625, -0.203125, -0.390625, -0.609375,
        -0.79296875, -1.01171875, -1.19921875, -1.41015625, -1.59765625,
        -1.796875, -2, -2.1953125, -2.40625, -2.6015625, -2.80859375,
        -3.00390625, -3.2265625, -3.4609375, -3.6484375, -3.8671875,
        -4.0546875, -4.27734375, -4.45703125, -4.6875, -4.8671875,
        -5.28125, -5.69921875, -6.1015625, -6.50390625, -6.91015625,
        -7.31640625, -7.71875, -8.1171875, -8.49609375, -8.8984375,
        -9.3046875, -9.71875, -10.26171875, -11.0859375, -12.0859375,
        -13.31640625, 2.31640625, 2.1171875, 1.91796875, 1.7109375,
        1.515625, 1.3203125, 1.1171875, 0.9140625, 0.71484375, 0.515625,
        0.3125, 0.109375, -0.1015625, -0.296875, -0.5, -0.69921875,
        -0.90234375, -1.10546875, -1.3046875, -1.50390625, -1.69921875,
        -1.8984375, -2.09765625, -2.30078125, -2.50390625, -2.70703125,
        -2.90625, -3.11328125, -3.34375, -3.5546875, -3.7578125,
        -3.9609375, -4.16796875, -4.3671875, -4.57421875, -4.77734375,
    ],
    dtype=np.float32,
)

CHERY_SORTED_DESC_ROW_TO_INTRINSIC_ROW = np.argsort(CHERY_ELEVATION_DEG)[::-1]


def align_spin_grid_to_original_scan(spin_grid: np.ndarray) -> np.ndarray:
    """
    Align LiDAR-GS range-view columns to the original Spin scan order.

    LiDAR-GS uses a centered 120 deg range view where column 0 corresponds to
    +60 deg in its local x-forward/y-left convention. The original Spin scan
    observed in verification is ordered roughly 150 -> 30 deg, which is the
    opposite horizontal direction for the same forward-facing 120 deg sector.
    """
    grid = np.asarray(spin_grid, dtype=np.float32)
    if not ALIGN_SPIN_GRID_TO_ORIGINAL_SCAN:
        return grid
    if SPIN_GRID_REORDER_ROWS_TO_INTRINSIC:
        if grid.shape[0] != CHERY_SORTED_DESC_ROW_TO_INTRINSIC_ROW.shape[0]:
            raise ValueError(
                "spin_grid beam count does not match Chery intrinsic elevations: "
                f"{grid.shape[0]} vs {CHERY_SORTED_DESC_ROW_TO_INTRINSIC_ROW.shape[0]}"
            )
        reordered = np.empty_like(grid)
        reordered[CHERY_SORTED_DESC_ROW_TO_INTRINSIC_ROW] = grid
        grid = reordered
    if SPIN_GRID_FLIP_COLUMNS:
        grid = np.flip(grid, axis=1)
    if SPIN_GRID_COLUMN_ROLL:
        grid = np.roll(grid, SPIN_GRID_COLUMN_ROLL, axis=1)
    return np.ascontiguousarray(grid)


def save_spin_grid_pcd_binary(file_path: str, spin_grid: np.ndarray):
    """
    Save a LiDAR spin grid to PCD binary.

    Args:
        file_path: destination .pcd file path
        spin_grid: numpy array with shape (H, W, 2), where the last dimension is
            range and intensity.
    """
    grid = np.asarray(spin_grid, dtype=np.float32)
    if grid.ndim != 3 or grid.shape[2] != 2:
        raise ValueError("spin_grid must be a (H, W, 2) array with range,intensity channels")

    h, w, _ = grid.shape
    n = h * w
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS range intensity\n"
        "SIZE 4 4\n"
        "TYPE F F\n"
        "COUNT 1 1\n"
        f"WIDTH {w}\n"
        f"HEIGHT {h}\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    )

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.ascontiguousarray(grid).tobytes())


class GaussianView(NamedTuple):
    gaussians: GaussianModel
    scene: Scene
    time_poses: dict


class TimePose(NamedTuple):
    camera_pose: np.array
    view: Camera
    valid_mask: np.array
    gt_mask: torch.Tensor


class ValidModeInfo(NamedTuple):
    model_id: int
    model_pose: np.array
    model_view: Camera
    model_gaussians: GaussianModel
    need_train: bool
    model_world_rotation: "torch.Tensor" = None
    model_world_translation: "torch.Tensor" = None
    model_world_quaternion: "torch.Tensor" = None


class EditObjInfo(NamedTuple):
    delete_obj_ids: list
    obj_id_offset_pairs: list
    add_id_path_pairs: list
    novel_poses: list


class RenderCache(NamedTuple):
    gt_dynamic_model: object
    valid_timestamp_model: dict
    model_id_scene_info: dict
    static_views: list
    view_by_timestamp: dict
    background: torch.Tensor


def get_logger(path, log_path):
    import logging
    # use the same named logger to avoid duplicates
    logger = logging.getLogger("render")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_file = os.path.join(log_path, "outputs.log")
    # add FileHandler only if an identical file handler hasn't been added
    has_file = any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == os.path.abspath(log_file)
        for h in logger.handlers
    )
    if not has_file:
        fileinfo = logging.FileHandler(log_file)
        fileinfo.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
        fileinfo.setFormatter(formatter)
        logger.addHandler(fileinfo)

    # add StreamHandler only if none present
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        controlshow = logging.StreamHandler()
        controlshow.setLevel(logging.INFO)
        controlshow.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s"))
        logger.addHandler(controlshow)

    return logger


def save_pcd_binary(file_path: str, points: np.ndarray):
    """
    Save Nx>=4 array to PCD binary with fields x y z intensity.

    Args:
        file_path: destination .pcd file path
        points: numpy array shape (N, >=4) where columns are x,y,z,intensity,...
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 4:
        raise ValueError("points must be a (N, >=4) array with x,y,z,intensity columns")

    N = pts.shape[0]
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {N}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {N}\n"
        "DATA binary\n"
    )

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Write header (ascii) then binary floats for x,y,z,intensity
    with open(file_path, "wb") as f:
        f.write(header.encode("ascii"))
        # write contiguous little-endian float32 data for first 4 columns
        data = pts[:, :4].astype(np.float32)
        f.write(data.tobytes())


def render_set(
    gt_dynamic_model,
    dataset,
    name,
    iteration,
    valid_timestamp_model,
    model_id_scene_info,
    views,
    pipeline,
    background,
    insert_objs,
):
    render_outputs={}
    path_name = dataset.model_path.split("/")
    render_path = os.path.join(dataset.output_path, "renders")
    gt_path = os.path.join(dataset.output_path, "gt")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gt_path, exist_ok=True)

    name_list = []
    per_view_dict = {}
    t_list = []  # 存储每帧渲染时间
    total_frames = len(views)  # 总渲染帧数

    # 记录总渲染开始时间
    total_render_start = time.time()
    original_l2ws = gt_dynamic_model.l2ws

    raydrop_unet_available = False
    ckpt_path = os.path.join(dataset.model_path, "ckpt", "refine.pth")
    if not os.path.exists(ckpt_path):
        logger.error(f"UNet checkpoint 不存在: {ckpt_path}")
    else:
        raydrop_unet_available = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # UNet输入通道从3→1
        unet = UNet(in_channels=1, out_channels=1)
        state = torch.load(ckpt_path, map_location="cpu")
        unet.load_state_dict(state)
        unet.to(device)
        unet.eval()

    lidar_hfov = gt_dynamic_model.get_fov_horizontal()

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        render_timestamp = view.image_name
        world_to_camera_pose = np.eye(4)
        world_to_camera_pose[:3, :3] = np.transpose(view.R)
        world_to_camera_pose[:3, 3] = view.T
        camera_to_world_pose = np.linalg.inv(world_to_camera_pose)

        valid_model_info = []
        for model_id in valid_timestamp_model[render_timestamp]:
            time_pose = model_id_scene_info[model_id].time_poses[render_timestamp]
            model_to_camera_pose = time_pose.camera_pose
            object_view = time_pose.view

            model_gaussian = model_id_scene_info[model_id].gaussians
            model_gaussian.eval()
            model_to_world = camera_to_world_pose @ model_to_camera_pose
            (
                model_world_rotation,
                model_world_translation,
                model_world_quaternion,
            ) = build_world_transform_tensors(model_to_world)
            valid_model_info.append(
                ValidModeInfo(
                    model_id=model_id,
                    model_pose=model_to_world,
                    model_view=object_view,
                    model_gaussians=model_gaussian,
                    need_train=False,
                    model_world_rotation=model_world_rotation,
                    model_world_translation=model_world_translation,
                    model_world_quaternion=model_world_quaternion,
                )
            )

        torch.cuda.synchronize()
        t0 = time.time()
        render_pkg = renderComposite(
            view,
            background,
            pipeline,
            valid_model_info,
            dataset.max_depth,
            insert_objs=insert_objs,
            retain_grad=False,
        )
        torch.cuda.synchronize()
        t1 = time.time()
        # 输出当前进程占用的显存（单位：MB）
        mem_allocated = torch.cuda.memory_allocated() / 1024 / 1024
        mem_reserved = torch.cuda.memory_reserved() / 1024 / 1024
        logger.info(
            f"[GPU] 当前帧显存占用: allocated={mem_allocated:.2f} MB, reserved={mem_reserved:.2f} MB"
        )
        t_list.append(t1 - t0)  # 记录单帧渲染时间

        rendering = render_pkg["render"]
        render_intensity = rendering[0:1, ...]
        depth = render_pkg["depth"]

        gt = view.original_image.cuda()
        ray_drop = gt[0:1, ...]
        gt_intensity = (gt[1:2, ...] * ray_drop).detach().cpu().numpy()
        gt_depth = (gt[2:3, ...] * ray_drop).detach().cpu().numpy()
        render_raydrop = rendering[1:2, ...]

        if raydrop_unet_available:
            raydrop_refine = unet(render_raydrop)
            render_raydrop_mask = torch.where(raydrop_refine > 0.5, 1, 0)
        else:
            render_raydrop_mask = torch.where(render_raydrop > 0.5, 1, 0)

        render_intensity = render_intensity * render_raydrop_mask
        depth = depth * render_raydrop_mask

        if True:
            depth_distortion_aware = render_pkg["mid_depth_diff"]
            depth_distortion_aware = torch.where(depth_distortion_aware < 0.3, 1, 0)
            render_intensity = render_intensity * depth_distortion_aware
            depth = depth * depth_distortion_aware

        depth_numpy = depth.detach().cpu().numpy()
        intensity_numpy = render_intensity.detach().cpu().numpy()
        spin_grid = np.stack([depth_numpy[0], intensity_numpy[0]], axis=-1).astype(
            np.float32
        )
        spin_grid = align_spin_grid_to_original_scan(spin_grid)

        point_with_intensity = pano_to_lidar_with_intensities(
            depth_numpy[0, :, :],
            intensity_numpy[0],
            lidar_K=None,
            beam_inclinations=view.beam_inclinations.detach().cpu().numpy(),
            lidar_hfov=lidar_hfov,
        )
        gt_point_with_intensity = pano_to_lidar_with_intensities(
            gt_depth[0, :, :],
            gt_intensity[0],
            lidar_K=None,
            beam_inclinations=view.beam_inclinations.detach().cpu().numpy(),
            lidar_hfov=lidar_hfov,
        )

        if False:  # 密度滤波
            make_raydrop = filter_pcd(point_with_intensity[:, :3])
            point_with_intensity = point_with_intensity[make_raydrop]

        if False:  # 转到baselidar系
            sensor2baselidar = gt_dynamic_model.get_sensor2baselidar(dataset.sensorid)
            points = point_with_intensity[:, :3]
            points = (
                np.pad(points[..., :3], ((0, 0), (0, 1)), constant_values=1)
                @ sensor2baselidar.T
            )[:, :3]
            point_with_intensity[:, :3] = points
            gt_points = gt_point_with_intensity[:, :3]
            gt_points = (
                np.pad(gt_points[..., :3], ((0, 0), (0, 1)), constant_values=1)
                @ sensor2baselidar.T
            )[:, :3]
            gt_point_with_intensity[:, :3] = gt_points

        # point_with_intensity_world = point_with_intensity
        # gt_point_with_intensity_world = gt_point_with_intensity
        # point_with_intensity_world[:, :3] = (
        #     np.pad(point_with_intensity[:, :3], ((0, 0), (0, 1)), constant_values=1)
        #     @ camera_to_world_pose.T
        # )[:, :3]
        # original_l2w = original_l2ws[idx]
        # gt_point_with_intensity_world[:, :3] = (
        #     np.pad(gt_point_with_intensity[:, :3], ((0, 0), (0, 1)), constant_values=1)
        #     @ original_l2w.T
        # )[:, :3]
        # np.savetxt(
        #     os.path.join(render_path, "{}.txt".format(render_timestamp)),
        #     point_with_intensity_world,
        # )
        # np.savetxt(
        #     os.path.join(gt_path, "{}.txt".format(render_timestamp)),
        #     gt_point_with_intensity_world,
        # )
        # 取当前时间戳，保留三位小数，作为key
        render_timestamp = time.time()
        render_outputs[f"{render_timestamp:.3f}"] = {
            "rendered": point_with_intensity,
            "gt": gt_point_with_intensity,
            "spin_grid": spin_grid,
        }

    # 计算总渲染时间
    total_render_time = time.time() - total_render_start

    # 计算帧率 (FPS = 总帧数 / 总时间)
    fps = total_frames / total_render_time if total_render_time > 0 else 0

    # 打印帧率信息
    logger.info(f"\n渲染完成:")
    logger.info(f"总渲染帧数: {total_frames}")
    logger.info(f"总渲染时间: {total_render_time:.4f} 秒")
    logger.info(f"平均帧率 (FPS): {fps:.4f}")

    # 计算单帧平均渲染时间
    if t_list:
        avg_frame_time = sum(t_list) / len(t_list)
        logger.info(f"单帧平均渲染时间: {avg_frame_time:.4f} 秒")

    return render_outputs

def render_sets(
    gt_dynamic_model,
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    edit_obj_info,
    insert_objs,
    logger,
):
    render_cache = prepare_render_cache(
        gt_dynamic_model,
        dataset,
        iteration,
        edit_obj_info,
        logger,
    )
    with torch.no_grad():
        render_outputs = render_set(
            render_cache.gt_dynamic_model,
            dataset,
            "test",
            iteration,
            render_cache.valid_timestamp_model,
            render_cache.model_id_scene_info,
            render_cache.static_views,
            pipeline,
            render_cache.background,
            insert_objs,
        )
    return render_outputs


def prepare_render_cache(
    gt_dynamic_model,
    dataset: ModelParams,
    iteration: int,
    edit_obj_info,
    logger,
):
    model_id_list = [0]
    if gt_dynamic_model.get_dynamic_obj_id_list() is not None:
        model_id_list.extend(gt_dynamic_model.get_dynamic_obj_id_list())
    model_id_list = list(dict.fromkeys(model_id_list))
    edit_obj_info = remap_edit_obj_info_ids(gt_dynamic_model, edit_obj_info)
    model_id_list = delete_specified_objs(model_id_list, edit_obj_info)

    with torch.no_grad():
        static_views = None
        model_id_scene_info = {}

        for model_id in model_id_list:
            model_gaussians = GaussianModel(
                dataset.feat_dim,
                dataset.n_offsets,
                dataset.voxel_size,
                dataset.update_depth,
                dataset.update_init_factor,
                dataset.update_hierachy_factor,
                dataset.use_feat_bank,
                dataset.appearance_dim,
                dataset.ratio,
                dataset.add_opacity_dist,
                dataset.add_cov_dist,
                dataset.add_color_dist,
                dataset.color_channel,
            )
            model_scene = Scene(
                dataset,
                model_id,
                gt_dynamic_model,
                model_gaussians,
                load_iteration=iteration,
                shuffle=False,
            )
            if not model_scene.init_status:
                logger.info(f"model_id {model_id} init_status is False")
                continue

            time_poses = {}
            total_views = model_scene.getTotalCameras()
            if model_id == 0:
                static_views = total_views
            for view in total_views:
                timestamp = view.image_name
                camera_pose = np.eye(4)
                camera_pose[:3, :3] = np.transpose(view.R)
                camera_pose[:3, 3] = view.T
                time_poses[timestamp] = TimePose(
                    camera_pose=camera_pose,
                    view=view,
                    valid_mask=view.img_mask,
                    gt_mask=view.original_image,
                )
            model_gaussians = move_specified_objs(
                model_id, model_gaussians, edit_obj_info
            )
            model_id_scene_info[model_id] = GaussianView(
                gaussians=model_gaussians, scene=model_scene, time_poses=time_poses
            )
            model_gaussians.eval()
        test_views = []
        test_timestamp = []
        for idx, scene_view in enumerate(static_views):
            render_timestamp = scene_view.image_name
            test_timestamp.append(int(render_timestamp))
            test_views.append(scene_view)
        valid_timestamp_model = {}
        for view in static_views:
            timestamp = view.image_name
            valid_timestamp_model[timestamp] = []
            for model_id in model_id_scene_info.keys():
                if timestamp in model_id_scene_info[model_id].time_poses:
                    valid_timestamp_model[timestamp].append(
                        model_id
                    )  # 记录每一帧timestep（0-50）下出现的obj的id

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

    return RenderCache(
        gt_dynamic_model=gt_dynamic_model,
        valid_timestamp_model=valid_timestamp_model,
        model_id_scene_info=model_id_scene_info,
        static_views=test_views,
        view_by_timestamp={view.image_name: view for view in test_views},
        background=background,
    )


def camera_pose_from_view(view: Camera) -> np.ndarray:
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = np.transpose(view.R)
    camera_pose[:3, 3] = view.T
    return camera_pose


def set_view_translation(view: Camera, trans: np.ndarray):
    trans = np.asarray(trans, dtype=np.float32).reshape(3)
    if not hasattr(view, "_base_T"):
        view._base_T = np.asarray(view.T, dtype=np.float32).copy()
        view._base_lidar_center = view.lidar_center.detach().clone()

    view.T = view._base_T + trans
    view.world_view_transform = (
        torch.tensor(getWorld2View2(view.R, view.T, view.trans, view.scale))
        .transpose(0, 1)
        .to(view.data_device)
    )
    view.full_proj_transform = (
        view.world_view_transform.unsqueeze(0).bmm(
            view.projection_matrix.unsqueeze(0)
        )
    ).squeeze(0)
    view.camera_center = view.world_view_transform.inverse()[3, :3]
    view.lidar_center = view.camera_center.reshape(1, 3)


def render_cached_frame(
    render_cache: RenderCache,
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    frame_idx: int,
    trans: np.ndarray,
    insert_objs,
):
    timestamp = str(frame_idx)
    if timestamp not in render_cache.view_by_timestamp:
        raise KeyError(f"frame {frame_idx} not found in preloaded render cache")

    view = render_cache.view_by_timestamp[timestamp]
    set_view_translation(view, trans)
    for model_id in render_cache.valid_timestamp_model[timestamp]:
        time_pose = render_cache.model_id_scene_info[model_id].time_poses[timestamp]
        object_view = time_pose.view
        set_view_translation(object_view, trans)
        render_cache.model_id_scene_info[model_id].time_poses[timestamp] = TimePose(
            camera_pose= camera_pose_from_view(object_view),
            view=object_view,
            valid_mask=object_view.img_mask,
            gt_mask=object_view.original_image,
        )

    return render_set(
        render_cache.gt_dynamic_model,
        dataset,
        "test",
        iteration,
        render_cache.valid_timestamp_model,
        render_cache.model_id_scene_info,
        [view],
        pipeline,
        render_cache.background,
        insert_objs,
    )


def delete_specified_objs(model_id_list, edit_obj_info):
    if edit_obj_info is not None:
        for delete_id in edit_obj_info.delete_obj_ids:
            if delete_id in model_id_list:
                model_id_list.remove(delete_id)
    logger.info(f"model_id_list {model_id_list}")
    return model_id_list


def move_specified_objs(model_id, model_gaussians, edit_obj_info):
    transform_R = torch.eye(
        3,
        device=model_gaussians._anchor.device,
        dtype=model_gaussians._anchor.dtype,
    )
    if edit_obj_info is not None:
        for move_pair in edit_obj_info.obj_id_offset_pairs:
            if move_pair[0] == model_id:
                transform_T = torch.tensor(
                    move_pair[1],
                    device=model_gaussians._anchor.device,
                    dtype=model_gaussians._anchor.dtype,
                )
                new_anchor = (
                    model_gaussians._anchor @ transform_R.T
                ) + transform_T.reshape(1, 3)
                model_gaussians._anchor.data.copy_(new_anchor)

    return model_gaussians


def remap_instance_id_to_model_id(gt_dynamic_model, instance_id):
    if hasattr(gt_dynamic_model, "instance_id_to_model_id"):
        return gt_dynamic_model.instance_id_to_model_id(instance_id)
    return int(instance_id)


def remap_edit_obj_info_ids(gt_dynamic_model, edit_obj_info):
    if edit_obj_info is None:
        return edit_obj_info
    return EditObjInfo(
        delete_obj_ids=[remap_instance_id_to_model_id(gt_dynamic_model, x) for x in edit_obj_info.delete_obj_ids],
        obj_id_offset_pairs=[
            [remap_instance_id_to_model_id(gt_dynamic_model, pair[0]), pair[1]]
            for pair in edit_obj_info.obj_id_offset_pairs
        ],
        add_id_path_pairs=edit_obj_info.add_id_path_pairs,
        novel_poses=edit_obj_info.novel_poses,
    )


def build_world_transform_tensors(model_to_world: np.ndarray):
    from scipy.spatial.transform import Rotation
    model_world_rotation = torch.tensor(
        model_to_world[:3, :3], dtype=torch.float32, device="cuda"
    )
    model_world_translation = torch.tensor(
        model_to_world[:3, 3], dtype=torch.float32, device="cuda"
    )
    model_world_quaternion_np = Rotation.from_matrix(model_to_world[:3, :3]).as_quat()
    model_world_quaternion = torch.tensor(
        [
            model_world_quaternion_np[3],
            model_world_quaternion_np[0],
            model_world_quaternion_np[1],
            model_world_quaternion_np[2],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    return model_world_rotation, model_world_translation, model_world_quaternion


def read_single_ply_to_obj(
    ply_file_path: str, obj_class: str = None, sim_pose: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """
    读取单个 PLY 文件，转换为单个 obj 字典
    """
    try:
        ply_data = PlyData.read(ply_file_path)
        vertices = ply_data["vertex"]
        num_vertices = len(vertices)
    except Exception as e:
        return None

    # 1. 位置 (x, y, z)
    xyz = torch.tensor(
        np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1),
        dtype=torch.float32,
        device="cuda",
    )

    # 2. 不透明度 (opacity)
    opacity = torch.tensor(
        vertices["opacity"], dtype=torch.float32, device="cuda"
    ).unsqueeze(
        1
    )  # (N,) -> (N, 1)
    opacity = torch.clamp(opacity, 0.0, 1.0)
    # 不透明度全设为1
    opacity[:] = 1.0

    # 3. 缩放
    scaling = torch.tensor(
        np.stack(
            [vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]], axis=1
        ),
        dtype=torch.float32,
        device="cuda",
    )
    scaling = torch.exp(scaling)

    # 4. 旋转四元数 (rot_0, rot_1, rot_2, rot_3)
    rot = torch.tensor(
        np.stack(
            [
                vertices["rot_0"],
                vertices["rot_1"],
                vertices["rot_2"],
                vertices["rot_3"],
            ],
            axis=1,
        ),
        dtype=torch.float32,
        device="cuda",
    )

    # 5. LiDAR color: [intensity, raydrop]
    color = torch.ones((num_vertices, 2), dtype=torch.float32, device="cuda")
    color[:, 0] = 1.0
    color[:, 1] = 1.0  # raydrop

    # 6. sim_pose: 确保是 (4,4) tensor 且在 device 上
    if sim_pose is not None:
        if isinstance(sim_pose, np.ndarray) or isinstance(sim_pose, list):
            sim_pose = torch.tensor(
                sim_pose, dtype=torch.float32, device="cuda"
            ).reshape(4, 4)
        else:
            sim_pose = sim_pose.to(device="cuda", dtype=torch.float32).reshape(4, 4)

    if sim_pose is not None:
        sim_pose = torch.tensor(sim_pose, dtype=torch.float32).reshape(4, 4)

    # 构造单个 obj 字典
    single_obj = {
        "obj_class": obj_class,
        "xyz": xyz,
        "color": color,
        "opacity": opacity,
        "scaling": scaling,
        "rot": rot,
        "sim_pose": sim_pose,
    }

    return single_obj


def batch_read_plys_to_insert_objs(
    edit_obj_info: EditObjInfo,
) -> List[Dict[str, torch.Tensor]]:
    """
    批量读取多个 PLY 文件，生成 insert_objs 列表（一个文件对应一个 obj）

    Args:
        edit_obj_info: 包含 add_id_path_pairs 的类实例（存储 (obj_class, 文件路径) 对）

    Returns:
        insert_objs: 列表，每个元素是一个 obj 字典
    """
    insert_objs = []
    for obj_class, sim_pose, obj_pcd_filepath in edit_obj_info.add_id_path_pairs:
        single_obj = read_single_ply_to_obj(obj_pcd_filepath, obj_class, sim_pose)
        if single_obj is not None:
            insert_objs.append(single_obj)
    return insert_objs

def parse_and_apply_edit_yaml(args, yaml_path: str):
    """
    Parse edit_config.yaml and map its fields to args.delete / args.move / args.add.
    Ignores novel_poses in the yaml. Ensures args.novel_poses exists (empty if absent).
    """
    if yaml_path is None:
        if not hasattr(args, "novel_poses"):
            args.novel_poses = []
        return

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"edit yaml not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    nodes = cfg.get("Nodes", {}) or {}
    rigid = nodes.get("RigidNodes", {}) or {}

    # remove -> args.remove: {"obj_ids": [...]}
    remove_section = rigid.get("remove", {}) or {}
    remove_ids = remove_section.get("instance_id", []) or []
    setattr(args, "remove", {"obj_ids": remove_ids})

    # trajectory -> args.move: {"obj_id_offset_pairs": [[id, [x,y,z]], ...]}
    traj = rigid.get("trajectory", {}) or {}
    ids = traj.get("instance_id", []) or []
    offsets = traj.get("offset", []) or []
    # normalize offsets into list-of-lists
    if offsets and isinstance(offsets[0], (int, float)):
        offsets = [offsets]
    if offsets and len(offsets) == 1 and len(ids) > 1:
        offsets = offsets * len(ids)
    move_pairs = []
    for i, iid in enumerate(ids):
        off = offsets[i] if i < len(offsets) else [0.0, 0.0, 0.0]
        move_pairs.append([iid, off])
    setattr(args, "move", {"obj_id_offset_pairs": move_pairs})

    # add -> args.add: {"obj_id_path_pairs": [(obj_class, sim_pose(4x4 list)|None, path), ...]}
    add_section = rigid.get("add", {}) or {}
    ref_ids = add_section.get("ref_id", []) or []
    add_objs = add_section.get("add_obj", []) or []
    add_offsets = add_section.get("offset", []) or []
    maxlen = max(len(add_objs), len(ref_ids), len(add_offsets))
    add_pairs = []
    for i in range(maxlen):
        ref = ref_ids[i] if i < len(ref_ids) else None
        path = add_objs[i] if i < len(add_objs) else None
        off = add_offsets[i] if i < len(add_offsets) else None
        sim_pose = None
        if isinstance(off, list) and len(off) == 3:
            sim_pose = np.eye(4).tolist()
            sim_pose[0][3] = float(off[0])
            sim_pose[1][3] = float(off[1])
            sim_pose[2][3] = float(off[2])
        add_pairs.append((ref, sim_pose, path))
    setattr(args, "add", {"obj_id_path_pairs": add_pairs})


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--test_frames", nargs="+", type=int, default=[])
    parser.add_argument("--edit_yaml", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="chery")
    parser.add_argument("--block_size", type=int, default=50)
    args = parser.parse_args(sys.argv[1:])
    out_dir = getattr(args, "output_path", None) or "."
    os.makedirs(out_dir, exist_ok=True)
    logger = get_logger(out_dir, args.model_path)
    # 解析 YAML 编辑配置并映射到 args（如果提供 --edit_yaml）
    parse_and_apply_edit_yaml(args, getattr(args, "edit_yaml", None))
    if len(args.test_frames) == 0 and not hasattr(args, "novel_poses"):
        bin_files = [f for f in os.listdir(os.path.join(args.source_path, "lidar", "bin")) if f.endswith(".bin")]
        bin_files = sorted(bin_files, key=lambda x: int(x.split(".")[0]))
        frame_num = len(bin_files)
        args.test_frames = [x for x in range(0, frame_num, 1)]
    logger.info("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    model_args = model.extract(args)

    if args.dataset == "chery":
        from scene.chery_dataloader import Chery_Dataloader as GT_Dataloader
    elif args.dataset == "zdrive":
        from scene.zdrive_dataloader import ZDrive_Dataloader as GT_Dataloader
    else:
        logger.info("\nUnsupported data format.")
        sys.exit(1)

    train_frame_times = []
    if args.test_frames is not None:
        train_frame_times = args.test_frames
    else:
        for item in args.novel_poses:
            train_frame_times.append(item["frame_id"])
    block_info = {}
    for frame_id in train_frame_times:
        block_id = frame_id // args.block_size
        if block_id not in block_info:
            block_info[block_id] = []
        block_info[block_id].append(frame_id)

    edit_obj_info = EditObjInfo(
        delete_obj_ids=args.delete["obj_ids"] if hasattr(args, "delete") else [],
        obj_id_offset_pairs=(
            args.move["obj_id_offset_pairs"] if hasattr(args, "move") else []
        ),
        add_id_path_pairs=args.add["obj_id_path_pairs"] if hasattr(args, "add") else [],
        novel_poses=args.novel_poses if hasattr(args, "novel_poses") else [],
    )
    objs = batch_read_plys_to_insert_objs(edit_obj_info)

    output_paths = []
    lidar_id = 0  # 默认使用 lidar_id 0
    for block_id, train_frame_times in block_info.items():
        model_args.block_id = block_id  # update block id
        gt_dynamic_model = GT_Dataloader(
            model_args, train=False, train_frame_times=train_frame_times
        )
        if hasattr(args, "novel_poses"):
            gt_dynamic_model.set_novel_poses_setting(args.novel_poses)

        render_outputs = render_sets(
            gt_dynamic_model,
            model_args,
            args.iteration,
            pipeline.extract(args),
            edit_obj_info,
            objs,
            logger,
        )
        for key, pcd in render_outputs.items():
                timestamp = float(key)
                print (f"[Lidar Renderer] Saving rendered point cloud for timestamp {timestamp}...")
                render_pcd = pcd["rendered"]
                render_output_filename = os.path.join(out_dir, f"ldr{lidar_id}_{timestamp}.pcd")
                save_pcd_binary(
                    render_output_filename,
                    render_pcd,
                )
                output_paths.append(render_output_filename)
        print(f"[Lidar Renderer] Rendering completed. Outputs saved to {out_dir}")
    logger.info(f"All rendered point clouds saved to {out_dir}")
    sys.exit(0)
