import torch
import numpy as np

import os
import sys

from third_party.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from third_party.chamfer3D.fscore import fscore
import os
import math
import yaml
from scipy.spatial.transform import Rotation


def filter_pcd(pcd, r=0.35, d=3):
    from scipy.spatial import KDTree

    radius = r
    tree = KDTree(pcd)
    density = np.array([len(tree.query_ball_point(point, radius)) for point in pcd])
    condition = density >= d
    marked = np.zeros_like(density, dtype=bool)
    marked[condition] = True
    # print(marked.shape)
    return marked


def cal_beam_inclinations():
    """
    根据gt2上使用的helios 5515的硬件规则生成 fov
    """
    beam_inclinations = []
    level1 = np.linspace(-55, -10, num=15, endpoint=False)
    beam_inclinations.extend(list(level1))
    level2 = np.linspace(-10, -8, num=1, endpoint=False)
    beam_inclinations.extend(list(level2))
    level3 = np.linspace(-8, 4, num=9, endpoint=False)
    beam_inclinations.extend(list(level3))
    level4 = np.linspace(4, 7, num=2, endpoint=False)
    beam_inclinations.extend(list(level4))
    level5 = np.linspace(7, 15, num=5)
    beam_inclinations.extend(list(level5))

    result = []
    for x in beam_inclinations:
        result.append(math.radians(x))  # (x*math.pi/180.)
    # print(result)
    return np.array(result)


def find_closest_label(beam_labels, angle):
    from bisect import bisect_left

    if angle >= beam_labels[-1]:
        return len(beam_labels) - 1
    elif angle <= beam_labels[0]:
        # return beam_labels[0]
        return 0
    pos = bisect_left(beam_labels, angle)
    before = beam_labels[pos - 1]
    after = beam_labels[pos]
    if after - angle < angle - before:
        # return after
        return pos
    else:
        # return before
        return pos - 1


def find_closest_labels(beam_labels, angles):
    beam_labels = np.asarray(beam_labels)
    angles = np.asarray(angles)
    raw_pos = np.searchsorted(beam_labels, angles, side="left")
    pos = np.clip(raw_pos, 0, len(beam_labels) - 1)

    left_pos = np.clip(pos - 1, 0, len(beam_labels) - 1)
    right_pos = pos
    left_dist = np.abs(angles - beam_labels[left_pos])
    right_dist = np.abs(beam_labels[right_pos] - angles)
    choose_left = (raw_pos > 0) & (
        (raw_pos == len(beam_labels)) | (left_dist <= right_dist)
    )
    return np.where(choose_left, left_pos, right_pos)


def lidar_to_pano_with_intensities(
    local_points_with_intensities: np.ndarray,
    lidar_H: int,
    lidar_W: int,
    lidar_K=None,
    cam_pos=None,
    beam_inclinations=None,
    max_depth=80,
    lidar_hfov=2 * np.pi / 3,
    ground=None,
    is_correction=False,
    sensor_id=None,
    pre_labels=None,
    s2b=None,
):

    local_points = local_points_with_intensities[:, :3]
    local_point_intensities = local_points_with_intensities[:, 3]
    if beam_inclinations is not None:
        use_beam_inclinations = True
    else:
        use_beam_inclinations = False
        fov_up, fov = lidar_K
        fov_down = fov - fov_up

    # Compute dists to lidar center.
    if cam_pos is not None:
        dists = np.linalg.norm(local_points - cam_pos, axis=1)
    else:
        dists = np.linalg.norm(local_points, axis=1)

    if ground is None:
        ground = np.zeros(local_point_intensities.shape, dtype=bool)
    if pre_labels is None:
        pre_labels = np.ones(local_point_intensities.shape) * 100

    # print("[ debug ] ground_correction.shape and type ",ground.shape,type(ground))
    # Fill pano and intensities.
    pano = np.zeros((lidar_H, lidar_W), dtype=np.float32)
    intensities = np.zeros((lidar_H, lidar_W), dtype=np.float32)
    mask = np.zeros((lidar_H, lidar_W), dtype=np.float32)

    # The training loaders do not use correction mode. Keep that common path fully
    # vectorized to avoid the per-point Python loop dominating startup time.
    if not is_correction:
        valid = dists < max_depth
        if not np.any(valid):
            return pano, intensities, mask

        valid_points = local_points[valid]
        valid_dists = dists[valid]
        valid_intensities = local_point_intensities[valid]
        valid_labels = pre_labels[valid]

        x = valid_points[:, 0]
        y = valid_points[:, 1]
        z = valid_points[:, 2]
        beta = lidar_hfov / 2 - np.arctan2(y, x)
        c = np.rint(beta / (lidar_hfov / lidar_W)).astype(np.int32)

        if use_beam_inclinations:
            alpha = np.arctan2(z, np.sqrt(x**2 + y**2 + z**2))
            r = lidar_H - find_closest_labels(beam_inclinations, alpha) - 1
        else:
            alpha = np.arctan2(z, np.sqrt(x**2 + y**2)) + fov_down / 180 * np.pi
            r = np.rint(lidar_H - alpha / (fov / 180 * np.pi / lidar_H)).astype(
                np.int32
            )

        in_bounds = (
            (r >= 0) & (r < lidar_H) & (c >= 0) & (c < lidar_W)
        )
        if not np.any(in_bounds):
            return pano, intensities, mask

        r = r[in_bounds]
        c = c[in_bounds]
        valid_dists = valid_dists[in_bounds]
        valid_intensities = valid_intensities[in_bounds]
        valid_labels = valid_labels[in_bounds]

        flat_indices = r * lidar_W + c
        order = np.lexsort((valid_dists, flat_indices))
        flat_sorted = flat_indices[order]
        keep = np.ones(flat_sorted.shape[0], dtype=bool)
        keep[1:] = flat_sorted[1:] != flat_sorted[:-1]
        chosen = order[keep]

        flat_indices = flat_indices[chosen]
        pano.reshape(-1)[flat_indices] = valid_dists[chosen].astype(np.float32)
        intensities.reshape(-1)[flat_indices] = valid_intensities[chosen].astype(
            np.float32
        )
        mask.reshape(-1)[flat_indices] = (valid_labels[chosen] > 8).astype(np.float32)
        return pano, intensities, mask

    for local_point, dist, local_point_intensity, is_ground, pre_label in zip(
        local_points, dists, local_point_intensities, ground, pre_labels
    ):
        # Check max depth.
        if dist >= max_depth:
            continue

        x, y, z = local_point
        beta = lidar_hfov / 2 - np.arctan2(y, x)
        c = int(round(beta / (lidar_hfov / lidar_W)))

        if use_beam_inclinations:
            alpha = np.arctan2(z, np.sqrt(x**2 + y**2 + z**2))
            r = find_closest_label(beam_inclinations, alpha)
            if is_correction:  # 只对地面做个简单的矫正 其他的扫描可能得从运动本身去估计
                if is_ground:
                    if (
                        sensor_id == 0 and alpha < 0
                    ):  # 主雷达（前）是平着装的 不需要额外考虑安装角度
                        dist = np.abs(
                            dist
                            * np.sin(alpha)
                            / (np.abs(np.sin(beam_inclinations[r])) + 1e-16)
                        )
                        if dist >= max_depth:
                            continue
                    else:
                        if s2b is not None:
                            tmp = local_point @ (s2b.T)[:3, :3]  # 旋转到baselidar系
                            new_alpha = np.arctan2(
                                tmp[2], np.sqrt(tmp[0] ** 2 + tmp[1] ** 2)
                            )
                            delta_alpha = new_alpha - alpha
                            if new_alpha < 0:
                                dist = np.abs(
                                    dist
                                    * np.sin(alpha + delta_alpha)
                                    / (
                                        np.abs(
                                            np.sin(beam_inclinations[r] + delta_alpha)
                                        )
                                        + 1e-16
                                    )
                                )
                                if dist >= max_depth:
                                    continue

            r = lidar_H - r - 1
        else:
            alpha = np.arctan2(z, np.sqrt(x**2 + y**2)) + fov_down / 180 * np.pi
            r = int(round(lidar_H - alpha / (fov / 180 * np.pi / lidar_H)))

        # Check out-of-bounds.
        if r >= lidar_H or r < 0 or c >= lidar_W or c < 0:
            continue

        # Set to min dist if not set.
        if pano[r, c] == 0.0:
            pano[r, c] = dist
            intensities[r, c] = local_point_intensity
            mask[r, c] = 1 if pre_label > 8 else 0
        elif pano[r, c] > dist:
            pano[r, c] = dist
            intensities[r, c] = local_point_intensity
            mask[r, c] = 1 if pre_label > 8 else 0

    return pano, intensities, mask


def lidar_to_pano_with_grad(
    pcd: torch.tensor,
    grad: torch.tensor,
    lidar_H: int,
    lidar_W: int,
    lidar_K=None,
    cam_pos=None,
    beam_inclination=None,
    max_depth=80,
):

    local_points = (pcd[:, :3] - cam_pos).detach().cpu().numpy()
    local_point_gradx = grad[:, 0].detach().cpu().numpy()  # W
    local_point_grady = grad[:, 1].detach().cpu().numpy()  # H
    beam_inclinations = beam_inclination.detach().cpu().numpy()
    if beam_inclinations is not None:
        use_beam_inclinations = True
    else:
        use_beam_inclinations = False
        fov_up, fov = lidar_K
        fov_down = fov - fov_up

    # Fill pano and intensities.
    pano = np.zeros((lidar_H, lidar_W))
    intensities = np.zeros((lidar_H, lidar_W))
    for local_points, gradx, grady in zip(
        local_points,
        local_point_gradx,
        local_point_grady,
    ):
        # Check max depth.

        x, y, z = local_points
        beta = np.pi - np.arctan2(y, x)
        c = int(round(beta / (2 * np.pi / lidar_W)))

        if use_beam_inclinations:
            alpha = np.arctan2(z, np.sqrt(x**2 + y**2))
            r = find_closest_label(beam_inclinations, alpha)
            r = lidar_H - r - 1
        else:
            alpha = np.arctan2(z, np.sqrt(x**2 + y**2)) + fov_down / 180 * np.pi
            r = int(round(lidar_H - alpha / (fov / 180 * np.pi / lidar_H)))

        # Check out-of-bounds.
        if r >= lidar_H or r < 0 or c >= lidar_W or c < 0:
            continue

        # Set to min dist if not set.
        if pano[r, c] == 0.0:
            pano[r, c] = gradx
            intensities[r, c] = grady
        else:
            pano[r, c] = max(gradx, pano[r, c])
            intensities[r, c] = max(grady, intensities[r, c])

    return pano, intensities


def pano_to_lidar_with_intensities(
    pano: np.ndarray, intensities, lidar_K=None, beam_inclinations=None, lidar_hfov=2 * np.pi / 3
):
    """
    Args:
        pano: (H, W), float32.
        intensities: (H, W), float32.
        lidar_K: lidar intrinsics (fov_up, fov)
        beam_inclinations: beam_inclinations (H,)
        lidar_hfov: lidar horizontal fov

    Return:
        local_points_with_intensities: (N, 4), float32, in lidar frame.
    """

    H, W = pano.shape
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )
    beta = -(i - W / 2.0) / W * lidar_hfov
    if beam_inclinations is not None:
        alpha = np.expand_dims(beam_inclinations[::-1], 1).repeat(W, 1)
    else:
        fov_up, fov = lidar_K
        alpha = (fov_up - j / H * fov) / 180.0 * np.pi
    dirs = np.stack(
        [
            np.cos(alpha) * np.cos(beta),
            np.cos(alpha) * np.sin(beta),
            np.sin(alpha),
        ],
        -1,
    )
    local_points = dirs * pano.reshape(H, W, 1)

    # local_points: (H, W, 3)
    # intensities : (H, W)
    # local_points_with_intensities: (H, W, 4)
    local_points_with_intensities = np.concatenate(
        [local_points, intensities.reshape(H, W, 1)], axis=2
    )

    # Filter empty points.
    idx = np.where(pano != 0.0)

    local_points_with_intensities = local_points_with_intensities[idx]
    # print("pano shape: ",local_points_with_intensities.shape)
    return local_points_with_intensities


def pano_to_lidar(pano, lidar_K=None, beam_inclinations=None):
    """
    Args:
        pano: (H, W), float32.
        lidar_K: lidar intrinsics (fov_up, fov)

    Return:
        local_points: (N, 3), float32, in lidar frame.
    """
    local_points_with_intensities = pano_to_lidar_with_intensities(
        pano=pano,
        intensities=np.zeros_like(pano),
        lidar_K=lidar_K,
        beam_inclinations=beam_inclinations,
    )
    return local_points_with_intensities[:, :3]


def pano_to_lidar_torch(pano, lidar_K=None, beam_inclinations=None, lidar_hfov=2 * np.pi / 3, device=None):
    """
    Torch implementation of pano_to_lidar returning (N,3) tensor on specified device.
    Accepts pano as torch.Tensor or numpy array.
    """
    import torch

    if not torch.is_tensor(pano):
        pano = torch.from_numpy(np.array(pano)).float()

    if device is None:
        device = pano.device if pano.is_cuda else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    pano = pano.to(device=device)
    H, W = pano.shape
    i = torch.arange(0, W, dtype=torch.float32, device=device).view(1, W).expand(H, W)
    j = torch.arange(0, H, dtype=torch.float32, device=device).view(H, 1).expand(H, W)
    beta = -(i - W / 2.0) / W * lidar_hfov

    if beam_inclinations is not None:
        if not torch.is_tensor(beam_inclinations):
            beam = torch.from_numpy(np.array(beam_inclinations)).to(device)
        else:
            beam = beam_inclinations.to(device)
        alpha = beam.flip(0).unsqueeze(1).expand(H, W)
    else:
        if lidar_K is None:
            raise ValueError("lidar_K must be provided when beam_inclinations is None")
        fov_up, fov = lidar_K
        alpha = (fov_up - j / H * fov) / 180.0 * math.pi

    dirs_x = torch.cos(alpha) * torch.cos(beta)
    dirs_y = torch.cos(alpha) * torch.sin(beta)
    dirs_z = torch.sin(alpha)
    dirs = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1)
    local_points = dirs * pano.unsqueeze(-1)
    mask = pano != 0
    if mask.sum() == 0:
        return torch.zeros((0, 3), device=device, dtype=torch.float32)
    local_points_nonzero = local_points[mask]
    return local_points_nonzero


def pano_to_lidar_with_intensities_torch(
    pano,
    intensities=None,
    lidar_K=None,
    beam_inclinations=None,
    lidar_hfov=2 * np.pi / 3,
    device=None,
    beam_inclinations_gpu=None,
):
    """
    GPU-resident pano→LiDAR projection with intensity channel.

    Keeps all intermediate tensors on GPU; only the final (N, 4) result
    [x, y, z, intensity] needs to be transferred to CPU.  This eliminates
    the per-frame CPU pano2lidar bottleneck that appears in batch-render mode.

    Args:
        pano: (H, W) depth panorama — torch.Tensor (GPU) or numpy array.
        intensities: (H, W) intensity map — same type as pano, or None → zeros.
        lidar_K: (fov_up, fov) tuple, used when beam_inclinations* are both None.
        beam_inclinations: (H,) numpy/tensor beam elevation angles (radians).
        lidar_hfov: horizontal field of view in radians.
        device: target torch device; inferred from pano if not given.
        beam_inclinations_gpu: pre-cached GPU tensor of beam_inclinations (skips
            per-call host→device copy when provided).

    Returns:
        torch.Tensor of shape (N, 4) on ``device``, dtype float32.
    """
    import torch
    if not torch.is_tensor(pano):
        pano = torch.from_numpy(np.asarray(pano, dtype=np.float32))
    if device is None:
        device = pano.device if pano.is_cuda else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    pano = pano.to(device=device, dtype=torch.float32)
    if intensities is None:
        intensities = torch.zeros_like(pano)
    elif not torch.is_tensor(intensities):
        intensities = torch.from_numpy(np.asarray(intensities, dtype=np.float32))
    intensities = intensities.to(device=device, dtype=torch.float32)

    H, W = pano.shape
    col_idx = torch.arange(W, dtype=torch.float32, device=device).view(1, W).expand(H, W)
    beta = -(col_idx - W / 2.0) / W * lidar_hfov

    if beam_inclinations_gpu is not None:
        alpha = beam_inclinations_gpu.to(device).flip(0).unsqueeze(1).expand(H, W)
    elif beam_inclinations is not None:
        if not torch.is_tensor(beam_inclinations):
            beam = torch.from_numpy(np.asarray(beam_inclinations, dtype=np.float32)).to(device)
        else:
            beam = beam_inclinations.to(device=device, dtype=torch.float32)
        alpha = beam.flip(0).unsqueeze(1).expand(H, W)
    else:
        if lidar_K is None:
            raise ValueError("lidar_K must be provided when beam_inclinations* are both None")
        row_idx = torch.arange(H, dtype=torch.float32, device=device).view(H, 1).expand(H, W)
        fov_up, fov = lidar_K
        import math
        alpha = (fov_up - row_idx / H * fov) / 180.0 * math.pi

    dirs_x = torch.cos(alpha) * torch.cos(beta)
    dirs_y = torch.cos(alpha) * torch.sin(beta)
    dirs_z = torch.sin(alpha)
    local_xyz = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1) * pano.unsqueeze(-1)

    mask = pano != 0.0
    if mask.sum() == 0:
        return torch.zeros((0, 4), device=device, dtype=torch.float32)
    pts = local_xyz[mask]
    inten = intensities[mask].unsqueeze(1)
    return torch.cat([pts, inten], dim=1)  # (N, 4)


def load_extrinsics(yaml_item):
    t = np.array(yaml_item["translation"]).astype(float)
    q = np.array(yaml_item["rpy"]).astype(float)
    r = Rotation.from_euler("xyz", (q[0], q[1], q[2]), degrees=True).as_matrix()
    T = np.hstack((r, t.reshape(3, 1)))
    T = np.vstack((T, np.array([[0, 0, 0, 1]])))
    return T


def load_yaml_str(calibration_file_path):
    param = open(calibration_file_path, "r").read()
    if not param:
        return ""
    if "%YAML:1.0" in param:
        param = param.replace("%YAML:1.0", "%YAML 1.0")
    param = (
        param.replace("...", "")
        .replace("!!opencv-matrix", "")
        .replace("!<tag:yaml.org,2002:opencv-matrix>", "")
    )
    params = yaml.load(param, Loader=yaml.FullLoader)
    return params


class PointsMeter:

    def __init__(self, scale, intrinsics, beam_inclinations=None):
        self.V = []
        self.N = 0
        self.scale = scale
        self.intrinsics = intrinsics
        self.beam_inclinations = beam_inclinations

    def clear(self):
        self.V = []
        self.N = 0

    def prepare_inputs(self, *inputs):
        # Return inputs as torch tensors on GPU if possible to avoid CPU work.
        import torch

        outputs = []
        for inp in inputs:
            if torch.is_tensor(inp):
                # keep on device (detach) but don't move to CPU
                outputs.append(inp.detach())
            else:
                # numpy array or other -> convert to torch and move to cuda if available
                try:
                    t = torch.from_numpy(np.array(inp))
                    if torch.cuda.is_available():
                        t = t.cuda()
                    outputs.append(t)
                except Exception:
                    outputs.append(inp)

        return outputs

    def update(self, preds, truths, Filter=False):
        preds = preds / self.scale
        truths = truths / self.scale
        # Try to keep computations on GPU: convert panos to lidar point clouds with torch.
        preds_t, truths_t = self.prepare_inputs(preds, truths)
        chamLoss = chamfer_3DDist()

        # preds_t and truths_t shape: [B, H, W] or [H, W]
        import torch
        if torch.is_tensor(preds_t):
            # take first batch if batched
            p = preds_t[0] if preds_t.dim() == 3 else preds_t
        else:
            p = torch.from_numpy(np.array(preds_t)).cuda()

        if torch.is_tensor(truths_t):
            g = truths_t[0] if truths_t.dim() == 3 else truths_t
        else:
            g = torch.from_numpy(np.array(truths_t)).cuda()

        # get pred and gt lidar points on device
        try:
            pred_lidar = pano_to_lidar_torch(p, lidar_K=self.intrinsics, beam_inclinations=self.beam_inclinations, device=p.device)
            gt_lidar = pano_to_lidar_torch(g, lidar_K=self.intrinsics, beam_inclinations=self.beam_inclinations, device=g.device)
        except Exception:
            # fallback to numpy implementation if torch version fails
            pred_np = p.detach().cpu().numpy()
            gt_np = g.detach().cpu().numpy()
            pred_lidar = pano_to_lidar(pred_np)
            gt_lidar = pano_to_lidar(gt_np)

        if Filter:
            # filter_pcd currently expects numpy; only run if requested (keeps backward compatibility)
            try:
                pred_lidar_np = pred_lidar.detach().cpu().numpy() if torch.is_tensor(pred_lidar) else np.array(pred_lidar)
                mask_raydrop = filter_pcd(pred_lidar_np)
                if torch.is_tensor(pred_lidar):
                    pred_lidar = pred_lidar[torch.from_numpy(mask_raydrop).to(pred_lidar.device)]
                else:
                    pred_lidar = pred_lidar[mask_raydrop]
            except Exception:
                pass

        # chamfer expects tensors with shape [B, N, 3]
        if isinstance(pred_lidar, np.ndarray):
            if pred_lidar.shape[0] == 0 or gt_lidar.shape[0] == 0:
                chamfer_dis = torch.tensor(0.0)
                f_score = 0.0
            else:
                d1, d2, idx1, idx2 = chamLoss(torch.FloatTensor(pred_lidar[None, ...]).cuda(), torch.FloatTensor(gt_lidar[None, ...]).cuda())
                chamfer_dis = d1.mean() + d2.mean()
                threshold = 0.05
                f_score, precision, recall = fscore(d1, d2, threshold)
                f_score = f_score.cpu()[0]
        else:
            if pred_lidar.shape[0] == 0 or gt_lidar.shape[0] == 0:
                chamfer_dis = torch.tensor(0.0, device=p.device)
                f_score = 0.0
            else:
                d1, d2, idx1, idx2 = chamLoss(pred_lidar.unsqueeze(0).float(), gt_lidar.unsqueeze(0).float())
                chamfer_dis = d1.mean() + d2.mean()
                threshold = 0.05
                f_score, precision, recall = fscore(d1, d2, threshold)
                f_score = f_score.cpu()[0]

        self.V.append([chamfer_dis.detach().cpu(), f_score])
        self.N += 1

    def measure(self):
        # return self.V / self.N
        assert self.N == len(self.V)
        return np.array(self.V).mean(0)

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "CD"), self.measure()[0], global_step)

    def report(self):
        return f"CD f-score = {self.measure()}"


def write_pcd(
    save_filename,
    points,
    utime1=None,
    utime2=None,
    distance=None,
    ring=None,
    intensity=None,
    semantic_flag=None,
):
    pcd_header = "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z utime1 utime2 distance ring intensity semantic_flag\nSIZE 4 4 4 4 4 4 2 1 1\nTYPE F F F I I F U U U\nCOUNT 1 1 1 1 1 1 1 1 1\nWIDTH 98765\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 98765\nDATA ascii"
    if save_filename.split(".")[-1] != "pcd":
        raise ValueError("InvalidFileExtensionError")
    if utime1 is None or utime1.shape[0] != points.shape[0]:
        utime1 = np.zeros((points.shape[0], 1))
        # print("utime1 data Error")
    if utime2 is None or utime2.shape[0] != points.shape[0]:
        utime2 = np.zeros((points.shape[0], 1))
        # print("utime2 data Error")
    if distance is None or distance.shape[0] != points.shape[0]:
        distance = np.zeros((points.shape[0], 1))
        # print("distance data Error")
    if ring is None or ring.shape[0] != points.shape[0]:
        ring = np.zeros((points.shape[0], 1))
        # print("ring data Error")
    if intensity is None or intensity.shape[0] != points.shape[0]:
        intensity = np.zeros((points.shape[0], 1))
        # print("intensity data Error")
    if semantic_flag is None or semantic_flag.shape[0] != points.shape[0]:
        semantic_flag = np.zeros((points.shape[0], 1))
        # print("semantic_flag data Error")

    data = np.concatenate(
        (points[:, :3], utime1, utime2, distance, ring, intensity, semantic_flag),
        axis=1,
    )
    pcd_header = pcd_header.replace("98765", str(data.shape[0]))
    np.savetxt(save_filename, data, header=pcd_header, comments="", fmt="%.3f")


def write_pcdi(save_filename, pcdi, selected_lidar, distance=None):
    points = pcdi[:, :3]
    intensity = pcdi[:, 3:4] * 255
    # distance = np.linalg.norm(points[...,:3], axis=1).reshape(points.shape[0],1)
    ring = np.full(intensity.shape, selected_lidar)
    write_pcd(
        save_filename,
        points,
        ring=ring,
        intensity=intensity.astype(np.uint8),
        distance=distance,
    )


def get_pcdi(pcdi, selected_lidar, distance=None):
    points = pcdi[:, :3]
    intensity = pcdi[:, 3:4] * 255
    ring = np.full(intensity.shape, selected_lidar)
    data = get_pcd(
        points, ring=ring, intensity=intensity.astype(np.uint8), distance=distance
    )

    return data


def get_pcd(
    points,
    utime1=None,
    utime2=None,
    distance=None,
    ring=None,
    intensity=None,
    semantic_flag=None,
):
    if utime1 is None or utime1.shape[0] != points.shape[0]:
        utime1 = np.zeros((points.shape[0], 1))
        # print("utime1 data Error")
    if utime2 is None or utime2.shape[0] != points.shape[0]:
        utime2 = np.zeros((points.shape[0], 1))
        # print("utime2 data Error")
    if distance is None or distance.shape[0] != points.shape[0]:
        distance = np.zeros((points.shape[0], 1))
        # print("distance data Error")
    if ring is None or ring.shape[0] != points.shape[0]:
        ring = np.zeros((points.shape[0], 1))
        # print("ring data Error")
    if intensity is None or intensity.shape[0] != points.shape[0]:
        intensity = np.zeros((points.shape[0], 1))
        # print("intensity data Error")
    if semantic_flag is None or semantic_flag.shape[0] != points.shape[0]:
        semantic_flag = np.zeros((points.shape[0], 1))
        # print("semantic_flag data Error")

    data = np.concatenate(
        (points[:, :3], utime1, utime2, distance, ring, intensity, semantic_flag),
        axis=1,
    )
    return data
