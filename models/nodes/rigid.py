from typing import Dict, List, Tuple, Optional
import logging
import random
import math
import torch
from torch.nn import Parameter
from torch.nn.functional import normalize, sigmoid
from scipy.interpolate import interp1d, CubicSpline

from models.modules import ConditionalDeformNetwork
from models.gaussians.basics import *
from models.gaussians.vanilla import VanillaGaussians

import numpy as np
import cv2
import matplotlib.cm as cm
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from plyfile import PlyData, PlyElement

logger = logging.getLogger()

# Threshold for detecting nearly opposite vectors in quaternion-from-vectors computation.
# When dot(from, to) < this value, the vectors are considered anti-parallel and the
# rotation is set to a 180-degree rotation around an arbitrary perpendicular axis.
OPPOSITE_VEC_DOT_THRESHOLD = -0.99999  # -(1 - 1e-5)

class RigidNodes(VanillaGaussians):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def num_instances(self):
        return self.instances_fv.shape[1]

    @property
    def num_frames(self):
        return self.instances_fv.shape[0]

    def get_pts_valid_mask(self):
        """
        get the mask for valid points
        """
        return self.instances_fv[self.cur_frame][self.point_ids[..., 0]]

    def set_cur_frame(self, frame_id: int):
        self.cur_frame = frame_id

    def register_normalized_timestamps(self, normalized_timestamps: int):
        self.normalized_timestamps = normalized_timestamps

    @staticmethod
    def interpolate_waypoints(
        waypoints: list,
        num_frames: int,
        method: str = "linear",
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> torch.Tensor:
        """将 waypoints 插值为指定帧数的轨迹偏移量。"""
        waypoints = np.array(waypoints)

        if waypoints.ndim != 2 or waypoints.shape[1] != 3:
            raise ValueError(f"waypoints 应该是 (K, 3)，实际 shape: {waypoints.shape}")

        t = np.linspace(0, 1, len(waypoints))
        t_new = np.linspace(0, 1, num_frames)
        offset_traj = np.zeros((num_frames, 3))

        if method == "linear":
            for i in range(3):
                f = interp1d(t, waypoints[:, i])
                offset_traj[:, i] = f(t_new)
        elif method == "cubic":
            if len(waypoints) < 3:
                raise ValueError("cubic 至少需要 >=3 个 waypoints")
            for i in range(3):
                f = CubicSpline(t, waypoints[:, i])
                offset_traj[:, i] = f(t_new)
        else:
            raise ValueError(f"Unsupported method: {method}")

        return torch.tensor(offset_traj, device=device, dtype=dtype)
    
    def _build_visibility_mask(self, time_window=None):
        fv = torch.zeros(
            (self.num_frames, 1),
            dtype=torch.bool,
            device=self.instances_fv.device,
        )

        if time_window is None:
            fv[:] = True
        else:
            assert len(time_window) == 2, \
                "time_window should be [start_frame, end_frame]"

            start_frame, end_frame = time_window

            if end_frame == -1:
                end_frame = self.num_frames

            assert 0 <= start_frame < self.num_frames
            assert start_frame < end_frame <= self.num_frames

            fv[start_frame:end_frame] = True

        return fv

    def create_from_pcd(self, instance_pts_dict: Dict[str, torch.Tensor]) -> None:
        """
        instance_pts_dict: {
            id in dataset: {
                "class_name": str,
                "pts": torch.Tensor, (N, 3)
                "colors": torch.Tensor, (N, 3)
                "poses": torch.Tensor, (num_frame, 4, 4)
                "size": torch.Tensor, (3, )
                "frame_info": torch.Tensor, (num_frame)
                "num_pts": int,
            },
        }
        """
        # collect all instances
        init_means = []
        init_colors = []
        instances_pose = []
        instances_size = []
        instances_fv = []
        point_ids = []
        print("instance_pts_dict keys:", instance_pts_dict.keys())
        for id_in_model, (id_in_dataset, v) in enumerate(instance_pts_dict.items()):
            # print("v.keys:", v.keys())
            if "pts" in v:
                if isinstance(v["pts"], list):
                    num_pts = len(v["pts"])
                    # 如果需要，可以将列表转换为张量
                    v["pts"] = torch.tensor(v["pts"], dtype=torch.float32)
                else:
                    num_pts = v["pts"].shape[0]
            else:
                num_pts = 10000  # 默认1000个点
                v["pts"] = torch.randn(num_pts, 3)  # 创建随机点

            if "colors" in v:
                if isinstance(v["colors"], list):
                    v["colors"] = torch.tensor(v["colors"], dtype=torch.float32)
                # 确保 colors 的数量与 pts 匹配
                if v["colors"].shape[0] != num_pts:
                    # 如果不匹配，创建随机颜色
                    v["colors"] = torch.rand(num_pts, 3)
            else:
                # 如果没有 colors，创建随机颜色
                v["colors"] = torch.rand(num_pts, 3)

            init_means.append(v["pts"])
            init_colors.append(v["colors"])
            instances_pose.append(v["poses"].unsqueeze(1))
            instances_size.append(v["size"])
            instances_fv.append(v["frame_info"].unsqueeze(1))
            # point_ids.append(torch.full((v["num_pts"], 1), id_in_model, dtype=torch.long)) # gls
            point_ids.append(torch.full((num_pts, 1), id_in_model, dtype=torch.long))

        init_means = torch.cat(init_means, dim=0).to(self.device)  # (N, 3)
        init_colors = torch.cat(init_colors, dim=0).to(self.device)  # (N, 3)
        instances_pose = torch.cat(instances_pose, dim=1).to(
            self.device
        )  # (num_frame, num_instances, 4, 4)
        self.instances_size = torch.stack(instances_size).to(
            self.device
        )  # (num_instances, 3)
        self.instances_fv = torch.cat(instances_fv, dim=1).to(
            self.device
        )  # (num_frame, num_instances)
        self.point_ids = torch.cat(point_ids, dim=0).to(self.device)
        instances_quats = self.get_instances_quats(instances_pose)
        instances_trans = instances_pose[..., :3, 3]

        # initialize the means, scales, quats, and colors
        self._means = Parameter(init_means)
        distances, _ = k_nearest_sklearn(self._means.data, 3)
        distances = torch.from_numpy(distances)
        avg_dist = distances.mean(dim=-1, keepdim=True).to(self.device)
        avg_dist = avg_dist.clamp(0.002, 100)
        self._scales = Parameter(torch.log(avg_dist.repeat(1, 3)))
        self._quats = Parameter(random_quat_tensor(self.num_points).to(self.device))
        dim_sh = num_sh_bases(self.sh_degree)

        # pose refinement
        self.instances_quats = Parameter(
            self.quat_act(instances_quats)
        )  # (num_frame, num_instances, 4)
        self.instances_trans = Parameter(
            instances_trans
        )  # (num_frame, num_instances, 3)

        fused_color = RGB2SH(init_colors)  # float range [0, 1]
        shs = torch.zeros((fused_color.shape[0], dim_sh, 3)).float().to(self.device)
        if self.sh_degree > 0:
            shs[:, 0, :3] = fused_color
            shs[:, 1:, 3:] = 0.0
        else:
            shs[:, 0, :3] = torch.logit(init_colors, eps=1e-10)
        self._features_dc = Parameter(shs[:, 0, :])
        self._features_rest = Parameter(shs[:, 1:, :])
        self._opacities = Parameter(
            torch.logit(0.1 * torch.ones(self.num_points, 1, device=self.device))
        )

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = self.get_gaussian_param_groups()
        param_groups[self.class_prefix + "ins_rotation"] = [self.instances_quats]
        param_groups[self.class_prefix + "ins_translation"] = [self.instances_trans]
        return param_groups

    def get_instances_quats(self, instances_pose: torch.Tensor) -> torch.Tensor:
        """
        Convert the pose to quaternion for all frames and instances
        """
        num_frames = instances_pose.shape[0]
        num_instances = instances_pose.shape[1]
        quats = torch.zeros(num_frames * num_instances, 4, device=self.device)

        poses = instances_pose[..., :3, :3].view(-1, 3, 3)
        valid_mask = self.instances_fv.view(-1)
        _quats = matrix_to_quaternion(poses[valid_mask])
        _quats = self.quat_act(_quats)

        quats[valid_mask] = _quats
        quats[~valid_mask, 0] = 1.0
        return quats.reshape(num_frames, num_instances, 4)

    def refinement_after(self, step: int, optimizer: torch.optim.Optimizer) -> None:
        assert step == self.step
        if self.step <= self.ctrl_cfg.warmup_steps:
            return
        with torch.no_grad():
            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.ctrl_cfg.reset_alpha_interval
            do_densification = (
                self.step < self.ctrl_cfg.stop_split_at
                and self.step % reset_interval
                > max(self.num_train_images, self.ctrl_cfg.refine_interval)
            )
            # split & duplicate
            print(
                f"Class {self.class_prefix} current points: {self.num_points} @ step {self.step}"
            )
            if do_densification:
                assert (
                    self.xys_grad_norm is not None
                    and self.vis_counts is not None
                    and self.max_2Dsize is not None
                )

                avg_grad_norm = self.xys_grad_norm / self.vis_counts
                high_grads = (
                    avg_grad_norm > self.ctrl_cfg.densify_grad_thresh
                ).squeeze()

                splits = (
                    self.get_scaling.max(dim=-1).values
                    > self.ctrl_cfg.densify_size_thresh * self.scene_scale
                ).squeeze()
                if self.step < self.ctrl_cfg.stop_screen_size_at:
                    splits |= (
                        self.max_2Dsize > self.ctrl_cfg.split_screen_size
                    ).squeeze()
                splits &= high_grads
                splits = splits.reshape(-1)
                nsamps = self.ctrl_cfg.n_split_samples
                (
                    split_means,
                    split_feature_dc,
                    split_feature_rest,
                    split_opacities,
                    split_scales,
                    split_quats,
                    split_ids,
                ) = self.split_gaussians(splits, nsamps)

                dups = (
                    self.get_scaling.max(dim=-1).values
                    <= self.ctrl_cfg.densify_size_thresh * self.scene_scale
                ).squeeze()
                dups &= high_grads
                dups = dups.reshape(-1)
                (
                    dup_means,
                    dup_feature_dc,
                    dup_feature_rest,
                    dup_opacities,
                    dup_scales,
                    dup_quats,
                    dup_ids,
                ) = self.dup_gaussians(dups)

                self._means = Parameter(
                    torch.cat([self._means.detach(), split_means, dup_means], dim=0)
                )
                # self.colors_all = Parameter(torch.cat([self.colors_all.detach(), split_colors, dup_colors], dim=0))
                self._features_dc = Parameter(
                    torch.cat(
                        [self._features_dc.detach(), split_feature_dc, dup_feature_dc],
                        dim=0,
                    )
                )
                self._features_rest = Parameter(
                    torch.cat(
                        [
                            self._features_rest.detach(),
                            split_feature_rest,
                            dup_feature_rest,
                        ],
                        dim=0,
                    )
                )
                self._opacities = Parameter(
                    torch.cat(
                        [self._opacities.detach(), split_opacities, dup_opacities],
                        dim=0,
                    )
                )
                self._scales = Parameter(
                    torch.cat([self._scales.detach(), split_scales, dup_scales], dim=0)
                )
                self._quats = Parameter(
                    torch.cat([self._quats.detach(), split_quats, dup_quats], dim=0)
                )
                self.point_ids = torch.cat([self.point_ids, split_ids, dup_ids], dim=0)

                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros_like(split_scales[:, 0]),
                        torch.zeros_like(dup_scales[:, 0]),
                    ],
                    dim=0,
                )

                split_idcs = torch.where(splits)[0]
                param_groups = self.get_gaussian_param_groups()
                dup_in_optim(optimizer, split_idcs, param_groups, n=nsamps)

                dup_idcs = torch.where(dups)[0]
                param_groups = self.get_gaussian_param_groups()
                dup_in_optim(optimizer, dup_idcs, param_groups, 1)

            # cull NOTE: Offset all the opacity reset logic by refine_every so that we don't
            # save checkpoints right when the opacity is reset (saves every 2k)
            if self.step % reset_interval > max(
                self.num_train_images, self.ctrl_cfg.refine_interval
            ):
                deleted_mask = self.cull_gaussians()
                param_groups = self.get_gaussian_param_groups()
                remove_from_optim(optimizer, deleted_mask, param_groups)
            print(f"Class {self.class_prefix} left points: {self.num_points}")

            # reset opacity
            if self.step % reset_interval == self.ctrl_cfg.refine_interval:
                # NOTE: in nerfstudio, reset_value = cull_alpha_thresh * 0.8
                # we align to original repo of gaussians spalting
                reset_value = torch.min(
                    self.get_opacity.data,
                    torch.ones_like(self._opacities.data)
                    * self.ctrl_cfg.reset_alpha_value,
                )
                self._opacities.data = torch.logit(reset_value)
                # reset the exp of optimizer
                for group in optimizer.param_groups:
                    if group["name"] == self.class_prefix + "opacity":
                        old_params = group["params"][0]
                        param_state = optimizer.state[old_params]
                        param_state["exp_avg"] = torch.zeros_like(
                            param_state["exp_avg"]
                        )
                        param_state["exp_avg_sq"] = torch.zeros_like(
                            param_state["exp_avg_sq"]
                        )
            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

    def cull_gaussians(self):
        """
        This function deletes gaussians with under a certain opacity threshold
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (self.get_opacity.data < self.ctrl_cfg.cull_alpha_thresh).squeeze()
        if self.ctrl_cfg.cull_out_of_bound:
            culls = culls | self.get_out_of_bound_mask()
        if self.step > self.ctrl_cfg.reset_alpha_interval:
            # cull huge ones
            toobigs = (
                torch.exp(self._scales).max(dim=-1).values
                > self.ctrl_cfg.cull_scale_thresh * self.scene_scale
            ).squeeze()
            culls = culls | toobigs
            if self.step < self.ctrl_cfg.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                culls = (
                    culls | (self.max_2Dsize > self.ctrl_cfg.cull_screen_size).squeeze()
                )
        culls = culls.reshape(-1)
        self._means = Parameter(self._means[~culls].detach())
        self._scales = Parameter(self._scales[~culls].detach())
        self._quats = Parameter(self._quats[~culls].detach())
        # self.colors_all = Parameter(self.colors_all[~culls].detach())
        self._features_dc = Parameter(self._features_dc[~culls].detach())
        self._features_rest = Parameter(self._features_rest[~culls].detach())
        self._opacities = Parameter(self._opacities[~culls].detach())
        self.point_ids = self.point_ids[~culls]

        print(f"     Cull: {n_bef - self.num_points}")
        return culls

    def split_gaussians(self, split_mask: torch.Tensor, samps: int = 2) -> Tuple:
        """
        This function splits gaussians that are too large
        """

        n_splits = split_mask.sum().item()
        print(f"    Split: {n_splits}")
        centered_samples = torch.randn(
            (samps * n_splits, 3), device=self.device
        )  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self._scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quat_act(self._quats[split_mask])  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self._means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        # new_colors_all = self.colors_all[split_mask].repeat(samps, 1, 1)
        new_feature_dc = self._features_dc[split_mask].repeat(samps, 1)
        new_feature_rest = self._features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self._opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self._scales[split_mask]) / size_fac).repeat(
            samps, 1
        )
        self._scales[split_mask] = torch.log(
            torch.exp(self._scales[split_mask]) / size_fac
        )
        # step 5, sample new quats
        new_quats = self._quats[split_mask].repeat(samps, 1)
        # step 6, sample new ids
        new_ids = self.point_ids[split_mask].repeat(samps, 1)
        return (
            new_means,
            new_feature_dc,
            new_feature_rest,
            new_opacities,
            new_scales,
            new_quats,
            new_ids,
        )

    def dup_gaussians(self, dup_mask: torch.Tensor) -> Tuple:
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        print(f"      Dup: {n_dups}")
        dup_means = self._means[dup_mask]
        # dup_colors = self.colors_all[dup_mask]
        dup_feature_dc = self._features_dc[dup_mask]
        dup_feature_rest = self._features_rest[dup_mask]
        dup_opacities = self._opacities[dup_mask]
        dup_scales = self._scales[dup_mask]
        dup_quats = self._quats[dup_mask]
        dup_ids = self.point_ids[dup_mask]
        return (
            dup_means,
            dup_feature_dc,
            dup_feature_rest,
            dup_opacities,
            dup_scales,
            dup_quats,
            dup_ids,
        )

    def get_out_of_bound_mask(self):
        """
        This function checks if the gaussians are out of instance boxes
        """
        # get the instance boxes
        per_pts_size = self.instances_size[self.point_ids[..., 0]]
        instance_pts = self._means

        mask = (instance_pts.abs() > per_pts_size / 2).any(dim=-1)
        return mask

    def transform_means(self, means: torch.Tensor) -> torch.Tensor:
        """
        transform the means of instances to world space
        according to the pose at the current frame
        """
        assert (
            means.shape[0] == self.point_ids.shape[0]
        ), "its a bug here, we need to pass the mask for points_ids"
        if self.in_test_set and (
            self.cur_frame - 1 > 0 and self.cur_frame + 1 < self.num_frames
        ):
            # use the previous and next frame to interpolate the pose
            _quats_prev_frame = self.instances_quats[self.cur_frame - 1]
            _quats_next_frame = self.instances_quats[self.cur_frame + 1]
            _quats_cur_frame = self.instances_quats[self.cur_frame]
            interpolated_quats = interpolate_quats(_quats_prev_frame, _quats_next_frame)

            inter_valid_mask = (
                self.instances_fv[self.cur_frame - 1]
                & self.instances_fv[self.cur_frame + 1]
            )
            quats_cur_frame = torch.where(
                inter_valid_mask[:, None], interpolated_quats, _quats_cur_frame
            )
        else:
            quats_cur_frame = self.instances_quats[self.cur_frame]  # (num_instances, 4)
        rot_cur_frame = quat_to_rotmat(
            self.quat_act(quats_cur_frame)
        )  # (num_instances, 3, 3)
        rot_per_pts = rot_cur_frame[self.point_ids[..., 0]]  # (num_points, 3, 3)

        if self.in_test_set and (
            self.cur_frame - 1 > 0 and self.cur_frame + 1 < self.num_frames
        ):
            _prev_ins_trans = self.instances_trans[self.cur_frame - 1]
            _next_ins_trans = self.instances_trans[self.cur_frame + 1]
            _cur_ins_trans = self.instances_trans[self.cur_frame]
            interpolated_trans = (_prev_ins_trans + _next_ins_trans) * 0.5

            inter_valid_mask = (
                self.instances_fv[self.cur_frame - 1]
                & self.instances_fv[self.cur_frame + 1]
            )
            trans_cur_frame = torch.where(
                inter_valid_mask[:, None], interpolated_trans, _cur_ins_trans
            )
        else:
            trans_cur_frame = self.instances_trans[self.cur_frame]  # (num_instances, 3)
        trans_per_pts = trans_cur_frame[self.point_ids[..., 0]]

        # transform the means to world space
        means = (
            torch.bmm(
                rot_per_pts.float(), means.unsqueeze(-1).float()  # 确保是 float32
            ).squeeze(-1)
            + trans_per_pts.float()
        )  # 确保 trans_per_pts 也是 float32
        return means

    def transform_quats(self, quats: torch.Tensor) -> torch.Tensor:
        """
        transform the quats of instances to world space
        according to the pose at the current frame
        """
        assert (
            quats.shape[0] == self.point_ids.shape[0]
        ), "its a bug here, we need to pass the mask for points_ids"
        global_quats_cur_frame = self.instances_quats[self.cur_frame]
        global_quats_per_pts = global_quats_cur_frame[self.point_ids[..., 0]]

        global_quats_per_pts = self.quat_act(global_quats_per_pts)
        _quats = self.quat_act(quats)
        return quat_mult(global_quats_per_pts, _quats)

    def load_ply_as_gs(
        self,
        ply_path: str,
        device: torch.device = torch.device("cpu"),
        sh_degree: int = 3,
        scale_factor: float = 1.0,
    ):
        """
        读取 3DGS PLY，并返回与本网络一致的字段 (线性 scale, 原始 opacity)。
        """
        plydata = PlyData.read(ply_path)
        v = plydata["vertex"]
        names = v.data.dtype.names
        N = v.count

        def num_sh_bases(deg: int) -> int:
            return (deg + 1) ** 2

        def col(name, dtype=torch.float32, vd=1):
            return torch.from_numpy(v[name]).to(dtype).view(N, vd)

        # ---------------- means -----------------
        means = (
            torch.stack(
                [col("x").squeeze(), col("y").squeeze(), col("z").squeeze()], 1
            ).to(device)
            * scale_factor
        )  # (N,3)

        # ---------------- normals (optional) ----
        normals = None
        if all(k in names for k in ("nx", "ny", "nz")):
            normals = torch.stack(
                [col("nx").squeeze(), col("ny").squeeze(), col("nz").squeeze()], 1
            ).to(device)

        # ---------------- scales --------
        scales = (
            torch.stack(
                [
                    col("scale_0").squeeze(),
                    col("scale_1").squeeze(),
                    col("scale_2").squeeze(),
                ],
                1,
            ).to(device)
            * scale_factor
        )  # (N,3)

        # ---------------- quats -----------------
        quats = torch.stack(
            [
                col("rot_0").squeeze(),
                col("rot_1").squeeze(),
                col("rot_2").squeeze(),
                col("rot_3").squeeze(),
            ],
            1,
        ).to(device)

        # ---------------- opacity (原始值) -------
        opacities = col("opacity").to(device)  # (N,1) or (N,)

        # ---------------- SH 0 阶 ----------------
        f_dc = torch.stack(
            [col("f_dc_0").squeeze(), col("f_dc_1").squeeze(), col("f_dc_2").squeeze()],
            1,
        ).to(
            device
        )  # (N,3,1)

        # ---------------- SH 高阶 ----------------
        exp_rest_dim = num_sh_bases(sh_degree) - 1  # e.g. 15 for L=3
        rest_names = sorted(
            [n for n in names if n.startswith("f_rest_")],
            key=lambda s: int(s.split("_")[-1]),
        )

        if len(rest_names) == 0:
            f_rest = torch.zeros((N, exp_rest_dim, 3), device=device)
        else:
            # 按列收集 → reshape
            f_rest_raw = torch.stack(
                [col(n).squeeze() for n in rest_names], 1
            )  # (N, K)
            if f_rest_raw.shape[1] % 3 != 0:
                raise ValueError("f_rest_* 列数不是 3 的倍数")
            f_rest = f_rest_raw.view(N, 3, exp_rest_dim).transpose(1, 2).contiguous().to(device)  # (N,K,3)


        # ---------------- 打包 --------------------
        asset = dict(
            means=Parameter(means),
            scales=Parameter(scales),
            quats=Parameter(quats),
            opacities=Parameter(opacities),
            features_dc=Parameter(f_dc),
            features_rest=Parameter(f_rest),
            normals=normals,
            num_points=N,
            sh_degree=sh_degree,
        )
        return asset

    def get_gaussians(self, cam: dataclass_camera) -> Dict[str, torch.Tensor]:
        filter_mask = torch.ones_like(self._means[:, 0], dtype=torch.bool)
        # filter_mask = (self.point_ids.squeeze(-1) != 1)
        self.filter_mask = filter_mask
        # NOTE: hack here, need to consider a gaussian filter for efficient rendering

        world_means = self.transform_means(self._means)
        world_quats = self.transform_quats(self._quats)

        # get colors of gaussians
        colors = torch.cat((self._features_dc[:, None, :], self._features_rest), dim=1)
        if self.sh_degree > 0:
            viewdirs = world_means.detach() - cam.camtoworlds.data[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.ctrl_cfg.sh_degree_interval, self.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])

        valid_mask = self.get_pts_valid_mask()

        activated_opacities = self.get_opacity * valid_mask.float().unsqueeze(-1)
        activated_scales = self.get_scaling
        activated_rotations = self.quat_act(world_quats)
        activated_colors = rgbs

        # collect gaussians information
        gs_dict = dict(
            _means=world_means[filter_mask],
            _opacities=activated_opacities[filter_mask],
            _rgbs=activated_colors[filter_mask],
            _scales=activated_scales[filter_mask],
            _quats=activated_rotations[filter_mask],
        )

        # check nan and inf in gs_dict
        for k, v in gs_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in gaussian {k} at step {self.step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in gaussian {k} at step {self.step}")

        self._gs_cache = {
            "_scales": activated_scales[filter_mask],
        }

        return gs_dict

    def edit_trajectory(
        self,
        instance_id: int,
        offset: list,
    ) -> Dict[str, torch.Tensor]:

        unique_vals, counts = torch.unique(self.point_ids, return_counts=True)
        if instance_id not in unique_vals:
            raise ValueError(
                f"Invalid instance id: {instance_id}, the valid instance id include: {unique_vals}"
            )

        offset = torch.tensor(
            offset, device=self.instances_trans.device, dtype=self.instances_trans.dtype
        )
        self.instances_trans[:, instance_id : instance_id + 1, :] += offset

    def edit_trajectory_interpolate(
        self,
        instance_id: int,
        offset_windows: list,
    ):
        """在现有轨迹上叠加分段时间窗口偏移。

        offset_windows: list of [start_frame, end_frame, [dx, dy, dz]]
        每个窗口内偏移从 0 平滑增长到 delta，窗口结束后偏移保持。
        多窗口累积叠加。自动根据运动方向变化修正朝向。
        """
        num_frames = self.num_frames
        offset_traj = torch.zeros(
            num_frames, 3,
            device=self.instances_trans.device,
            dtype=self.instances_trans.dtype,
        )

        for start_f, end_f, delta in offset_windows:
            if start_f == end_f:
                # start == end: 整条轨迹平移，所有帧统一加 delta
                offset_traj += torch.tensor(delta, device=offset_traj.device, dtype=offset_traj.dtype)
                continue

            start_f = max(0, min(start_f, num_frames - 1))
            end_f = min(max(end_f, start_f + 1), num_frames)
            delta = torch.tensor(delta, device=offset_traj.device, dtype=offset_traj.dtype)
            frames_in_window = end_f - start_f
            for f in range(start_f, end_f):
                t = (f - start_f) / frames_in_window
                t = t * t * (3 - 2 * t)  # smoothstep ease-in-out
                offset_traj[f] += delta * t
            if end_f < num_frames:
                offset_traj[end_f:] += delta

        # 旋转修正：根据运动方向变化调整朝向
        positions_orig = self.instances_trans[:, instance_id, :].clone()
        positions_new = positions_orig + offset_traj

        directions_orig = torch.diff(positions_orig, n=1, dim=0,
                                     append=positions_orig[-1:] - positions_orig[-2:-1])
        directions_new = torch.diff(positions_new, n=1, dim=0,
                                    append=positions_new[-1:] - positions_new[-2:-1])

        q_delta = self.rotation_from_vectors(directions_orig, directions_new)
        original_quats = self.instances_quats[:, instance_id, :].clone()
        self.instances_quats[:, instance_id, :] = quat_mult(q_delta, original_quats)

        self.instances_trans[:, instance_id, :] = positions_new

    def generate_legend_image(
        self,
        ids: torch.Tensor,  # shape (N,)
        colors: torch.Tensor,  # shape (N,3) float 0~1
        image_output_pth: str,
    ) -> np.ndarray:
        """
        依据 ids 与 colors 生成 legend png，并返回 numpy(H,W,3,uint8)。

        若 save_path 给定则落盘。
        """
        ids_np = ids.to("cpu").numpy()
        colors_np = colors.to("cpu").numpy()
        N = len(ids_np)

        fig_h = max(2, N * 0.35)  # 高度随类别数增
        fig, ax = plt.subplots(figsize=(2.0, fig_h), dpi=100)

        # 反向画让 0 在最下方（完全可按自己习惯）
        for row, (idx, col) in enumerate(zip(ids_np[::-1], colors_np[::-1])):
            # 左边色块
            ax.add_patch(
                patches.Rectangle(
                    (0, row), width=1.0, height=1.0, facecolor=col, edgecolor="none"
                )
            )
            # 右边文字
            ax.text(
                1.2,
                row + 0.5,
                str(int(idx)),
                va="center",
                ha="left",
                fontsize=10,
                color="black",
            )

        ax.set_xlim(0, 3)
        ax.set_ylim(0, N)
        ax.axis("off")
        fig.tight_layout(pad=0)

        if image_output_pth is not None and not os.path.exists(image_output_pth):
            fig.savefig(image_output_pth, bbox_inches="tight", pad_inches=0)
            print(f"[legend] saved image -> {image_output_pth}")

        plt.close(fig)

    def color_legned_gaussians(self, image_output_pth: str) -> Dict[str, torch.Tensor]:
        unique_vals, counts = torch.unique(self.point_ids, return_counts=True)

        # To find instance
        N = len(unique_vals)
        cmap = cm.get_cmap("gist_ncar")  # 连续调色板
        colors_np = cmap(np.linspace(0, 1, N, endpoint=False))[
            :, :3
        ]  # (N,3)  float 0~1

        print("\n========== Instance ↔ Color Mapping ==========")
        for idx, unique_val in enumerate(unique_vals):
            color = colors_np[idx]  # 0~1 float
            color_255 = (color * 255).astype(int)  # 转成RGB整数

            print(
                f"instance_id = {int(unique_val):3d} | "
                f"rgb = {color_255} | "
                f"num_pts = {int(counts[idx])}"
            )
        print("=============================================\n")

        colors = torch.from_numpy(colors_np).to(
            dtype=self._features_dc.dtype,  # 通常是 torch.float32 或 float16
            device=self._features_dc.device,  # same GPU
        )

        # colors_sh_format = colors - 0.5
        colors_logit_format = torch.logit(torch.clamp(colors, 1e-6, 1 - 1e-6))

        for index, unique_val in enumerate(unique_vals):
            instance_mask = self.point_ids.squeeze(-1) == unique_val
            self._features_dc[instance_mask, :] = colors_logit_format[index]
        self.ctrl_cfg.sh_degree = 0

        self.generate_legend_image(unique_vals, colors, image_output_pth)

    def get_instance_activated_gs_dict(self, ins_id: int) -> Dict[str, torch.Tensor]:
        pts_mask = self.point_ids[..., 0] == ins_id
        if pts_mask.sum() < 100:
            return None
        local_means = self._means[pts_mask]
        activated_opacities = torch.sigmoid(self._opacities[pts_mask])
        activated_scales = torch.exp(self._scales[pts_mask])
        activated_local_rotations = self.quat_act(self._quats[pts_mask])
        gaussian_dict = {
            "means": local_means,
            "opacities": activated_opacities,
            "scales": activated_scales,
            "quats": activated_local_rotations,
            "sh_dcs": self._features_dc[pts_mask],
            "sh_rests": self._features_rest[pts_mask],
            "ids": self.point_ids[pts_mask],
        }
        return gaussian_dict

    def compute_reg_loss(self) -> Dict[str, torch.Tensor]:
        loss_dict = super().compute_reg_loss()
        scaling_reg = self.reg_cfg.get("scaling_reg", None)
        if scaling_reg is not None:
            w = scaling_reg.w
            precentile = scaling_reg.precentile
            stop_after = scaling_reg.stop_after
            start_after = scaling_reg.start_after

            if self.step < stop_after and self.step > start_after and w > 0:
                scale_prod = self._gs_cache["_scales"].prod(dim=-1)
                p = torch.kthvalue(
                    scale_prod, int(scale_prod.shape[0] * precentile)
                ).values
                # penalize the scales that are too large
                loss_dict["scaling_percentile_reg"] = (
                    torch.relu(scale_prod - p).mean() * w
                )

        # temporal smooth regularization
        temporal_smooth_reg = self.reg_cfg.get("temporal_smooth_reg", None)
        if temporal_smooth_reg is not None:
            instance_mask = self.instances_fv[self.cur_frame]
            if instance_mask.sum() > 0:
                trans_cfg = temporal_smooth_reg.get("trans", None)
                if trans_cfg is not None:
                    fi_interval = random.randint(1, trans_cfg.smooth_range)
                    if (
                        self.cur_frame >= fi_interval
                        and self.cur_frame < self.num_frames - fi_interval
                    ):
                        valid_mask = (
                            self.instances_fv[self.cur_frame - fi_interval]
                            & self.instances_fv[self.cur_frame + fi_interval]
                            & self.instances_fv[self.cur_frame]
                        )
                        if valid_mask.sum() > 0:
                            cur_trans = self.instances_trans[self.cur_frame]
                            pre_trans = self.instances_trans[
                                self.cur_frame - fi_interval
                            ].data
                            next_trans = self.instances_trans[
                                self.cur_frame + fi_interval
                            ].data
                            loss = (
                                (
                                    next_trans[valid_mask]
                                    + pre_trans[valid_mask]
                                    - 2 * cur_trans[valid_mask]
                                )
                                .abs()
                                .mean()
                            )
                            loss_dict["trans_temporal_smooth"] = loss * trans_cfg.w
        return loss_dict

    def state_dict(self) -> Dict:
        state_dict = super().state_dict()
        state_dict.update(
            {
                "points_ids": self.point_ids,
                "instances_size": self.instances_size,
                "instances_fv": self.instances_fv,
            }
        )
        return state_dict

    def load_state_dict(self, state_dict: Dict, **kwargs) -> str:
        self.point_ids = state_dict.pop("points_ids")
        self.instances_size = state_dict.pop("instances_size")
        self.instances_fv = state_dict.pop("instances_fv")
        self.instances_trans = Parameter(
            torch.zeros(self.num_frames, self.num_instances, 3, device=self.device)
        )
        self.instances_quats = Parameter(
            torch.zeros(self.num_frames, self.num_instances, 4, device=self.device)
        )
        msg = super().load_state_dict(state_dict, **kwargs)
        return msg

    # editting functions
    def remove_instances(self, remove_id_list: List[int]) -> None:
        """
        remove instances from the model

        Args:
            remove_id_list: list of instance ids to be removed
        """
        for ins_ids in remove_id_list:
            mask = ~(self.point_ids[..., 0] == ins_ids)
            self._means = Parameter(self._means[mask])
            self._scales = Parameter(self._scales[mask])
            self._quats = Parameter(self._quats[mask])
            self._features_dc = Parameter(self._features_dc[mask])
            self._features_rest = Parameter(self._features_rest[mask])
            self._opacities = Parameter(self._opacities[mask])
            self.point_ids = self.point_ids[mask]

    def collect_gaussians_from_ids(self, ids: List[int]) -> Dict:
        gaussian_dict = {}

        for instance_id in ids:
            mask = self.point_ids[..., 0] == instance_id
            if not mask.any():
                existing_ids = torch.unique(self.point_ids[..., 0]).tolist()
                raise ValueError(
                    f"Instance {instance_id} not found.\n"
                    f"Existing ids: {sorted(existing_ids)}"
                )
            gaussian_dict[instance_id] = {
                "_means": self._means[mask],
                "_scales": self._scales[mask],
                "_quats": self._quats[mask],
                "_features_dc": self._features_dc[mask],
                "_features_rest": self._features_rest[mask],
                "_opacities": self._opacities[mask],
                "point_ids": self.point_ids[mask],
            }

        return gaussian_dict

    def replace_instances(self, replace_dict: Dict[int, int]) -> None:
        """
        replace instances from the model

        Args:
            replace_dict: {
                ins_id(to be replaced): ins_id(replace with)
                ...
            }
        """
        new_gaussians_dict = self.collect_gaussians_from_ids(replace_dict.values())
        for ins_id, new_id in replace_dict.items():
            self.remove_instances([ins_id])
            new_gaussian = new_gaussians_dict[new_id]
            self._means = Parameter(
                torch.cat([self._means, new_gaussian["_means"]], dim=0)
            )
            self._scales = Parameter(
                torch.cat([self._scales, new_gaussian["_scales"]], dim=0)
            )
            self._quats = Parameter(
                torch.cat([self._quats, new_gaussian["_quats"]], dim=0)
            )
            self._features_dc = Parameter(
                torch.cat([self._features_dc, new_gaussian["_features_dc"]], dim=0)
            )
            self._features_rest = Parameter(
                torch.cat([self._features_rest, new_gaussian["_features_rest"]], dim=0)
            )
            self._opacities = Parameter(
                torch.cat([self._opacities, new_gaussian["_opacities"]], dim=0)
            )
            # keeps original point ids
            self.point_ids = torch.cat(
                [self.point_ids, torch.full_like(new_gaussian["point_ids"], ins_id)],
                dim=0,
            )

    def replace_instance_with_ply(
        self,
        target_id: int,  # 要被替换掉的旧实例 id
        ply_path: str,  # 新 *.ply
        new_id: int = None,
    ):  # 新实例在场景里想用的 id；默认沿用旧 id
        """
        用 ply 文件中的 3D-Gaussian 资产替换掉场景里的一个实例
        ----------------------------------------------------------
        1. 读 ply -> 得到 means / scales / quats / features / opacities
        2. 删掉旧实例 target_id 的全部高斯
        3. 把新高斯 append 回各个张量，并生成 point_ids
        """

        def quat_mul(q1, q2):
            """
            Hamilton 乘法：两个 (N,4) 四元数 → (N,4)
            约定顺序 [w, x, y, z]
            """
            w1, x1, y1, z1 = q1.unbind(-1)
            w2, x2, y2, z2 = q2.unbind(-1)
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            return torch.stack([w, x, y, z], dim=-1)

        def apply_RT_to_gs(gs, R=torch.eye(3), t=None):
            """
            就地对 gs 字典做刚体变换:
                x' = R x + t
                q' = R ⊗ q
            """
            dev = gs["means"].device
            R = R.to(dev)
            if t is not None:
                t = t.to(dev)

            # means
            gs["means"].data = (R @ gs["means"].T).T
            if t is not None:
                gs["means"].data += t

            # quats
            rot_q = matrix_to_quaternion(R)
            rot_q = rot_q.expand(gs["quats"].shape[0], -1)
            gs["quats"].data = normalize(quat_mul(rot_q, gs["quats"]), dim=-1)

        # 1. 读取 ply
        gs = self.load_ply_as_gs(
            ply_path, device=self.device, sh_degree=self.sh_degree, scale_factor=1.0
        )

        R = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)

        t_final = torch.tensor([0.0, 0.0, 0.3], device=self.device)

        apply_RT_to_gs(gs, R, t_final)  # 先旋转再做最终平移

        # 2. 删旧实例
        self.remove_instances([target_id])

        # 3. 新实例 id（如果想用同一个 id 就复用）
        if new_id is None:
            new_id = target_id

        N_new = gs["means"].shape[0]
        new_point_ids = torch.full((N_new, 1), new_id, device=self.device)

        self._means = Parameter(torch.cat([self._means, gs["means"]], dim=0))
        self._scales = Parameter(torch.cat([self._scales, gs["scales"]], dim=0))
        self._quats = Parameter(torch.cat([self._quats, gs["quats"]], dim=0))
        self._opacities = Parameter(
            torch.cat([self._opacities, gs["opacities"]], dim=0)
        )
        self._features_dc = Parameter(
            torch.cat([self._features_dc, gs["features_dc"]], dim=0)
        )
        self._features_rest = Parameter(
            torch.cat([self._features_rest, gs["features_rest"]], dim=0)
        )
        self.point_ids = torch.cat([self.point_ids, new_point_ids], dim=0)

        print(f"[info] instance {target_id} 已被来自 {ply_path} 的 {N_new} 个高斯替换")

    def _apply_trajectory_and_rotation(
        self,
        target_id: int,
        new_instance_trans: torch.Tensor,
        new_instance_quats: torch.Tensor,
        waypoints: list = None,
        method: str = "linear",
        ego_traj: torch.Tensor = None,
    ):
        num_frames = self.instances_trans.shape[0]

        # 确定参考轨迹 (F, 3)
        if target_id == -1:
            assert ego_traj is not None, "target_id=-1 时必须提供 ego_traj"
            ref_trans = ego_traj  # (F, 3)
        else:
            ref_trans = self.instances_trans[:, target_id, :].clone()  # (F, 3)

        # waypoints 处理
        if waypoints is not None and len(waypoints) > 0:
            K = len(waypoints)

            if K == 1:
                # 单点：全局平移
                offset = torch.tensor(
                    waypoints[0], device=ref_trans.device, dtype=ref_trans.dtype
                )
                new_instance_trans[:, 0, :] = ref_trans + offset

            elif K >= 2:
                # 多点：按 method 插值
                if K == 2 and method == "cubic":
                    print(
                        f"[warning] waypoints 只有 {K} 个点，cubic 需要至少 3 个点，降级为 linear"
                    )
                    method = "linear"

                offset_traj = self.interpolate_waypoints(
                    waypoints=waypoints,
                    num_frames=num_frames,
                    method=method,
                    device=ref_trans.device,
                    dtype=ref_trans.dtype,
                )
                new_instance_trans[:, 0, :] = ref_trans + offset_traj
        else:
            # 无 waypoints
            new_instance_trans[:, 0, :] = ref_trans

        positions_new = new_instance_trans.squeeze(1)  # (F,3)
        dirs = torch.diff(positions_new, dim=0, append=positions_new[-1:])

        # 最后一帧沿用前一帧方向
        if len(dirs) > 1:
            dirs[-1] = dirs[-2]

        # 静止帧处理
        speed_eps = 1e-4

        for i in range(1, len(dirs)):
            if torch.norm(dirs[i, :2]) < speed_eps:
                dirs[i] = dirs[i - 1]

        # 第一帧也可能静止
        if torch.norm(dirs[0, :2]) < speed_eps:
            dirs[0] = torch.tensor(
                [1.0, 0.0, 0.0], device=dirs.device, dtype=dirs.dtype
            )

        dirs = dirs / (torch.norm(dirs, dim=-1, keepdim=True) + 1e-8)

        yaw = torch.atan2(dirs[:, 1], dirs[:, 0])

        asset_yaw_offset = torch.pi
        yaw += asset_yaw_offset

        # unwrap yaw 防止 pi -> -pi 跳变
        for i in range(1, len(yaw)):
            delta = yaw[i] - yaw[i - 1]
            if delta > torch.pi:
                yaw[i] -= 2 * torch.pi
            elif delta < -torch.pi:
                yaw[i] += 2 * torch.pi

        half_yaw = yaw * 0.5

        qw = torch.cos(half_yaw)
        qx = torch.zeros_like(qw)
        qy = torch.zeros_like(qw)
        qz = torch.sin(half_yaw)

        final_quats_data = torch.stack([qw, qx, qy, qz], dim=-1)

        for i in range(1, len(final_quats_data)):
            if torch.dot(final_quats_data[i - 1], final_quats_data[i]) < 0:
                final_quats_data[i] = -final_quats_data[i]

        new_instance_quats[:, 0, :] = final_quats_data

        return new_instance_trans, new_instance_quats

    def add_instance_with_ply(
        self,
        target_id: int,
        ply_path: str,
        waypoints: list = None,
        method: str = "linear",
        ego_traj: torch.Tensor = None,
        time_window: list = None,
        dataset=None,
        models: dict = None,
        brightness_radius: float = 3.0,
    ):
        new_id = self.instances_fv.shape[1]

        self.instances_fv = torch.cat(
            [
                self.instances_fv,
                self._build_visibility_mask(time_window),
            ],
            dim=1,
        )

        gs = self.load_ply_as_gs(
            ply_path, device=self.device, sh_degree=self.sh_degree, scale_factor=1.0
        )

        if dataset is not None:
            from models.color_utils import (
                match_gaussian_brightness,
                compute_nearby_rigid_luma,
                compute_local_luma,
            )

            # --- compute reference trajectory ---
            if target_id == -1:
                ref_trans = ego_traj  # (F, 3) torch tensor
            else:
                ref_trans = self.instances_trans[:, target_id, :]

            # --- determine waypoint offset ---
            if waypoints and len(waypoints) > 0:
                wp_arr = np.asarray(waypoints[0] if len(waypoints) == 1
                                    else waypoints)
                offset = torch.tensor(
                    wp_arr.mean(axis=0) if wp_arr.ndim == 2 else wp_arr,
                    device=ref_trans.device, dtype=ref_trans.dtype,
                )
            else:
                offset = torch.zeros(3, device=ref_trans.device, dtype=ref_trans.dtype)

            num_f = ref_trans.shape[0]

            if time_window is None:
                start_frame = 0
                end_frame = num_f
            else:
                start_frame, end_frame = time_window
                if end_frame == -1:
                    end_frame = num_f

            mid_frame = (start_frame + end_frame - 1) // 2

            sample_frames = [
                start_frame,
                mid_frame,
                end_frame - 1,
            ]
            positions_frames = [
                (ref_trans[f].float() + offset.float(), f)
                for f in sample_frames
            ]

            ref_name = None
            reference_luma = None

            # 1) nearby rigid only
            reference_luma = compute_nearby_rigid_luma(
                self, positions_frames,
                radius=brightness_radius, sh_degree=self.sh_degree,
            )
            if reference_luma is not None:
                ref_name = "nearby_rigid"
            elif models is not None:
                # 2) all nearby Gaussians
                reference_luma = compute_local_luma(
                    models, positions_frames,
                    radius=brightness_radius, sh_degree=self.sh_degree,
                )
                if reference_luma is not None:
                    ref_name = "local"

            if reference_luma is not None:
                gs, scale = match_gaussian_brightness(
                    gs, None, self.sh_degree, reference_luma=reference_luma,
                )
                if scale is not None:
                    print(f"[brightness] {ply_path.split('/')[-1]}: scale={scale:.3f} (ref={ref_name})")
            else:
                # 3) background image
                gs, scale = match_gaussian_brightness(gs, dataset, self.sh_degree)
                if scale is not None:
                    print(f"[brightness] {ply_path.split('/')[-1]}: scale={scale:.3f} (ref=background)")

        N_new = gs["means"].shape[0]
        new_point_ids = torch.full((N_new, 1), new_id, device=self.device)

        self._means = Parameter(torch.cat([self._means, gs["means"]], dim=0))
        self._scales = Parameter(torch.cat([self._scales, gs["scales"]], dim=0))
        self._quats = Parameter(torch.cat([self._quats, gs["quats"]], dim=0))
        self._opacities = Parameter(
            torch.cat([self._opacities, gs["opacities"]], dim=0)
        )
        self._features_dc = Parameter(
            torch.cat([self._features_dc, gs["features_dc"]], dim=0)
        )
        self._features_rest = Parameter(
            torch.cat([self._features_rest, gs["features_rest"]], dim=0)
        )

        # 确定参考轨迹和旋转
        if target_id == -1:
            assert ego_traj is not None, "target_id=-1 时必须提供 ego_traj"
            ref_trans = ego_traj.unsqueeze(1)  # (F, 1, 3)
            ref_quats = torch.zeros(self.num_frames, 1, 4, device=self.device)
            ref_quats[:, 0, 0] = 1.0  # 单位四元数 [1, 0, 0, 0]
        else:
            ref_trans = self.instances_trans[:, target_id : target_id + 1, :].clone()
            ref_quats = self.instances_quats[:, target_id : target_id + 1, :].clone()

        new_instance_trans = ref_trans.clone()
        new_instance_quats = ref_quats.clone()

        new_instance_trans, new_instance_quats = self._apply_trajectory_and_rotation(
            target_id=target_id,
            new_instance_trans=new_instance_trans,
            new_instance_quats=new_instance_quats,
            waypoints=waypoints,
            method=method,
            ego_traj=ego_traj,
        )

        new_trans_param = Parameter(
            torch.cat([self.instances_trans, new_instance_trans], dim=1)
        )
        new_quat_param = Parameter(
            torch.cat([self.instances_quats, new_instance_quats], dim=1)
        )

        self.instances_trans = new_trans_param
        self.instances_quats = new_quat_param
        self.point_ids = torch.cat([self.point_ids, new_point_ids], dim=0)
        print(f"[info] 已添加 instance {new_id} ，来自 {ply_path} 的 {N_new} 个高斯")

    def add_instance(
        self,
        target_id: int,
        add_id: int,
        waypoints: list = None,
        method: str = "linear",
        ego_traj: torch.Tensor = None,
        time_window: list = None,
    ):
        new_id = self.instances_fv.shape[1]

        self.instances_fv = torch.cat(
            [
                self.instances_fv,
                self._build_visibility_mask(time_window),
            ],
            dim=1,
        )

        new_gaussians_dict = self.collect_gaussians_from_ids([add_id])
        gs = new_gaussians_dict[0]
        print(gs.keys())

        N_new = gs["_means"].shape[0]
        new_point_ids = torch.full((N_new, 1), new_id, device=self.device)

        self._means = Parameter(torch.cat([self._means, gs["_means"]], dim=0))
        self._scales = Parameter(torch.cat([self._scales, gs["_scales"]], dim=0))
        self._quats = Parameter(torch.cat([self._quats, gs["_quats"]], dim=0))
        self._opacities = Parameter(
            torch.cat([self._opacities, gs["_opacities"]], dim=0)
        )
        self._features_dc = Parameter(
            torch.cat([self._features_dc, gs["_features_dc"]], dim=0)
        )
        self._features_rest = Parameter(
            torch.cat([self._features_rest, gs["_features_rest"]], dim=0)
        )

        if target_id == -1:
            assert ego_traj is not None, "target_id=-1 时必须提供 ego_traj"
            ref_trans = ego_traj.unsqueeze(1)
            ref_quats = torch.zeros(self.num_frames, 1, 4, device=self.device)
            ref_quats[:, 0, 0] = 1.0
        else:
            ref_trans = self.instances_trans[:, target_id : target_id + 1, :].clone()
            ref_quats = self.instances_quats[:, target_id : target_id + 1, :].clone()

        new_instance_trans = ref_trans.clone()
        new_instance_quats = ref_quats.clone()

        new_instance_trans, new_instance_quats = self._apply_trajectory_and_rotation(
            target_id=target_id,
            new_instance_trans=new_instance_trans,
            new_instance_quats=new_instance_quats,
            waypoints=waypoints,
            method=method,
            ego_traj=ego_traj,
        )

        new_trans_param = Parameter(
            torch.cat([self.instances_trans, new_instance_trans], dim=1)
        )
        new_quat_param = Parameter(
            torch.cat([self.instances_quats, new_instance_quats], dim=1)
        )

        self.instances_trans = new_trans_param
        self.instances_quats = new_quat_param
        self.point_ids = torch.cat([self.point_ids, new_point_ids], dim=0)
        print(
            f"[info] 已添加 instance_{new_id} ，来自原场景 instance_{add_id} 的 {N_new} 个高斯"
        )

    def export_instance_to_ply(
        self,
        path: str,
        instance_id: Optional[int] = None,
        alpha_thresh: float = 0.001,
    ):
        """
        导出指定实例(们)的高斯资产为 3DGS PLY。

        Args:
            path          : 输出文件路径
            instance_id  : int；若 None 则导出全部
        """

        # --- 构造全局 mask -------------------------------------------------
        ids = self.point_ids[..., 0]  # (N,)
        if instance_id is None:
            mask_inst = torch.ones_like(ids, dtype=torch.bool)
        else:
            mask_inst = ids == instance_id  # (N,)

        alphas = self.get_opacity.squeeze(-1)  # (N,)
        mask_alpha = alphas > alpha_thresh

        mask = mask_inst & mask_alpha  # (N,)

        if mask.sum() == 0:
            print(f"[export_instance_to_ply] nothing to export for id {instance_id}")
            return

        # 3. 取数据并转 CPU
        m = self._means[mask].cpu().numpy()  # (M,3)

        sigma = self._scales[mask].cpu().numpy()

        q = self._quats[mask].cpu().numpy()  # (M,4)
        op = self._opacities[mask].cpu().numpy().squeeze()  # (M,)  logit

        fdc = self._features_dc[mask].cpu().numpy()  # (M,3)
        frest = self._features_rest[mask].cpu().numpy()  # (M,K,3)
        M, K = frest.shape[0], frest.shape[1]

        # 4. 组织结构化数组
        dtype_list = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("f_dc_0", "f4"),
            ("f_dc_1", "f4"),
            ("f_dc_2", "f4"),
        ]

        for k in range(K * 3):
            dtype_list.append((f"f_rest_{k}", "f4"))

        dtype_list += [
            ("opacity", "f4"),
            ("scale_0", "f4"),
            ("scale_1", "f4"),
            ("scale_2", "f4"),
            ("rot_0", "f4"),
            ("rot_1", "f4"),
            ("rot_2", "f4"),
            ("rot_3", "f4"),
        ]

        arr = np.empty(M, dtype=dtype_list)

        # pos
        arr["x"], arr["y"], arr["z"] = m[:, 0], m[:, 1], m[:, 2]

        # normals = 0
        arr["nx"].fill(0)
        arr["ny"].fill(0)
        arr["nz"].fill(0)

        # dc
        arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = fdc[:, 0], fdc[:, 1], fdc[:, 2]

        # rest
        if K > 0:
            frest_flat = frest.reshape(M, -1)  # (M,K*3)
            for k in range(K * 3):
                arr[f"f_rest_{k}"] = frest_flat[:, k]

        # opacity (logit)
        arr["opacity"] = op

        # scale
        arr["scale_0"], arr["scale_1"], arr["scale_2"] = (
            sigma[:, 0],
            sigma[:, 1],
            sigma[:, 2],
        )

        # quat
        arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = (
            q[:, 0],
            q[:, 1],
            q[:, 2],
            q[:, 3],
        )

        # 5. 写 PLY
        ply_el = PlyElement.describe(arr, "vertex")
        PlyData([ply_el], text=False).write(path)

        print(f"[export_instance_to_ply] saved {arr.shape[0]} gaussians to {path}")

    def rotation_from_vectors(self, vec_from, vec_to):
        """
        计算从一个向量到另一个向量的旋转四元数。
        假定四元数格式为 [w, x, y, z]。

        :param vec_from: (N, 3) 起始方向向量
        :param vec_to: (N, 3) 目标方向向量
        :return: (N, 4) 旋转四元数
        """
        vec_from = normalize(vec_from, dim=-1)
        vec_to = normalize(vec_to, dim=-1)

        dot = torch.sum(vec_from * vec_to, dim=-1)
        cross = torch.cross(vec_from, vec_to, dim=-1)

        quat = torch.zeros(vec_from.shape[0], 4, device=vec_from.device)

        quat[:, 0] = torch.sqrt(0.5 * (1 + dot))
        xyz_part = cross / (2 * quat[:, 0].unsqueeze(-1) + 1e-8)  # 加一个小的epsilon防止除以0
        quat[:, 1:] = xyz_part

        opposite_mask = dot < OPPOSITE_VEC_DOT_THRESHOLD
        if torch.any(opposite_mask):
            quat[opposite_mask] = torch.tensor([0.0, 0.0, 1.0, 0.0], device=quat.device).expand(torch.sum(opposite_mask), 4)

        return normalize(quat, dim=-1)
