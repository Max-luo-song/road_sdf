from typing import Dict, List, Tuple
from omegaconf import OmegaConf
import logging

import torch
import torch.nn as nn
from torch.nn import Parameter
from pytorch3d.ops import knn_points

from models.gaussians.basics import *

logger = logging.getLogger()

class RoadNodes(nn.Module):

    def __init__(
        self,
        class_name: str,
        ctrl: OmegaConf,
        reg: OmegaConf = None,
        networks: OmegaConf = None,
        scene_scale: float = 30.,
        scene_origin: torch.Tensor = torch.zeros(3),
        num_train_images: int = 300,
        device: torch.device = torch.device("cuda"),
        **kwargs
    ):
        super().__init__()
        self.class_prefix = class_name + "#"
        self.ctrl_cfg = ctrl
        self.reg_cfg = reg
        self.networks_cfg = networks
        self.scene_scale = scene_scale
        self.scene_origin = scene_origin
        self.num_train_images = num_train_images
        self.step = 0
        
        self.device = device
        self.ball_gaussians=self.ctrl_cfg.get("ball_gaussians", False)
        self.gaussian_2d = self.ctrl_cfg.get("gaussian_2d", False)
        
        # for evaluation
        self.in_test_set = False
        
        # init models
        self.xys_grad_norm = None
        self.max_2Dsize = None
        self._means = torch.zeros(1, 3, device=self.device)
        if self.ball_gaussians:
            self._scales = torch.zeros(1, 1, device=self.device)
        else:
            if self.gaussian_2d:
                self._scales = torch.zeros(1, 2, device=self.device)
            else:
                self._scales = torch.zeros(1, 3, device=self.device)
        self._quats = torch.zeros(1, 4, device=self.device)
        self._opacities = torch.zeros(1, 1, device=self.device)
        self._features_dc = torch.zeros(1, 3, device=self.device)
        self._features_rest = torch.zeros(1, num_sh_bases(self.sh_degree) - 1, 3, device=self.device)
        self._lane_mask = None
        field_cfg = (networks or {}).get("road_surface", {})
        self.surface_field = None
        if field_cfg.get("enabled", False):
            from models.road_surface import RoadSurfaceField
            # Auxiliary initialization must not change the existing GS random stream.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(field_cfg.get("initialization_seed", 2718))
                self.surface_field = RoadSurfaceField(field_cfg).to(self.device)
        
    @property
    def sh_degree(self):
        return self.ctrl_cfg.road_sh_degree

    @property
    def surface_orientation_enabled(self):
        return (self.surface_field is not None
                and self.surface_field.cfg.get("align_normals", False))

    def create_from_pcd(
        self,
        init_means: torch.Tensor,
        init_colors: torch.Tensor,
        lane_samples: Dict[str, torch.Tensor] = None,
    ) -> bool:
        if init_means.numel() == 0 or init_colors.numel() == 0:
            logger.warning(f"RoadNodes initialization skipped: empty input point cloud for {self.class_prefix}")
            return True

        grid_spacing = 0.1 # 10cm一个点
        k = 3
        chunk_size = 10000  # 每次处理1万个网格点，显存压力极小

        # 1. 建立基础网格(以grid_spacing为基准，再偏移一些)
        x_min, x_max = init_means[:, 0].min(), init_means[:, 0].max()
        y_min, y_max = init_means[:, 1].min(), init_means[:, 1].max()
        if torch.isclose(x_min, x_max) or torch.isclose(y_min, y_max):
            logger.warning(f"RoadNodes initialization skipped: insufficient spatial extent for {self.class_prefix}")
            return True

        x_coords = torch.arange(x_min, x_max, grid_spacing, device=self.device)
        y_coords = torch.arange(y_min, y_max, grid_spacing, device=self.device)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='ij')
        grid_a = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        offset = grid_spacing / 2.0
        grid_b = grid_a + torch.tensor([offset, 0.0], device=self.device)   # X 偏移
        grid_c = grid_a + torch.tensor([0.0, offset], device=self.device)   # Y 偏移
        grid_d = grid_a + torch.tensor([offset, offset], device=self.device) # 对角线偏移
        grid_xy_all = torch.cat([grid_a, grid_b, grid_c, grid_d], dim=0)

        resampled_means_list = []
        resampled_colors_list = []

        # 2. 分块处理网格点
        for i in range(0, grid_xy_all.shape[0], chunk_size):
            grid_xy = grid_xy_all[i : i + chunk_size]
            
            # 只计算当前 chunk 到原始点云的距离
            dist_mat = torch.cdist(grid_xy, init_means[:, :2]) 
            topk_dist, topk_idx = torch.topk(dist_mat, k, largest=False, dim=-1)
            del dist_mat
            torch.cuda.empty_cache()
            
            valid_mask = topk_dist[:, 0] < (grid_spacing * 4.0) #阈值调大一些
            if not valid_mask.any(): continue

            # 执行插值逻辑 
            curr_weights = 1.0 / (topk_dist[valid_mask] + 1e-6)
            curr_weights /= curr_weights.sum(dim=-1, keepdim=True)
            
            interp_z = (init_means[topk_idx[valid_mask], 2] * curr_weights).sum(dim=-1, keepdim=True)
            resampled_means_list.append(torch.cat([grid_xy[valid_mask], interp_z], dim=-1))
            
            neighbor_rgb = init_colors[topk_idx[valid_mask]]
            resampled_colors_list.append((neighbor_rgb * curr_weights[..., None]).sum(dim=1))

        if len(resampled_means_list) == 0:
            logger.warning(f"RoadNodes initialization skipped: no valid resampled points generated for {self.class_prefix}")
            return True

        resampled_means = torch.cat(resampled_means_list, dim=0)
        resampled_colors = torch.cat(resampled_colors_list, dim=0)

        self._means = Parameter(resampled_means, requires_grad=self.surface_field is None)
        num_points = resampled_means.shape[0]
        
        fixed_value = grid_spacing * 0.4
        fixed_scale_tensor = torch.full((num_points, 1), fixed_value, device=self.device)
        epsilon = 0.005
        # NOTE(gls)：让高斯在 XY 方向平铺，Z 方向极薄
        base_scale = torch.cat([fixed_scale_tensor, fixed_scale_tensor, torch.ones_like(fixed_scale_tensor) * epsilon], dim=-1)

        if lane_samples is not None:
            lane_points = lane_samples["points"]
            lane_tangents = lane_samples["tangents"]
            lane_dist2, lane_indices, _ = knn_points(
                resampled_means[None, :, :2], lane_points[None, :, :2], K=1
            )
            lane_indices = lane_indices[0, :, 0]
            lane_dist = torch.sqrt(lane_dist2[0, :, 0].clamp_min(0.0))
            lane_height_error = torch.abs(
                resampled_means[:, 2] - lane_points[lane_indices, 2]
            )
            self._lane_mask = (
                (lane_dist <= self.ctrl_cfg.get("lane_influence_radius", 0.2))
                & (lane_height_error <= self.ctrl_cfg.get("lane_height_tolerance", 0.3))
            )
            base_scale[self._lane_mask, 0] = self.ctrl_cfg.get(
                "lane_scale_parallel", 0.04
            )
            base_scale[self._lane_mask, 1] = self.ctrl_cfg.get(
                "lane_scale_perpendicular", 0.02
            )

        if self.ball_gaussians:
            # 如果是球形，取最大值
            self._scales = Parameter(torch.log(fixed_scale_tensor))
        else:
            self._scales = Parameter(torch.log(base_scale))

        # 旋转初始化：w=1, 意味着高斯主轴对齐世界坐标轴 (配合你压缩 Z 的策略，正好贴合路面)
        unit_quats = torch.zeros((num_points, 4), dtype=torch.float, device=self.device)
        unit_quats[:, 0] = 1.0 
        if self._lane_mask is not None and self._lane_mask.any():
            tangents = lane_tangents[lane_indices[self._lane_mask]]
            yaw = torch.atan2(tangents[:, 1], tangents[:, 0])
            unit_quats[self._lane_mask, 0] = torch.cos(yaw * 0.5)
            unit_quats[self._lane_mask, 3] = torch.sin(yaw * 0.5)
            logger.info(
                f"Lane-aware RoadNodes: {self._lane_mask.sum().item()}/{num_points} "
                f"Gaussians within {self.ctrl_cfg.get('lane_influence_radius', 0.2):.2f} m"
            )
        self._quats = Parameter(unit_quats, requires_grad=not self.surface_orientation_enabled)

        if (self._lane_mask is not None and self._lane_mask.any()
                and not (self.surface_field is not None
                         and self.surface_field.cfg.get("free_scales", False))):
            lane_mask = self._lane_mask

            def freeze_lane_perpendicular_scale(grad):
                grad = grad.clone()
                grad[lane_mask, 1] = 0
                return grad

            self._scales.register_hook(freeze_lane_perpendicular_scale)

        # SH 颜色初始化
        dim_sh = num_sh_bases(self.sh_degree)
        fused_color = RGB2SH(resampled_colors) 
        shs = torch.zeros((num_points, dim_sh, 3), device=self.device)
        
        if self.sh_degree > 0:
            shs[:, 0, :3] = fused_color
            # 高阶系数置零
            shs[:, 1:, :] = 0.0
        else:
            shs[:, 0, :3] = torch.logit(resampled_colors, eps=1e-10)
            
        self._features_dc = Parameter(shs[:, 0, :])
        self._features_rest = Parameter(shs[:, 1:, :])
        
        # 透明度初始化(会训练，初始化随机即可)
        self._opacities = Parameter(torch.logit(0.2 * torch.ones(num_points, 1, device=self.device)))
        if self.surface_field is not None:
            self.surface_field.initialize(resampled_means.detach(), init_means.detach())
        return False
    
    @property
    def colors(self):
        if self.sh_degree > 0:
            return SH2RGB(self._features_dc)
        else:
            return torch.sigmoid(self._features_dc)
    @property
    def shs_0(self):
        return self._features_dc
    @property
    def shs_rest(self):
        return self._features_rest
    @property
    def num_points(self):
        return self._means.shape[0]
    @property
    def get_scaling(self):
        if self.surface_field is not None and self.surface_field.cfg.get("free_scales", False):
            fixed_normal_scale = self.surface_field.cfg.get("fixed_normal_scale", None)
            if fixed_normal_scale is None:
                return torch.exp(self._scales)
            if self.ball_gaussians or self.gaussian_2d:
                raise ValueError("fixed_normal_scale requires ordinary 3D RoadNodes")
            xy_scale = torch.exp(self._scales[..., :2])
            normal_scale = torch.full_like(self._scales[..., 2:3], float(fixed_normal_scale))
            return torch.cat((xy_scale, normal_scale), dim=-1)
        if self.ball_gaussians:
            if self.gaussian_2d:
                scaling = torch.exp(self._scales).repeat(1, 2)
                scaling = torch.cat([scaling, torch.zeros_like(scaling[..., :1])], dim=-1)
                return scaling
            else:
                return torch.exp(self._scales).repeat(1, 3)
        else:
            if self.gaussian_2d:
                scaling = torch.exp(self._scales)
                scaling = torch.cat([scaling[..., :2], torch.zeros_like(scaling[..., :1])], dim=-1)
                return scaling
            else:
                raw_scale = torch.exp(self._scales)
                # 分别限制XY和Z
                scale_xy = torch.clamp(raw_scale[..., :2], min=0.02, max=0.08)
                scale_z = torch.clamp(raw_scale[..., 2:3], min=0.003, max=0.01)
                return torch.cat([scale_xy, scale_z], dim=-1)
                # return torch.exp(self._scales)
    @property
    def get_opacity(self):
        if self.surface_field is not None and self.surface_field.cfg.get("free_opacity", False):
            return torch.sigmoid(self._opacities)
        lowest_opacity = self.reg_cfg.get("lowest_opacity", None)
        # print("lowest_opacity:", lowest_opacity)
        return torch.sigmoid(self._opacities).clamp(min=lowest_opacity)

    @property
    def get_quats(self):
        return self.quat_act(self._quats)
    
    # NOTE(gls)：更新加一个epsilon，防止除0
    def quat_act(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True)
        # 防止范数为 0（添加一个很小的 epsilon，如 1e-8）
        norm = torch.clamp(norm, min=1e-8)
        return x / norm

    def preprocess_per_train_step(self, step: int):
        self.step = step

    def _surface_sync_key(self):
        return (self.surface_field.geometry_version(), id(self._means),
                self._means._version, id(self._quats), self._quats._version,
                self._means.device, self._means.dtype, self.surface_orientation_enabled)

    @torch.no_grad()
    def sync_surface_height(self):
        field = self.surface_field
        if (field is None or not field.initialized
                or self.step < field.cfg.get("apply_after", 1000)):
            return
        # Optimizer writes advance tensor versions. Forward immediately following
        # a completed bake, fixed geometry and frozen geometry reuse baked GS.
        # This also detects checkpoint loads, splitting and external optimizers.
        if getattr(self, "_surface_synced_key", None) == self._surface_sync_key():
            return
        base_key = (field.base_geometry_version(), id(self._means), self._means._version,
                    self._means.device, self._means.dtype)
        cache_valid = (getattr(self, "_surface_base_key", None) == base_key
                       and hasattr(self, "_surface_base_cache"))
        if self.surface_orientation_enabled and not cache_valid:
            self._surface_base_cache = (self._means.new_empty(self.num_points),
                                        torch.empty(self.num_points, dtype=torch.bool, device=self._means.device),
                                        self._means.new_empty(self.num_points, 2))
        chunk_size = field.cfg.get("query_chunk_size", 65536)
        for start in range(0, self.num_points, chunk_size):
            means = self._means[start:start + chunk_size]
            if self.surface_orientation_enabled:
                from models.road_surface import align_road_quaternions
                base = tuple(value[start:start + chunk_size] for value in self._surface_base_cache)
                if not cache_valid:
                    computed = field.base_height(means[:, :2], with_gradient=True)
                    for target, value in zip(base, computed):
                        target.copy_(value)
                height, normals, supported = field.height_and_normal(means[:, :2], base_geometry=base)
                quats = self._quats[start:start + chunk_size]
                aligned = align_road_quaternions(quats, normals)
                quats.copy_(torch.where(supported[..., None], aligned, quats))
            else:
                height, supported = field.height(means[:, :2])
            means[:, 2].copy_(torch.where(supported, height, means[:, 2]))
        # Our own Z/rotation writes also change versions; cache the final tokens.
        self._surface_synced_key = self._surface_sync_key()
        self._surface_base_key = ((field.base_geometry_version(), id(self._means), self._means._version,
                                  self._means.device, self._means.dtype)
                                 if self.surface_orientation_enabled else None)
    
    # NOTE(gls): 关闭路面致密化
    def postprocess_per_train_step(
        self,
        step: int,
        optimizer: torch.optim.Optimizer,
        radii: torch.Tensor,
        xys_grad: torch.Tensor,
        last_size: int,
    ) -> None:
        split_start = self.ctrl_cfg.get("edge_split_start", None)
        split_step = self.ctrl_cfg.get("edge_split_step", None)
        if split_start is None or split_step is None:
            return
        if split_start is not None and step == split_start:
            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

        self.after_train(radii, xys_grad, last_size)
        if split_step is not None and step == split_step:
            self.edge_split(optimizer)
            self.sync_surface_height()

    def edge_split(self, optimizer: torch.optim.Optimizer) -> None:
        ratio = self.ctrl_cfg.get("edge_split_ratio", 0.05)
        scores = self.xys_grad_norm / self.vis_counts.clamp_min(1)
        valid = scores > 0
        if not valid.any():
            logger.warning("Road edge split skipped: no visible road Gaussians")
            return
        n_split = min(valid.sum().item(), max(1, int(valid.sum().item() * ratio)))
        split_mask = torch.zeros_like(valid)
        split_mask[torch.topk(scores.masked_fill(~valid, -1), n_split).indices] = True

        parent_means = self._means.detach()[split_mask]
        parent_scales = self.get_scaling.detach()[split_mask]
        offsets = parent_scales[:, None, :2] * parent_means.new_tensor(
            [[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]]
        )[None]
        child_means = parent_means[:, None].repeat(1, 4, 1)
        child_means[:, :, :2] += offsets

        child_scales = self._scales.detach()[split_mask, None].repeat(1, 4, 1)
        child_scales[:, :, :2] -= math.log(2.0)

        self._means.data[split_mask] = child_means[:, 0]
        self._scales.data[split_mask] = child_scales[:, 0]
        self._means = Parameter(torch.cat([self._means.detach(), child_means[:, 1:].reshape(-1, 3)], dim=0),
                                requires_grad=self.surface_field is None)
        self._scales = Parameter(torch.cat([self._scales.detach(), child_scales[:, 1:].reshape(-1, 3)], dim=0))
        self._quats = Parameter(torch.cat([self._quats.detach(), self._quats.detach()[split_mask].repeat_interleave(3, dim=0)], dim=0),
                                requires_grad=not self.surface_orientation_enabled)
        self._features_dc = Parameter(torch.cat([self._features_dc.detach(), self._features_dc.detach()[split_mask].repeat_interleave(3, dim=0)], dim=0))
        self._features_rest = Parameter(torch.cat([self._features_rest.detach(), self._features_rest.detach()[split_mask].repeat_interleave(3, dim=0)], dim=0))
        self._opacities = Parameter(torch.cat([self._opacities.detach(), self._opacities.detach()[split_mask].repeat_interleave(3, dim=0)], dim=0))

        dup_in_optim(optimizer, split_mask, self.get_gaussian_param_groups(), n=3)
        logger.info(f"Road edge split: {n_split} parents -> {n_split * 4} children")
        self.xys_grad_norm = None
        self.vis_counts = None
        self.max_2Dsize = None

    def after_train(
        self,
        radii: torch.Tensor,
        xys_grad: torch.Tensor,
        last_size: int,
    ) -> None:
        with torch.no_grad():
            n_filtered = int(self.filter_mask.sum().item())
            radii = reduce_per_gaussian(radii, n_filtered, reduce="amax")
            grads = reduce_per_gaussian(xys_grad.norm(dim=-1), n_filtered, reduce="mean")
            
            # keep track of a moving average of grad norms
            visible_mask = radii > 0
            full_mask = torch.zeros(self.num_points, device=radii.device, dtype=torch.bool)
            full_mask[self.filter_mask] = visible_mask
            
            if self.xys_grad_norm is None:
                self.xys_grad_norm = torch.zeros(self.num_points, device=grads.device, dtype=grads.dtype)
                self.xys_grad_norm[self.filter_mask] = grads
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                assert self.vis_counts is not None
                self.vis_counts[full_mask] = self.vis_counts[full_mask] + 1
                self.xys_grad_norm[full_mask] = grads[visible_mask] + self.xys_grad_norm[full_mask]

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros(self.num_points, device=radii.device, dtype=torch.float32)
            newradii = radii[visible_mask]
            self.max_2Dsize[full_mask] = torch.maximum(
                self.max_2Dsize[full_mask], newradii / float(last_size)
            )
        
    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            self.class_prefix+"xyz": [self._means],
            self.class_prefix+"sh_dc": [self._features_dc],
            self.class_prefix+"sh_rest": [self._features_rest],
            self.class_prefix+"opacity": [self._opacities],
            self.class_prefix+"scaling": [self._scales],
            self.class_prefix+"rotation": [self._quats],
        }
    
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        groups = self.get_gaussian_param_groups()
        if self.surface_field is not None:
            groups[self.class_prefix + "surface_field"] = [
                p for p in self.surface_field.parameters() if p.requires_grad
            ]
        return groups

    # NOTE(gls)： 受控致密化，体现在致密化高斯限制在路面上
    def refinement_after_undercontrol(self, step, optimizer: torch.optim.Optimizer) -> None:
            assert step == self.step
            if self.step <= self.ctrl_cfg.warmup_steps:
                return
                
            # 1. 定义路面厚度约束
            ROAD_EPSILON = 0.005
            ROAD_LOG_EPSILON = math.log(ROAD_EPSILON)

            with torch.no_grad():
                reset_interval = self.ctrl_cfg.reset_alpha_interval
                # 开启致密化
                do_densification = (
                    self.step < self.ctrl_cfg.stop_split_at
                    and self.step % reset_interval > max(self.num_train_images, self.ctrl_cfg.refine_interval)
                )
                
                print(f"Class {self.class_prefix} current points: {self.num_points} @ step {self.step}")
                
                if do_densification:
                    assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                    
                    avg_grad_norm = self.xys_grad_norm / self.vis_counts
                    high_grads = (avg_grad_norm > self.ctrl_cfg.densify_grad_thresh).squeeze()
                    
                    # 计算当前路面的平均 Z 轴高度，确保新点不会偏移
                    avg_road_z = self._means[:, 2].mean()

                    # --- Split 逻辑 ---
                    splits = (self.get_scaling.max(dim=-1).values > self.ctrl_cfg.densify_size_thresh * self.scene_scale).squeeze()
                    if self.step < self.ctrl_cfg.stop_screen_size_at:
                        splits |= (self.max_2Dsize > self.ctrl_cfg.split_screen_size).squeeze()
                    splits &= high_grads
                    nsamps = self.ctrl_cfg.n_split_samples
                    (split_means, split_feature_dc, split_feature_rest, split_opacities, split_scales, split_quats) = self.split_gaussians(splits, nsamps)

                    # 投影 Split 生成的点：强制 Z 轴位置和 Scale
                    if not self.surface_field or not self.surface_field.cfg.get("free_scales", False):
                        split_means[:, 2] = avg_road_z
                        split_scales[:, 2] = ROAD_LOG_EPSILON

                    # --- Duplicate 逻辑 ---
                    dups = (self.get_scaling.max(dim=-1).values <= self.ctrl_cfg.densify_size_thresh * self.scene_scale).squeeze()
                    dups &= high_grads
                    (dup_means, dup_feature_dc, dup_feature_rest, dup_opacities, dup_scales, dup_quats) = self.dup_gaussians(dups)
                    
                    # 投影 Duplicate 生成的点：强制 Z 轴位置和 Scale
                    if not self.surface_field or not self.surface_field.cfg.get("free_scales", False):
                        dup_means[:, 2] = avg_road_z
                        dup_scales[:, 2] = ROAD_LOG_EPSILON

                    # 合并参数
                    self._means = Parameter(torch.cat([self._means.detach(), split_means, dup_means], dim=0),
                                            requires_grad=self.surface_field is None)
                    self._features_dc = Parameter(torch.cat([self._features_dc.detach(), split_feature_dc, dup_feature_dc], dim=0))
                    self._features_rest = Parameter(torch.cat([self._features_rest.detach(), split_feature_rest, dup_feature_rest], dim=0))
                    self._opacities = Parameter(torch.cat([self._opacities.detach(), split_opacities, dup_opacities], dim=0))
                    self._scales = Parameter(torch.cat([self._scales.detach(), split_scales, dup_scales], dim=0))
                    self._quats = Parameter(torch.cat([self._quats.detach(), split_quats, dup_quats], dim=0),
                                            requires_grad=not self.surface_orientation_enabled)
                    
                    self.max_2Dsize = torch.cat([self.max_2Dsize, torch.zeros_like(split_scales[:, 0]), torch.zeros_like(dup_scales[:, 0])], dim=0)
                    
                    param_groups = self.get_gaussian_param_groups()
                    dup_in_optim(optimizer, torch.where(splits)[0], param_groups, n=nsamps)
                    dup_in_optim(optimizer, torch.where(dups)[0], param_groups, 1)

                # --- Cull & Reset 逻辑 (保持原样) ---
                if self.step % reset_interval > max(self.num_train_images, self.ctrl_cfg.refine_interval):
                    deleted_mask = self.cull_gaussians()
                    param_groups = self.get_gaussian_param_groups()
                    remove_from_optim(optimizer, deleted_mask, param_groups)
                
                print(f"Class {self.class_prefix} left points: {self.num_points}")
                        
                if self.step % reset_interval == self.ctrl_cfg.refine_interval:
                    reset_value = torch.min(self.get_opacity.data, torch.ones_like(self._opacities.data) * self.ctrl_cfg.reset_alpha_value)
                    self._opacities.data = torch.logit(reset_value)
                    for group in optimizer.param_groups:
                        if group["name"] == self.class_prefix+"opacity":
                            param_state = optimizer.state.get(group["params"][0])
                            if param_state and "exp_avg" in param_state:
                                param_state["exp_avg"].zero_()
                                param_state["exp_avg_sq"].zero_()

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
        if self.step > self.ctrl_cfg.reset_alpha_interval:
            # cull huge ones
            toobigs = (
                torch.exp(self._scales).max(dim=-1).values > 
                self.ctrl_cfg.cull_scale_thresh * self.scene_scale
            ).squeeze()
            culls = culls | toobigs
            if self.step < self.ctrl_cfg.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                culls = culls | (self.max_2Dsize > self.ctrl_cfg.cull_screen_size).squeeze()
        self._means = Parameter(self._means[~culls].detach(),
                                requires_grad=self.surface_field is None)
        self._scales = Parameter(self._scales[~culls].detach())
        self._quats = Parameter(self._quats[~culls].detach(),
                                requires_grad=not self.surface_orientation_enabled)
        # self.colors_all = Parameter(self.colors_all[~culls].detach())
        self._features_dc = Parameter(self._features_dc[~culls].detach())
        self._features_rest = Parameter(self._features_rest[~culls].detach())
        self._opacities = Parameter(self._opacities[~culls].detach())

        print(f"     Cull: {n_bef - self.num_points}")
        return culls

    def split_gaussians(self, split_mask: torch.Tensor, samps: int) -> Tuple:
        """
        This function splits gaussians that are too large
        """

        n_splits = split_mask.sum().item()
        print(f"    Split: {n_splits}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            self.get_scaling[split_mask].repeat(samps, 1) * centered_samples
            # torch.exp(self._scales[split_mask].repeat(samps, 1)) * centered_samples
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
        new_scales = torch.log(torch.exp(self._scales[split_mask]) / size_fac).repeat(samps, 1)
        self._scales[split_mask] = torch.log(torch.exp(self._scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self._quats[split_mask].repeat(samps, 1)
        return new_means, new_feature_dc, new_feature_rest, new_opacities, new_scales, new_quats

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
        return dup_means, dup_feature_dc, dup_feature_rest, dup_opacities, dup_scales, dup_quats

    def get_gaussians(self, cam: dataclass_camera) -> Dict:
        # Optimizer steps can update the field after the per-step synchronization.
        # Evaluation/export uses baked Gaussian positions and needs no field query.
        if self.training:
            self.sync_surface_height()
        filter_mask = torch.ones_like(self._means[:, 0], dtype=torch.bool)
        self.filter_mask = filter_mask
        
        # get colors of gaussians
        colors = torch.cat((self._features_dc[:, None, :], self._features_rest), dim=1)
        if self.sh_degree > 0:
            viewdirs = self._means.detach() - cam.camtoworlds.data[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.ctrl_cfg.sh_degree_interval, self.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])
            
        activated_opacities = self.get_opacity
        activated_scales = self.get_scaling
        activated_rotations = self.get_quats
        actovated_colors = rgbs
        
        # collect gaussians information
        gs_dict = dict(
            _means=self._means[filter_mask],
            _opacities=activated_opacities[filter_mask],
            _rgbs=actovated_colors[filter_mask],
            _scales=activated_scales[filter_mask],
            _quats=activated_rotations[filter_mask],
        )
        
        # check nan and inf in gs_dict
        for k, v in gs_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in gaussian {k} at step {self.step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in gaussian {k} at step {self.step}")
                
        return gs_dict
    
    def compute_reg_loss(self):
        loss_dict = {}
        sharp_shape_reg_cfg = self.reg_cfg.get("sharp_shape_reg", None)
        if sharp_shape_reg_cfg is not None:
            w = sharp_shape_reg_cfg.w
            max_gauss_ratio = sharp_shape_reg_cfg.max_gauss_ratio
            step_interval = sharp_shape_reg_cfg.step_interval
            if self.step % step_interval == 0:
                # scale regularization
                scale_exp = self.get_scaling
                scale_reg = torch.maximum(scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1), torch.tensor(max_gauss_ratio)) - max_gauss_ratio
                scale_reg = scale_reg.mean() * w
                loss_dict["sharp_shape_reg"] = scale_reg

        flatten_reg = self.reg_cfg.get("flatten", None)
        if flatten_reg is not None:
            sclaings = self.get_scaling
            min_scale, _ = torch.min(sclaings, dim=1)
            min_scale = torch.clamp(min_scale, 0, 30)
            flatten_loss = torch.abs(min_scale).mean()
            loss_dict["flatten"] = flatten_loss * flatten_reg.w
        
        sparse_reg = self.reg_cfg.get("sparse_reg", None)
        if sparse_reg:
            if (self.cur_radii > 0).sum():
                opacity = torch.sigmoid(self._opacities)
                opacity = opacity.clamp(1e-6, 1-1e-6)
                log_opacity = opacity * torch.log(opacity)
                log_one_minus_opacity = (1-opacity) * torch.log(1 - opacity)
                sparse_loss = -1 * (log_opacity + log_one_minus_opacity)[self.cur_radii > 0].mean()
                loss_dict["sparse_reg"] = sparse_loss * sparse_reg.w

        # compute the max of scaling
        max_s_square_reg = self.reg_cfg.get("max_s_square_reg", None)
        if max_s_square_reg is not None and not self.ball_gaussians:
            loss_dict["max_s_square"] = torch.mean((self.get_scaling.max(dim=1).values) ** 2) * max_s_square_reg.w
        return loss_dict
    
    def load_state_dict(self, state_dict: Dict, **kwargs) -> str:
        N = state_dict["_means"].shape[0]
        self._means = Parameter(torch.zeros((N,) + self._means.shape[1:], device=self.device),
                                requires_grad=self.surface_field is None)
        self._scales = Parameter(torch.zeros((N,) + self._scales.shape[1:], device=self.device))
        self._quats = Parameter(torch.zeros((N,) + self._quats.shape[1:], device=self.device),
                                requires_grad=not self.surface_orientation_enabled)
        self._features_dc = Parameter(torch.zeros((N,) + self._features_dc.shape[1:], device=self.device))
        self._features_rest = Parameter(torch.zeros((N,) + self._features_rest.shape[1:], device=self.device))
        self._opacities = Parameter(torch.zeros((N,) + self._opacities.shape[1:], device=self.device))
        msg = super().load_state_dict(state_dict, **kwargs)
        return msg
    
    def export_gaussians_to_ply(self, alpha_thresh: float) -> Dict:
        means = self._means
        direct_color = self.colors
        
        activated_opacities = self.get_opacity
        mask = activated_opacities.squeeze() > alpha_thresh
        return {
            "positions": means[mask],
            "colors": direct_color[mask],
        }
