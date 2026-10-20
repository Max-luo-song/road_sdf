"""Road-only height field and auxiliary SDF renderer (no CUDA extensions required).

Inspired by ExtraGS §3.2, not a reproduction of its unavailable implementation.
Distances, heights, ray bounds and finite-difference steps are in world metres.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from utils.road_surface_prior import build_road_surface_grid


class HashGrid2D(nn.Module):
    """Bilinearly interpolated multiresolution hash features in normalized XY."""

    def __init__(self, levels=8, features=2, table_size=16384,
                 min_resolution=16, max_resolution=512):
        super().__init__()
        if levels < 1 or features < 1 or table_size < 1:
            raise ValueError("Hash-grid dimensions must be positive")
        resolutions = torch.logspace(math.log10(min_resolution),
                                     math.log10(max_resolution), levels).round().long()
        self.register_buffer("resolutions", resolutions)
        self.tables = nn.Parameter(torch.empty(levels, table_size, features))
        nn.init.uniform_(self.tables, -1e-4, 1e-4)
        self.out_dim = levels * features

    def forward(self, xy):
        outputs = []
        offsets = xy.new_tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.long)
        for level, resolution in enumerate(self.resolutions):
            p = xy * resolution
            base = p.floor().long()
            frac = p - base
            corners = base[..., None, :] + offsets
            indices = (corners[..., 0] ^ (corners[..., 1] * 2654435761)) % self.tables.shape[1]
            weights = torch.where(offsets.bool(), frac[..., None, :],
                                  1 - frac[..., None, :]).prod(-1)
            outputs.append((self.tables[level][indices] * weights[..., None]).sum(-2))
        return torch.cat(outputs, -1)


class RoadSurfaceField(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        self.encoding = HashGrid2D(
            levels=cfg.get("levels", 8), features=cfg.get("features", 2),
            table_size=cfg.get("table_size", 16384),
            min_resolution=cfg.get("min_resolution", 16),
            max_resolution=cfg.get("max_resolution", 512),
        )
        width = cfg.get("hidden_dim", 64)
        self.geometry = nn.Sequential(nn.Linear(self.encoding.out_dim, width), nn.Softplus(),
                                      nn.Linear(width, 2))
        nn.init.zeros_(self.geometry[-1].weight)
        nn.init.zeros_(self.geometry[-1].bias)
        # Color has a separate encoding: texture gradients cannot change geometry features.
        self.color_encoding = HashGrid2D(
            levels=cfg.get("levels", 8), features=cfg.get("features", 2),
            table_size=cfg.get("table_size", 16384),
            min_resolution=cfg.get("min_resolution", 16),
            max_resolution=cfg.get("max_resolution", 512),
        )
        self.color = nn.Sequential(nn.Linear(self.color_encoding.out_dim + 9, width),
                                   nn.ReLU(), nn.Linear(width, width), nn.ReLU(),
                                   nn.Linear(width, 3), nn.Sigmoid())
        for name, value in {
            "origin": torch.zeros(2), "extent": torch.ones(2),
            "cell_size": torch.tensor(1.), "base_heights": torch.empty(0, 0),
            "base_valid": torch.empty(0, 0, dtype=torch.bool),
            "fallback_height": torch.tensor(0.),
            "lidar_train": torch.empty(0, 3), "lidar_holdout": torch.empty(0, 3),
        }.items():
            self.register_buffer(name, value)
        self.max_delta = float(cfg.get("max_height_delta", 0.5))
        if self.max_delta <= 0:
            raise ValueError("max_height_delta must be positive")
        self.learn_height = cfg.get("learn_height", True)
        if not self.learn_height:
            self.encoding.requires_grad_(False)
            self.geometry.requires_grad_(False)
        self.register_buffer("last_ray_fraction", torch.tensor(0.), persistent=False)
        self.register_buffer("last_opacity", torch.tensor(0.), persistent=False)

    @property
    def initialized(self):
        return self.base_heights.numel() > 0

    @torch.no_grad()
    def initialize(self, road_points, lidar_points):
        self.origin, self.base_heights, self.base_valid, size = build_road_surface_grid(
            road_points, self.cfg.get("base_cell_size", 0.1),
            self.cfg.get("max_grid_cells", 4_000_000))
        self.cell_size.fill_(size)
        self.extent = (road_points[:, :2].amax(0) - self.origin).clamp_min(size)
        self.fallback_height.copy_(road_points[:, 2].median())
        # A local generator does not disturb Gaussian initialization or image sampling.
        generator = torch.Generator(device=lidar_points.device).manual_seed(1729)
        indices = torch.randperm(len(lidar_points), device=lidar_points.device,
                                 generator=generator)[:self.cfg.get("max_lidar_points", 50000)]
        points = lidar_points.detach()[indices]
        n_holdout = int(len(points) * self.cfg.get("holdout_fraction", 0.1))
        self.lidar_holdout = points[:n_holdout].clone()
        self.lidar_train = points[n_holdout:].clone()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Buffers depend on scene extent and LiDAR count, unlike network parameters.
        for name in ("base_heights", "base_valid", "lidar_train", "lidar_holdout"):
            if prefix + name in state_dict:
                setattr(self, name, torch.empty_like(state_dict[prefix + name], device=self.origin.device))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def normalized_xy(self, xy):
        return (xy - self.origin) / self.extent

    def base_height(self, xy):
        """Masked bilinear interpolation; never connect across unsupported grid holes."""
        if not self.initialized:
            raise RuntimeError("RoadSurfaceField must be initialized before querying")
        p = (xy - self.origin) / self.cell_size - 0.5
        lower = p.floor().long()
        frac = p - lower
        numerator = torch.zeros_like(xy[..., 0])
        denominator = torch.zeros_like(numerator)
        height, width = self.base_heights.shape
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            ix, iy = lower[..., 0] + dx, lower[..., 1] + dy
            inside = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
            ix, iy = ix.clamp(0, width - 1), iy.clamp(0, height - 1)
            weight = (frac[..., 0] if dx else 1 - frac[..., 0])
            weight = weight * (frac[..., 1] if dy else 1 - frac[..., 1])
            weight = weight * (inside & self.base_valid[iy, ix])
            numerator = numerator + weight * self.base_heights[iy, ix]
            denominator = denominator + weight
        supported = denominator > 1e-6
        base = torch.where(supported, numerator / denominator.clamp_min(1e-6), self.fallback_height)
        return base, supported

    def height(self, xy):
        base, supported = self.base_height(xy)
        prediction = self.geometry(self.encoding(self.normalized_xy(xy)))
        delta = self.max_delta * torch.tanh(prediction[..., 0])
        return base + delta * supported, supported

    def sdf(self, points):
        base, supported = self.base_height(points[..., :2])
        prediction = self.geometry(self.encoding(self.normalized_xy(points[..., :2])))
        height = base + self.max_delta * torch.tanh(prediction[..., 0]) * supported
        # A positive, bounded cos(theta); starts close to one for nearly flat roads.
        slope = 0.1 + 0.9 * torch.sigmoid(prediction[..., 1] + 4.)
        return slope * (points[..., 2] - height), supported

    def colors(self, xy, directions):
        features = self.color_encoding(self.normalized_xy(xy))
        # Low-order directional encoding limits per-view memorization.
        direction_features = torch.cat((directions, directions.sin(), directions.cos()), -1)
        return self.color(torch.cat((features, direction_features), -1))

    def intrusion_confidence(self, points, cfg):
        with torch.no_grad():
            height, supported = self.height(points[:, :2])
            delta = points[:, 2] - height
            transition = max(float(cfg.get("transition", 0.05)), 1e-6)
            return (supported * torch.sigmoid((delta + cfg.get("below_margin", 0.1)) / transition)
                    * torch.sigmoid((cfg.get("above_margin", 0.15) - delta) / transition))

    def render_rays(self, origins, directions):
        """Local-band NeuS-style CDF compositing around a detached base-surface hit.

        Bounds depend only on the initial surface, shared by fixed/learned runs.
        Returned ray indices select only supported downward intersections.
        """
        with torch.no_grad():
            dz = directions[:, 2]
            downward = dz < -self.cfg.get("min_downward_cos", 0.02)
            safe_dz = torch.where(downward, dz, -torch.ones_like(dz))
            t = (self.fallback_height - origins[:, 2]) / safe_dz
            for _ in range(4):
                xy = origins[:, :2] + t[:, None] * directions[:, :2]
                base, _ = self.base_height(xy)
                t = (base - origins[:, 2]) / safe_dz
            xy = origins[:, :2] + t[:, None] * directions[:, :2]
            _, supported = self.base_height(xy)
            band = self.cfg.get("ray_height_band", 1.0)
            if band <= self.max_delta:
                raise ValueError("ray_height_band must exceed max_height_delta")
            half_width = band / safe_dz.abs()
            near, far = self.cfg.get("near", 0.1), self.cfg.get("far", 120.)
            keep = downward & supported & (t > near) & (t < far)
            indices = keep.nonzero(as_tuple=True)[0]
            start = (t[keep] - half_width[keep]).clamp_min(near)
            end = (t[keep] + half_width[keep]).clamp_max(far)
            u = torch.linspace(0, 1, self.cfg.get("samples_per_ray", 48) + 1,
                               device=origins.device, dtype=origins.dtype)
            distances = start[:, None] + (end - start)[:, None] * u
        if not len(indices):
            return {"indices": indices}
        ray_dirs = directions[indices]
        points = origins[indices, None] + distances[..., None] * ray_dirs[:, None]
        sdf, _ = self.sdf(points)
        cdf = torch.sigmoid(sdf * self.cfg.get("inverse_sdf_scale", 40.))
        midpoints = (points[:, :-1] + points[:, 1:]) * 0.5
        _, valid = self.base_height(midpoints[..., :2])
        alpha = ((cdf[:, :-1] - cdf[:, 1:]) / cdf[:, :-1].clamp_min(1e-6)).clamp(0, 1)
        alpha = alpha * valid
        transmittance = torch.cumprod(torch.cat((torch.ones_like(alpha[:, :1]),
                                                1 - alpha), -1), -1)[:, :-1]
        weights = alpha * transmittance
        colors = self.colors(midpoints[..., :2], ray_dirs[:, None].expand_as(midpoints))
        opacity = weights.sum(-1)
        return {"indices": indices, "rgb": (weights[..., None] * colors).sum(-2),
                "opacity": opacity,
                "depth": (weights * (distances[:, :-1] + distances[:, 1:]) * 0.5).sum(-1),
                "regularization_points": midpoints.detach().reshape(-1, 3)}

    def regularization(self, points):
        """Finite differences support CPU/CUDA double backprop without custom kernels."""
        eps = self.cfg.get("finite_difference_eps", 0.05)
        sdf, supported = self.sdf(points)
        derivatives, valid = [], supported
        for axis in range(3):
            offset = torch.zeros_like(points)
            offset[:, axis] = eps
            plus, vp = self.sdf(points + offset)
            minus, vm = self.sdf(points - offset)
            derivatives.append((plus - minus) / (2 * eps))
            valid = valid & vp & vm
        if not valid.any():
            return sdf.sum() * 0
        return (torch.stack(derivatives, -1)[valid].norm(dim=-1) - 1).square().mean()

    def training_losses(self, image_infos, cam, step):
        """Train auxiliary geometry/color using sampled, unoccluded road pixels."""
        if not self.initialized or step >= self.cfg.get("freeze_after", 30000):
            return {}
        losses = {}
        if step % self.cfg.get("step_interval", 1):
            return losses
        self.last_ray_fraction.zero_()
        self.last_opacity.zero_()
        seed = self.cfg.get("sampling_seed", 31415) + int(step)
        generator = torch.Generator(device=self.origin.device).manual_seed(seed)
        if self.learn_height and len(self.lidar_train):
            lidar_generator = torch.Generator(device=self.origin.device).manual_seed(seed + 100000)
            count = min(self.cfg.get("lidar_batch_size", 2048), len(self.lidar_train))
            points = self.lidar_train[torch.randint(len(self.lidar_train), (count,),
                                                   device=self.origin.device, generator=lidar_generator)]
            predicted, valid = self.height(points[:, :2])
            if valid.any():
                losses["road_field_lidar"] = self.cfg.get("lidar_w", 0.1) * F.smooth_l1_loss(
                    predicted[valid], points[valid, 2], beta=self.cfg.get("lidar_huber_beta", 0.05))
        mask = image_infos["road_masks"].bool().clone()
        for key in ("dynamic_masks", "egocar_masks", "sky_masks"):
            if key in image_infos:
                mask &= ~image_infos[key].bool()
        pixels = mask.nonzero()
        if not len(pixels):
            return losses
        count = min(self.cfg.get("rays_per_image", 512), len(pixels))
        pixels = pixels[torch.randperm(len(pixels), device=pixels.device, generator=generator)[:count]]
        # Match gsplat's pixel-center convention and the corrected camera used for GS.
        uv = torch.stack((pixels[:, 1] + 0.5, pixels[:, 0] + 0.5,
                          torch.ones_like(pixels[:, 0])), -1).to(self.origin)
        rotation = cam.camtoworlds[:3, :3].detach()
        directions = F.normalize((uv @ torch.linalg.inv(cam.Ks.detach()).T) @ rotation.T, dim=-1)
        origins = cam.camtoworlds[:3, 3].detach().expand_as(directions)
        rendered = self.render_rays(origins, directions)
        self.last_ray_fraction.fill_(len(rendered["indices"]) / count)
        if "rgb" not in rendered:
            return losses
        self.last_opacity.copy_(rendered["opacity"].detach().mean())
        selected = pixels[rendered["indices"]]
        target = image_infos["pixels"][selected[:, 0], selected[:, 1]].detach()
        losses["road_field_rgb"] = self.cfg.get("rgb_w", 1.) * F.l1_loss(rendered["rgb"], target)
        losses["road_field_coverage"] = self.cfg.get("coverage_w", 0.05) * (
            -rendered["opacity"].clamp_min(1e-6).log()).mean()
        if self.learn_height and self.cfg.get("eikonal_w", 0.01):
            points = rendered["regularization_points"]
            count = min(self.cfg.get("eikonal_points", 1024), len(points))
            points = points[torch.randperm(len(points), device=points.device, generator=generator)[:count]]
            losses["road_field_eikonal"] = self.cfg.get("eikonal_w", 0.01) * self.regularization(points)
        return losses

    @torch.no_grad()
    def metrics(self):
        result = {"road_field_ray_fraction": self.last_ray_fraction,
                  "road_field_opacity": self.last_opacity}
        if len(self.lidar_holdout):
            points = self.lidar_holdout
            height, valid = self.height(points[:, :2])
            base, _ = self.base_height(points[:, :2])
            if valid.any():
                result["road_field_holdout_mae"] = (height[valid] - points[valid, 2]).abs().mean()
                result["road_field_initial_holdout_mae"] = (base[valid] - points[valid, 2]).abs().mean()
                result["road_field_height_delta_abs"] = (height[valid] - base[valid]).abs().mean()
        return result
