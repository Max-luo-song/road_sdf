"""ExtraGS §3.2 road SDF, adapted to the existing QCraft camera and mask inputs.

The paper specifies the field and losses but not its ray sampling schedule. We
sample a narrow band around the current height intersection for stable NeuS CDF
integration on the existing rasterization pipeline.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from models.road_surface import HashGrid2D


class ExtraGSRoadField(nn.Module):
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
        self.elevation = nn.Sequential(
            nn.Linear(self.encoding.out_dim, width), nn.Softplus(), nn.Linear(width, 1))
        self.slope = nn.Sequential(
            nn.Linear(self.encoding.out_dim, width), nn.Softplus(), nn.Linear(width, 1))
        nn.init.zeros_(self.elevation[-1].weight)
        nn.init.zeros_(self.elevation[-1].bias)
        nn.init.zeros_(self.slope[-1].weight)
        nn.init.constant_(self.slope[-1].bias, 4.)
        self.direction_frequencies = int(cfg.get("direction_frequencies", 4))
        color_input = self.encoding.out_dim + 3 + 6 * self.direction_frequencies
        self.color = nn.Sequential(
            nn.Linear(color_input, width), nn.ReLU(), nn.Linear(width, width),
            nn.ReLU(), nn.Linear(width, 3), nn.Sigmoid())
        self.register_buffer("origin", torch.zeros(2))
        self.register_buffer("extent", torch.ones(2))
        self.register_buffer("reference_height", torch.zeros(()))
        self.register_buffer("ready", torch.tensor(False))
        self.register_buffer("last_ray_fraction", torch.tensor(0.), persistent=False)
        self.register_buffer("last_opacity", torch.tensor(0.), persistent=False)

    @property
    def initialized(self):
        return bool(self.ready)

    @torch.no_grad()
    def initialize(self, road_points, lidar_points=None):
        if len(road_points) == 0:
            raise ValueError("ExtraGS road field needs road initialization points")
        minimum = road_points[:, :2].amin(0)
        maximum = road_points[:, :2].amax(0)
        self.origin.copy_(minimum)
        self.extent.copy_((maximum - minimum).clamp_min(1e-3))
        median = road_points[:, 2].median()
        self.reference_height.copy_(median)
        self.elevation[-1].bias.fill_(median)
        self.ready.fill_(True)

    def normalized_xy(self, xy):
        return (xy - self.origin) / self.extent

    def supported(self, xy):
        margin = float(self.cfg.get("support_margin", 3.0))
        return ((xy >= self.origin - margin) &
                (xy <= self.origin + self.extent + margin)).all(-1)

    def height(self, xy):
        value = self.elevation(self.encoding(self.normalized_xy(xy)))[..., 0]
        return value, self.supported(xy)

    @torch.no_grad()
    def height_and_normal(self, xy):
        features, jacobian = self.encoding.value_and_jacobian(self.normalized_xy(xy))
        linear, activation, output = self.elevation
        hidden = linear(features)
        height = output(activation(hidden))[..., 0]
        derivative = torch.where(hidden * activation.beta > activation.threshold,
                                 1., torch.sigmoid(hidden * activation.beta))
        feature_gradient = (derivative * output.weight[0]) @ linear.weight
        gradient = (feature_gradient[..., None] * jacobian).sum(-2) / self.extent
        normal = F.normalize(torch.cat((-gradient, torch.ones_like(height[..., None])), -1), dim=-1)
        return height, normal, self.supported(xy)

    def geometry_version(self):
        values = (self.origin, self.extent, *self.encoding.parameters(),
                  *self.elevation.parameters())
        return tuple((id(value), value._version) for value in values)

    def sdf(self, points):
        xy = points[..., :2]
        features = self.encoding(self.normalized_xy(xy))
        height = self.elevation(features)[..., 0]
        cosine = torch.sigmoid(self.slope(features)[..., 0])
        return cosine * (points[..., 2] - height), self.supported(xy)

    def colors(self, xy, directions):
        features = self.encoding(self.normalized_xy(xy))
        encoded = [directions]
        for level in range(self.direction_frequencies):
            angle = (2 ** level) * math.pi * directions
            encoded.extend((angle.sin(), angle.cos()))
        return self.color(torch.cat((features, *encoded), -1))

    def intrusion_confidence(self, points, cfg):
        with torch.no_grad():
            height, supported = self.height(points[:, :2])
            delta = points[:, 2] - height
            transition = max(float(cfg.get("transition", 0.05)), 1e-6)
            return (supported * torch.sigmoid((delta + cfg.get("below_margin", 0.1)) / transition)
                    * torch.sigmoid((cfg.get("above_margin", 0.15) - delta) / transition))

    def render_rays(self, origins, directions):
        with torch.no_grad():
            dz = directions[:, 2]
            downward = dz < -self.cfg.get("min_downward_cos", 0.02)
            safe_dz = torch.where(downward, dz, -torch.ones_like(dz))
            t = (self.reference_height - origins[:, 2]) / safe_dz
            for _ in range(4):
                xy = origins[:, :2] + t[:, None] * directions[:, :2]
                height, _ = self.height(xy)
                t = (height - origins[:, 2]) / safe_dz
            xy = origins[:, :2] + t[:, None] * directions[:, :2]
            near, far = self.cfg.get("near", 0.1), self.cfg.get("far", 120.)
            keep = downward & self.supported(xy) & (t > near) & (t < far)
            indices = keep.nonzero(as_tuple=True)[0]
            if len(indices) == 0:
                return {"indices": indices}
            band = float(self.cfg.get("ray_height_band", 1.0))
            half_width = band / safe_dz[keep].abs()
            start = (t[keep] - half_width).clamp_min(near)
            end = (t[keep] + half_width).clamp_max(far)
            u = torch.linspace(0, 1, self.cfg.get("samples_per_ray", 48) + 1,
                               device=origins.device, dtype=origins.dtype)
            distances = start[:, None] + (end - start)[:, None] * u
        ray_dirs = directions[indices]
        points = origins[indices, None] + distances[..., None] * ray_dirs[:, None]
        sdf, _ = self.sdf(points)
        cdf = torch.sigmoid(sdf * self.cfg.get("inverse_sdf_scale", 40.))
        midpoint = (points[:, :-1] + points[:, 1:]) * 0.5
        valid = self.supported(midpoint[..., :2])
        alpha = ((cdf[:, :-1] - cdf[:, 1:]) / cdf[:, :-1].clamp_min(1e-6)).clamp(0, 1)
        alpha = alpha * valid
        transmittance = torch.cumprod(torch.cat((torch.ones_like(alpha[:, :1]),
                                                1 - alpha), -1), -1)[:, :-1]
        weights = alpha * transmittance
        rgb = self.colors(midpoint[..., :2], ray_dirs[:, None].expand_as(midpoint))
        return {"indices": indices, "rgb": (weights[..., None] * rgb).sum(-2),
                "opacity": weights.sum(-1), "regularization_points": midpoint.detach().reshape(-1, 3)}

    def regularization(self, points):
        points = points.detach().requires_grad_(True)
        sdf, supported = self.sdf(points)
        if not supported.any():
            return sdf.sum() * 0
        gradient, = torch.autograd.grad(sdf.sum(), points, create_graph=True)
        return (gradient[supported].norm(dim=-1) - 1).square().mean()

    def training_losses(self, image_infos, cam, step):
        if not self.initialized or step % self.cfg.get("step_interval", 1):
            return {}
        self.last_ray_fraction.zero_()
        self.last_opacity.zero_()
        mask = image_infos["road_masks"].bool().clone()
        for key in ("dynamic_masks", "egocar_masks", "sky_masks"):
            if key in image_infos:
                mask &= ~image_infos[key].bool()
        pixels = mask.nonzero()
        if len(pixels) == 0:
            return {}
        generator = torch.Generator(device=pixels.device).manual_seed(
            self.cfg.get("sampling_seed", 31415) + int(step))
        count = min(self.cfg.get("rays_per_image", 512), len(pixels))
        pixels = pixels[torch.randperm(len(pixels), device=pixels.device, generator=generator)[:count]]
        uv = torch.stack((pixels[:, 1] + 0.5, pixels[:, 0] + 0.5,
                          torch.ones_like(pixels[:, 0])), -1).to(self.origin)
        rotation = cam.camtoworlds[:3, :3].detach()
        directions = F.normalize((uv @ torch.linalg.inv(cam.Ks.detach()).T) @ rotation.T, dim=-1)
        origins = cam.camtoworlds[:3, 3].detach().expand_as(directions)
        rendered = self.render_rays(origins, directions)
        self.last_ray_fraction.fill_(len(rendered["indices"]) / count)
        if "rgb" not in rendered:
            return {}
        self.last_opacity.copy_(rendered["opacity"].detach().mean())
        selected = pixels[rendered["indices"]]
        target = image_infos["pixels"][selected[:, 0], selected[:, 1]].detach()
        losses = {"road_field_rgb": self.cfg.get("rgb_w", 1.) * F.l1_loss(rendered["rgb"], target)}
        regularization_points = rendered["regularization_points"]
        maximum = int(self.cfg.get("eikonal_points", 0))
        if maximum > 0 and len(regularization_points) > maximum:
            sample = torch.randperm(len(regularization_points), device=regularization_points.device,
                                    generator=generator)[:maximum]
            regularization_points = regularization_points[sample]
        losses["road_field_eikonal"] = self.cfg.get("eikonal_w", 0.01) * self.regularization(
            regularization_points)
        return losses

    @torch.no_grad()
    def metrics(self):
        return {"road_field_ray_fraction": self.last_ray_fraction,
                "road_field_opacity": self.last_opacity}
