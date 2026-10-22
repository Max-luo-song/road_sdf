"""ExtraGS RSG node initialized directly from the processed road point cloud.

The paper does not specify Gaussian seeding density. XY voxel averaging is the
only dataset adapter here; no legacy staggered grid, lane samples, or KNN color
interpolation is used.
"""

import logging

import torch
from torch.nn import Parameter

from models.gaussians.basics import RGB2SH, num_sh_bases
from models.nodes.road import RoadNodes


logger = logging.getLogger(__name__)


class ExtraGSRoadNodes(RoadNodes):
    @torch.no_grad()
    def create_from_pcd(self, init_means, init_colors, lane_samples=None):
        if init_means.numel() == 0 or init_colors.numel() == 0:
            raise ValueError("ExtraGS RSG needs processed road points and colors")
        if not self.appearance_only or not self.surface_field.cfg.get("paper_rsg", False):
            raise ValueError("ExtraGSRoadNodes requires the paper RSG field")
        cell_size = float(self.ctrl_cfg.get("rsg_init_cell_size", 0.1))
        tangent_scale = float(self.ctrl_cfg.get("rsg_tangent_scale", 0.07))
        if cell_size <= 0 or tangent_scale <= 0:
            raise ValueError("RSG initialization cell and scale must be positive")

        source_means = init_means.to(self.device)
        source_colors = init_colors.to(self.device).clamp(1e-6, 1 - 1e-6)
        cells = torch.floor(source_means[:, :2] / cell_size).long()
        _, inverse = torch.unique(cells, dim=0, return_inverse=True)
        count = int(inverse.max().item()) + 1
        weights = source_means.new_zeros(count)
        weights.index_add_(0, inverse, torch.ones_like(source_means[:, 0]))
        means = source_means.new_zeros(count, 3)
        colors = source_colors.new_zeros(count, 3)
        means.index_add_(0, inverse, source_means)
        colors.index_add_(0, inverse, source_colors)
        means /= weights[:, None]
        colors /= weights[:, None]

        self._means = Parameter(means, requires_grad=False)
        self._scales = Parameter(
            means.new_full((count, 2), tangent_scale).log(), requires_grad=False)
        quats = means.new_zeros(count, 4)
        quats[:, 0] = 1.
        self._quats = Parameter(quats, requires_grad=False)
        opacity = float(self.surface_field.cfg["initial_opacity"])
        self._opacities = Parameter(
            torch.logit(means.new_full((count, 1), opacity)), requires_grad=False)
        sh = means.new_zeros(count, num_sh_bases(self.sh_degree), 3)
        if self.sh_degree > 0:
            sh[:, 0] = RGB2SH(colors)
        else:
            sh[:, 0] = torch.logit(colors)
        self._features_dc = Parameter(sh[:, 0])
        self._features_rest = Parameter(sh[:, 1:])
        self.surface_field.initialize(means.detach(), source_means.detach())
        logger.info("ExtraGS RSG initialized %d Gaussians from %d processed road points",
                    count, len(source_means))
        return False
