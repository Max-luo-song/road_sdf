import unittest

import torch

from utils.road_supervision import compute_road_supervision_losses, unpremultiply_road_rgb


class RoadSupervisionTests(unittest.TestCase):
    def test_color_and_coverage_reach_road_gaussians(self):
        color = torch.full((5, 5, 3), 0.25, requires_grad=True)
        opacity = torch.full((5, 5, 1), 0.5, requires_grad=True)
        outputs = {
            "RoadNodes_color": unpremultiply_road_rgb(color * opacity, opacity),
            "RoadNodes_opacity": opacity,
            "RoadNodes_contribution": opacity * 0.5,
            "opacity": torch.ones_like(opacity),
        }
        image = {
            "pixels": torch.ones(5, 5, 3),
            "road_masks": torch.ones(5, 5, dtype=torch.bool),
        }
        config = {
            "road_color": {"w": 1, "erosion_px": 0},
            "road_coverage": {"w": 1, "erosion_px": 0, "target_alpha": 0.9},
            "road_contribution": {"w": 1, "erosion_px": 0, "target_share": 0.9},
        }
        losses = compute_road_supervision_losses(outputs, image, config, 0)
        self.assertEqual(set(losses), {
            "road_color_loss", "road_coverage_loss", "road_contribution_loss"})
        sum(losses.values()).backward()
        self.assertIsNotNone(color.grad)
        self.assertGreater(color.grad.abs().sum().item(), 0)
        self.assertIsNotNone(opacity.grad)
        self.assertGreater(opacity.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
