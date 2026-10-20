import unittest
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

from utils.road_render_diagnostics import (
    depth_edges, expected_depth_normals, normalize_rgb, render_layers, roi_mask, summarize,
)


@dataclass
class Gaussians:
    _means: torch.Tensor
    _quats: torch.Tensor
    _scales: torch.Tensor
    _opacities: torch.Tensor
    _rgbs: torch.Tensor

    @property
    def rgbs(self):
        return self._rgbs


class OrderedCompositor:
    """Three ordered splats at one pixel, with exact front-to-back compositing."""

    gaussian_classes = {"RoadNodes": 5, "Background": 0, "RigidNodes": 1}
    render_cfg = SimpleNamespace(near_plane=.1, far_plane=120., get=lambda *args: 0.)

    def __init__(self, labels):
        self.pts_labels = torch.tensor(labels)

    def render_gaussians(self, gs, cam, **kwargs):
        alpha = gs._opacities.flatten()
        transmittance = torch.cumprod(torch.cat((torch.ones(1), 1 - alpha)), 0)[:-1]
        weights = alpha * transmittance
        rgb = (weights[:, None] * gs._rgbs).sum(0).reshape(1, 1, 3)
        opacity = weights.sum().reshape(1, 1, 1)
        depth = ((weights * gs._means[:, 2]).sum() / opacity.clamp_min(1e-8)).reshape(1, 1, 1)
        return dict(rgb_gaussians=rgb, opacity=opacity, depth=depth), None


class RoadRenderDiagnosticsTests(unittest.TestCase):
    def test_visible_contributions_keep_occlusion_and_removal_changes_it(self):
        gs = Gaussians(torch.tensor([[0., 0., 1.], [0., 0., 2.], [0., 0., 3.]]),
                       torch.zeros(3, 4), torch.ones(3, 3), torch.full((3, 1), .5),
                       torch.eye(3))
        trainer = OrderedCompositor([5, 0, 1])
        full, _ = trainer.render_gaussians(gs, None)
        layers, contribution, counts = render_layers(trainer, gs, None, full)
        torch.testing.assert_close(contribution.flatten(), torch.tensor([.5, .25, .125]))
        torch.testing.assert_close(contribution.sum(-1, keepdim=True), full["opacity"])
        self.assertEqual(counts, dict(road=1, background=1, other=1))
        # Background isolated alpha is .5, but its visible contribution is only .25.
        self.assertAlmostEqual(float(layers["background"]["opacity"]), .5)
        torch.testing.assert_close(layers["no_background"]["rgb_gaussians"].flatten(),
                                   torch.tensor([.5, 0., .25]))
        torch.testing.assert_close(gs._rgbs, torch.eye(3))
        # Missing Background produces zero isolated maps and leaves other layers intact.
        trainer.pts_labels[:] = 5
        layers, _, counts = render_layers(trainer, gs, None, full)
        self.assertEqual(counts["background"], 0)
        self.assertEqual(float(layers["background"]["opacity"]), 0.)

    def test_normalization_distinguishes_dark_color_from_low_coverage(self):
        rgb = np.array([[[.02, .04, .06], [.001, .001, .001]]], dtype=np.float32)
        color, valid = normalize_rgb(rgb, np.array([[.1, .01]]))
        np.testing.assert_allclose(color[0, 0], [.2, .4, .6], atol=1e-6)
        self.assertEqual(valid.tolist(), [[True, False]])
        np.testing.assert_array_equal(color[0, 1], 0)

    def test_depth_discontinuities_require_valid_neighbors(self):
        depth = np.array([[1., 1., 10.], [1., 1., 10.]], dtype=np.float32)
        valid = np.ones_like(depth, bool)
        valid[:, 2] = False
        np.testing.assert_array_equal(depth_edges(depth, valid), 0)
        self.assertGreater(depth_edges(depth, np.ones_like(valid)).max(), 1)

    def test_expected_depth_normals_for_plane(self):
        depth = np.ones((5, 6), dtype=np.float32) * 2
        normals, valid = expected_depth_normals(depth, np.ones_like(depth, bool), np.eye(3))
        np.testing.assert_allclose(normals[valid], np.tile([0., 0., -1.], (12, 1)))
        self.assertFalse(valid[0].any())
        self.assertFalse(valid[:, -1].any())

    def test_empty_regions_and_other_occluders_do_not_make_claims(self):
        shape = (2, 2)
        layer = dict(rgb_gaussians=np.full((*shape, 3), .02, dtype=np.float32),
                     opacity=np.full((*shape, 1), .5, dtype=np.float32),
                     depth=np.ones((*shape, 1), dtype=np.float32))
        contribution = np.zeros((*shape, 3), dtype=np.float32)
        contribution[..., 2] = .5  # All road splats are occluded by another class.
        from tools.diagnose_road_render import THRESHOLDS
        stats = summarize({"road": layer, "full": layer}, contribution,
                          layer["rgb_gaussians"], layer["rgb_gaussians"],
                          roi_mask(shape, [0, .5, 1, 1]), THRESHOLDS)
        self.assertEqual(stats["supported_road_pixels"], 0)
        self.assertIsNone(stats["dark_road_weak_coverage_fraction"])
        self.assertEqual(stats["contribution_alpha_max_abs_error"], 0)
        with self.assertRaises(ValueError):
            roi_mask(shape, [0, 0, 2, 1])


if __name__ == "__main__":
    unittest.main()
