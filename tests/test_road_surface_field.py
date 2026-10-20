import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

from models.road_surface import RoadSurfaceField, align_road_quaternions
from utils.config import load_config


def make_field(**overrides):
    config = dict(levels=2, features=2, table_size=64, min_resolution=4,
                  max_resolution=8, hidden_dim=16, base_cell_size=0.25,
                  max_height_delta=0.5, samples_per_ray=64, ray_height_band=0.8,
                  rays_per_image=16, lidar_batch_size=16, eikonal_points=16,
                  freeze_after=100, inverse_sdf_scale=40.)
    config.update(overrides)
    field = RoadSurfaceField(config)
    axis = torch.arange(-2., 2.01, 0.125)
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack((x.flatten(), y.flatten(), torch.zeros(x.numel())), -1)
    field.initialize(points, points)
    return field


class RoadSurfaceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        torch.set_num_threads(1)

    def test_initialization_support_and_bounded_residual(self):
        field = make_field()
        xy = torch.tensor([[0.1, 0.1], [100., 100.]])
        height, valid = field.height(xy)
        self.assertTrue(torch.equal(valid, torch.tensor([True, False])))
        self.assertTrue(torch.equal(height, torch.zeros(2)))
        with torch.no_grad():
            field.geometry[-1].bias[0] = 100
        moved, _ = field.height(xy)
        self.assertAlmostEqual(moved[0].item(), 0.5)
        self.assertEqual(moved[1].item(), 0.)

    def test_surface_normals_for_plane_and_frozen_geometry(self):
        for learn_height in (True, False):
            field = make_field(learn_height=learn_height)
            rows, cols = field.base_heights.shape
            y, x = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
            xy_grid = field.origin + (torch.stack((x, y), -1) + 0.5) * field.cell_size
            field.base_heights.copy_(0.2 * xy_grid[..., 0] - 0.3 * xy_grid[..., 1])
            query = torch.tensor([[0.13, 0.22], [0.51, -0.41]], requires_grad=True)
            with torch.no_grad():
                height, normals, supported = field.height_and_normal(query)
            expected = F.normalize(torch.tensor([-0.2, 0.3, 1.]), dim=0)
            self.assertTrue(supported.all())
            torch.testing.assert_close(normals, expected.expand_as(normals))
            torch.testing.assert_close(height, 0.2 * query[:, 0] - 0.3 * query[:, 1])
            self.assertFalse(height.requires_grad or normals.requires_grad)
            self.assertIsNone(query.grad)
            self.assertTrue(all(p.grad is None for p in field.parameters()))

    def test_normals_equal_learned_sdf_zero_surface_gradient(self):
        field = make_field()
        with torch.no_grad():
            field.geometry[-1].weight.normal_(std=0.2)
        xy = torch.tensor([[0.13, 0.22], [0.51, -0.41], [100., 100.]])
        height, normals, supported = field.height_and_normal(xy)
        self.assertEqual(supported.tolist(), [True, True, False])
        points = torch.cat((xy[:2], height[:2, None]), -1).requires_grad_(True)
        sdf, _ = field.sdf(points)
        gradient, = torch.autograd.grad(sdf.sum(), points)
        torch.testing.assert_close(normals[:2], F.normalize(gradient, dim=-1))
        self.assertTrue(all(p.grad is None for p in field.parameters()))

    def test_hash_feature_jacobian_matches_autograd(self):
        field = make_field()
        xy = torch.tensor([[0.13, 0.22], [-0.15, 1.14]], requires_grad=True)
        with torch.no_grad():
            field.encoding.tables.uniform_(-0.3, 0.3)
        features, jacobian = field.encoding.value_and_jacobian(xy)
        reference = field.encoding(xy)
        torch.testing.assert_close(features, reference)
        for channel in range(reference.shape[-1]):
            derivative, = torch.autograd.grad(reference[:, channel].sum(), xy, retain_graph=True)
            torch.testing.assert_close(jacobian[:, channel], derivative)

    def test_base_gradient_with_holes_edges_and_no_support(self):
        field = make_field()
        with torch.no_grad():
            field.base_heights.normal_(std=0.1)
            field.base_valid[7:10, 8:11] = False
        xy = torch.cat((torch.rand(64, 2) * 4.6 - 2.3, torch.tensor([[100., 100.]])))
        xy.requires_grad_(True)
        reference, supported = field.base_height(xy)
        expected, = torch.autograd.grad(reference.sum(), xy)
        value, valid, gradient = field.base_height(xy.detach(), with_gradient=True)
        torch.testing.assert_close(value, reference)
        self.assertTrue(torch.equal(valid, supported))
        torch.testing.assert_close(gradient, expected, atol=2e-6, rtol=2e-5)

    def test_analytic_normals_match_autograd_after_nontrivial_geometry_updates(self):
        field = make_field()
        with torch.no_grad():
            field.encoding.tables.uniform_(-0.2, 0.2)
            field.geometry[-1].weight.normal_(std=0.3)
            field.base_heights.normal_(std=0.02)
        for saturation in (False, True):
            if saturation:
                with torch.no_grad():
                    field.geometry[0].bias.fill_(25.)
            xy = (torch.rand(32, 2) * 2 - 1).requires_grad_(True)
            expected_height, _ = field.height(xy)
            gradient, = torch.autograd.grad(expected_height.sum(), xy)
            expected_normal = F.normalize(torch.cat((-gradient, torch.ones(32, 1)), -1), dim=-1)
            height, normal, _ = field.height_and_normal(xy)
            torch.testing.assert_close(height, expected_height)
            torch.testing.assert_close(normal, expected_normal, atol=2e-6, rtol=2e-5)

    def test_orientation_alignment_preserves_yaw_without_accumulating_tilt(self):
        yaw = torch.tensor([0., 0.7, -2.5, 3.14159265])
        original = torch.zeros(4, 4)
        original[:, 0], original[:, 3] = (yaw / 2).cos(), (yaw / 2).sin()
        normals = F.normalize(torch.tensor([[-0.2, 0.3, 1.]]).expand(4, -1), dim=-1)
        aligned = align_road_quaternions(original, normals)
        w, x, y, z = aligned.unbind(-1)
        local_z = torch.stack((2 * (x*z + w*y), 2 * (y*z - w*x),
                               1 - 2 * (x*x + y*y)), -1)
        torch.testing.assert_close(local_z, normals)
        torch.testing.assert_close(aligned.norm(dim=-1), torch.ones(4))
        torch.testing.assert_close(align_road_quaternions(aligned, normals), aligned)
        flat = normals.new_tensor([0., 0., 1.]).expand_as(normals)
        torch.testing.assert_close(align_road_quaternions(aligned, flat), original)

    def test_background_tracks_surface_without_geometry_gradient(self):
        field = make_field()
        points = torch.tensor([[0., 0., 0.], [100., 100., 0.]], requires_grad=True)
        original = field.intrusion_confidence(points, {})
        with torch.no_grad():
            field.geometry[-1].bias[0] = 2
        shifted = field.intrusion_confidence(points, {})
        self.assertGreater(original[0].item(), 0.8)
        self.assertLess(shifted[0].item(), 0.01)
        self.assertEqual(original[1].item(), 0)
        self.assertFalse(shifted.requires_grad)

    def test_sdf_ray_rendering_depth_and_invalid_rays(self):
        field = make_field()
        origins = torch.tensor([[0., 0., 2.], [0., 0., 2.], [20., 20., 2.]])
        directions = torch.tensor([[0., 0., -1.], [0., 0., 1.], [0., 0., -1.]])
        rendered = field.render_rays(origins, directions)
        self.assertEqual(rendered["indices"].tolist(), [0])
        self.assertGreater(rendered["opacity"].item(), 0.99)
        self.assertAlmostEqual(rendered["depth"].item(), 2., delta=0.03)
        with torch.no_grad():
            field.geometry[-1].bias[0] = torch.atanh(torch.tensor(0.4))
        moved = field.render_rays(origins, directions)
        self.assertAlmostEqual(moved["depth"].item(), 1.8, delta=0.03)

    def test_image_reconstruction_updates_geometry_and_color(self):
        field = make_field(eikonal_w=0.01)
        # Camera looking down at the road, deliberately shifted target colors.
        c2w = torch.eye(4)
        c2w[:3, :3] = torch.diag(torch.tensor([1., -1., -1.]))
        c2w[2, 3] = 2.
        camera = SimpleNamespace(camtoworlds=c2w, Ks=torch.tensor([
            [8., 0., 2.], [0., 8., 2.], [0., 0., 1.]]))
        image = dict(road_masks=torch.ones(4, 4, dtype=torch.bool),
                     pixels=torch.zeros(4, 4, 3))
        losses = field.training_losses(image, camera, step=1)
        self.assertIn("road_field_rgb", losses)
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        # Isolate RGB: no LiDAR/regularizer may provide the geometry gradient here.
        losses["road_field_rgb"].backward()
        self.assertGreater(field.geometry[-1].bias.grad.abs().sum().item(), 0.)
        self.assertGreater(field.color[-2].weight.grad.abs().sum().item(), 0.)

    def test_photometric_geometry_recovers_synthetic_height(self):
        field = make_field(inverse_sdf_scale=20., samples_per_ray=96)
        # Known world-space road texture, held fixed: color cannot absorb the error.
        field.colors = lambda xy, directions: torch.stack(
            (0.5 + xy[..., 0] * 0.2, 0.5 + xy[..., 1] * 0.2,
             torch.full_like(xy[..., 0], 0.5)), -1)
        origins = torch.tensor([[-0.5, 0., 2.], [0.5, 0., 2.],
                                [0., -0.5, 2.], [0., 0.5, 2.]])
        directions = F.normalize(torch.tensor([[0.4, 0., -1.], [-0.4, 0., -1.],
                                                [0., 0.4, -1.], [0., -0.4, -1.]]), dim=-1)
        with torch.no_grad():
            field.geometry[-1].bias[0] = torch.atanh(torch.tensor(0.4))
            target = field.render_rays(origins, directions)["rgb"].clone()
            field.geometry[-1].bias.zero_()
        # Optimize only the height bias; all observations are photometric.
        optimizer = torch.optim.Adam([field.geometry[-1].bias], lr=0.04)
        before = F.mse_loss(field.render_rays(origins, directions)["rgb"], target).item()
        for _ in range(60):
            optimizer.zero_grad()
            loss = F.mse_loss(field.render_rays(origins, directions)["rgb"], target)
            loss.backward()
            optimizer.step()
        recovered, _ = field.height(torch.zeros(1, 2))
        self.assertAlmostEqual(recovered.item(), 0.2, delta=0.03)
        self.assertLess(loss.item(), before * 0.05)

    def test_fixed_control_baked_height_and_checkpoint_roundtrip(self):
        learned = make_field()
        fixed = make_field(learn_height=False)
        fixed.load_state_dict(copy.deepcopy(learned.state_dict()))
        xy = torch.tensor([[0., 0.], [0.5, 0.5]])
        self.assertTrue(torch.equal(learned.height(xy)[0], fixed.height(xy)[0]))
        self.assertFalse(any(p.requires_grad for p in fixed.geometry.parameters()))
        self.assertTrue(any(p.requires_grad for p in fixed.color.parameters()))
        with torch.no_grad():
            learned.geometry[-1].bias[0] = 0.4
        baked = learned.height(xy)[0].detach()
        self.assertFalse(baked.requires_grad)
        state = copy.deepcopy(learned.state_dict())
        restored = RoadSurfaceField(learned.cfg)
        restored.load_state_dict(state)
        self.assertTrue(torch.equal(baked, restored.height(xy)[0]))
        self.assertTrue(torch.equal(learned.lidar_train, restored.lidar_train))

    def test_fixed_and_learned_sample_identical_image_rays(self):
        learned = make_field()
        fixed = make_field(learn_height=False)
        c2w = torch.eye(4)
        c2w[:3, :3] = torch.diag(torch.tensor([1., -1., -1.]))
        c2w[2, 3] = 2.
        camera = SimpleNamespace(camtoworlds=c2w, Ks=torch.tensor([
            [8., 0., 3.], [0., 8., 3.], [0., 0., 1.]]))
        image = dict(road_masks=torch.ones(6, 6, dtype=torch.bool),
                     pixels=torch.zeros(6, 6, 3))
        rays = []
        for field in (learned, fixed):
            original = field.render_rays

            def capture(origins, directions, renderer=original):
                rays.append(directions.clone())
                return renderer(origins, directions)

            field.render_rays = capture
            field.training_losses(image, camera, step=1)
        self.assertTrue(torch.equal(rays[0], rays[1]))

    def test_auxiliary_color_features_do_not_directly_update_geometry(self):
        field = make_field()
        field.colors(torch.zeros(2, 2), torch.tensor([[0., 0., -1.]]).expand(2, -1)).sum().backward()
        self.assertIsNone(field.encoding.tables.grad)
        self.assertTrue(all(p.grad is None for p in field.geometry.parameters()))
        self.assertIsNotNone(field.color_encoding.tables.grad)

    def test_masks_freeze_and_sampling_do_not_change_global_rng(self):
        field = make_field()
        camera = SimpleNamespace(camtoworlds=torch.eye(4), Ks=torch.eye(3))
        image = dict(road_masks=torch.zeros(2, 2, dtype=torch.bool),
                     pixels=torch.zeros(2, 2, 3))
        rng = torch.random.get_rng_state().clone()
        losses = field.training_losses(image, camera, step=1)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertNotIn("road_field_rgb", losses)
        self.assertEqual(field.training_losses(image, camera, step=100), {})

    def test_eikonal_backward_and_lidar_optimization(self):
        field = make_field()
        points = torch.tensor([[0., 0., 0.], [0.5, 0.5, 0.2]])
        loss = field.regularization(points)
        loss.backward()
        self.assertTrue(torch.isfinite(field.geometry[-1].bias.grad).all())
        with torch.no_grad():
            field.geometry[-1].bias[0] = 1.
        optimizer = torch.optim.Adam(field.parameters(), lr=0.02)
        for _ in range(30):
            optimizer.zero_grad()
            height, _ = field.height(points[:, :2])
            loss = height.square().mean()
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), 0.005)


class ConfigTests(unittest.TestCase):
    def test_experiment_overlays_preserve_baseline(self):
        root = Path(__file__).resolve().parents[1]
        learned = load_config(root / "configs/road_height_learned.yaml")
        fixed = load_config(root / "configs/road_height_fixed.yaml")
        baseline = load_config(root / "configs/omnire_extended_cam_lidar.yaml")
        self.assertEqual(learned.model.Background, baseline.model.Background)
        self.assertEqual(learned.trainer, fixed.trainer)
        self.assertFalse(fixed.model.RoadNodes.networks.road_surface.learn_height)
        self.assertTrue(learned.model.RoadNodes.networks.road_surface.learn_height)
        self.assertNotIn("_base_", learned)

    def test_config_inheritance_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("_base_: config.yaml\n")
            with self.assertRaisesRegex(ValueError, "Cyclic"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
