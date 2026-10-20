"""Exercise the actual trainer class dispatch without optional CUDA/data imports.

Only BasicTrainer's rasterizer-dependent losses are replaced. All MultiTrainer
methods, including duplicate-method resolution, and RoadNodes height sync are
compiled unchanged from their source classes, rather than copied into a fixture.
"""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch
from torch import nn

from test_road_surface_field import make_field


ROOT = Path(__file__).resolve().parents[1]


class RasterizerFreeBase(nn.Module):
    def compute_losses(self, outputs, image_infos, cam_infos, gs):
        return {"rgb_loss": outputs["rgb"].square().mean()}

    def compute_metrics(self, outputs, image_infos):
        return {"psnr": outputs["rgb"].mean()}

    def optimizer_step(self):
        self.optimizer.step()


def load_source_class(path, name, **symbols):
    module = ast.parse(path.read_text(encoding="utf-8-sig"))
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == name)
    namespace = dict(torch=torch, nn=nn, Parameter=nn.Parameter,
                     Dict=Dict, List=List, Optional=Optional, Tuple=Tuple,
                     OmegaConf=OmegaConf, DrivingDataset=object,
                     dataclass_camera=SimpleNamespace, logger=logging.getLogger(__name__))
    namespace.update(symbols)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class RoadTrainerIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        torch.set_num_threads(1)
        self.trainer_class = load_source_class(
            ROOT / "models/trainers/scene_graph.py", "MultiTrainer", BasicTrainer=RasterizerFreeBase)
        self.road_class = load_source_class(ROOT / "models/nodes/road.py", "RoadNodes")

    def fixture(self, enabled=True):
        trainer = self.trainer_class.__new__(self.trainer_class)
        nn.Module.__init__(trainer)
        road = self.road_class.__new__(self.road_class)
        nn.Module.__init__(road)
        road.surface_field = make_field(apply_after=0) if enabled else None
        road._means = nn.Parameter(torch.tensor([[0., 0., 0.], [0.5, 0.5, 0.]]),
                                   requires_grad=False)
        road._quats = nn.Parameter(torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
                                   requires_grad=False)
        road.step = trainer.step = 1
        trainer.models = {"RoadNodes": road}
        camera_pose = torch.eye(4)
        camera_pose[:3, :3] = torch.diag(torch.tensor([1., -1., -1.]))
        camera_pose[2, 3] = 2.
        trainer._road_field_camera = SimpleNamespace(
            camtoworlds=camera_pose,
            Ks=torch.tensor([[8., 0., 2.], [0., 8., 2.], [0., 0., 1.]]))
        image = dict(road_masks=torch.ones(4, 4, dtype=torch.bool), pixels=torch.zeros(4, 4, 3))
        outputs = {"rgb": nn.Parameter(torch.full((4, 4, 3), 0.5))}
        return trainer, road, image, outputs

    def test_actual_loss_dispatch_backward_optimizer_and_baked_height(self):
        trainer, road, image, outputs = self.fixture()
        field = road.surface_field
        # Deliberately inconsistent initial height and independent anchor observation.
        field.lidar_train[:, 2] = 0.15
        before_geometry = field.geometry[-1].weight.detach().clone()
        before_color = field.color[-2].weight.detach().clone()
        trainer.optimizer = torch.optim.Adam(list(field.parameters()) + [outputs["rgb"]], lr=0.001)
        losses = trainer.compute_losses(outputs, image, {}, None)
        self.assertTrue({"rgb_loss", "road_field_rgb", "road_field_lidar",
                         "road_field_coverage", "road_field_eikonal"}.issubset(losses))
        sum(losses.values()).backward()
        for parameter in (field.geometry[-1].weight, field.color[-2].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0.)
        trainer.optimizer_step()
        self.assertFalse(torch.equal(before_geometry, field.geometry[-1].weight))
        self.assertFalse(torch.equal(before_color, field.color[-2].weight))
        self.assertGreater(road._means[:, 2].abs().sum().item(), 0.)
        self.assertTrue(torch.equal(road._means[:, 2], field.height(road._means[:, :2])[0].detach()))
        metrics = trainer.compute_metrics(outputs, image)
        self.assertIn("road_field_height_delta_abs", metrics)
        self.assertIn("road_field_ray_fraction", metrics)
        self.assertGreater(metrics["road_field_ray_fraction"].item(), 0.)

    def test_disabled_field_preserves_base_losses_and_metrics(self):
        trainer, road, image, outputs = self.fixture(enabled=False)
        self.assertEqual(set(trainer.compute_losses(outputs, image, {}, None)), {"rgb_loss"})
        self.assertEqual(set(trainer.compute_metrics(outputs, image)), {"psnr"})

    def test_e1_scale_opacity_and_cull_keep_sdf_geometry_frozen(self):
        _, road, _, _ = self.fixture()
        road.surface_field.cfg.update(align_normals=True, free_scales=True, free_opacity=True)
        road._scales = nn.Parameter(torch.log(torch.tensor([[0.12, 0.15, 0.04],
                                                            [0.12, 0.15, 0.04]])))
        road._opacities = nn.Parameter(torch.logit(torch.full((2, 1), 0.1)))
        road._features_dc = nn.Parameter(torch.zeros(2, 3))
        road._features_rest = nn.Parameter(torch.zeros(2, 3, 3))
        road.reg_cfg = {"lowest_opacity": 0.2}
        road.ctrl_cfg = SimpleNamespace(cull_alpha_thresh=0.005, reset_alpha_interval=4000)
        torch.testing.assert_close(road.get_scaling, road._scales.exp())
        torch.testing.assert_close(road.get_opacity, torch.full((2, 1), 0.1))
        road.cull_gaussians()
        self.assertFalse(road._means.requires_grad)
        self.assertFalse(road._quats.requires_grad)
        self.assertTrue(road._scales.requires_grad)
        self.assertTrue(road._opacities.requires_grad)

    def test_actual_sync_bakes_height_and_orientation_and_retains_unsupported(self):
        _, road, _, _ = self.fixture()
        field = road.surface_field
        field.cfg.update(align_normals=True, query_chunk_size=1)
        # A known sloped surface; the second Gaussian is outside its support.
        rows, cols = field.base_heights.shape
        y, x = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
        field.base_heights.copy_(0.2 * (field.origin[0] + (x + 0.5) * field.cell_size))
        road._means.data[1] = torch.tensor([100., 100., 7.])
        road.sync_surface_height()
        w, x, y, z = road._quats[0]
        normal = torch.stack((2 * (x*z + w*y), 2 * (y*z - w*x), 1 - 2 * (x*x + y*y)))
        expected = torch.nn.functional.normalize(torch.tensor([-0.2, 0., 1.]), dim=0)
        torch.testing.assert_close(normal, expected)
        torch.testing.assert_close(road._means[1], torch.tensor([100., 100., 7.]))
        torch.testing.assert_close(road._quats[1], torch.tensor([1., 0., 0., 0.]))
        self.assertIsNone(road._quats.grad)
        self.assertTrue(all(p.grad is None for p in field.parameters()))
        before = road._quats.detach().clone()
        road.sync_surface_height()
        torch.testing.assert_close(road._quats, before)

    def test_legacy_orientation_and_warmup_remain_unchanged(self):
        _, road, _, _ = self.fixture()
        original = road._quats.detach().clone()
        road.sync_surface_height()
        torch.testing.assert_close(road._quats, original)
        road.surface_field.cfg.update(align_normals=True, apply_after=100)
        road.sync_surface_height()
        torch.testing.assert_close(road._quats, original)

    def test_checkpoint_restores_baked_rotation_as_geometry_not_rgb_parameter(self):
        _, road, _, _ = self.fixture()
        road.surface_field.cfg["align_normals"] = True
        road.device = torch.device("cpu")
        for name, shape in (("_scales", (2, 3)), ("_features_dc", (2, 3)),
                            ("_features_rest", (2, 3, 3)), ("_opacities", (2, 1))):
            setattr(road, name, nn.Parameter(torch.zeros(shape)))
        road._quats.data[:] = torch.tensor([0.9, 0.1, 0.2, 0.3])
        state = {key: value.clone() for key, value in road.state_dict().items()}
        road.load_state_dict(state)
        torch.testing.assert_close(road._quats, state["_quats"])
        self.assertFalse(road._quats.requires_grad)
        self.assertFalse(road._means.requires_grad)

    def test_sync_reuses_bake_and_base_cache_until_geometry_changes(self):
        _, road, _, _ = self.fixture()
        field = road.surface_field
        field.cfg["align_normals"] = True
        with patch.object(field, "height_and_normal", wraps=field.height_and_normal) as query, \
                patch.object(field, "base_height", wraps=field.base_height) as base:
            road.sync_surface_height()
            self.assertEqual((query.call_count, base.call_count), (1, 1))
            road.step += 1
            road.sync_surface_height()
            with torch.no_grad():
                field.color[-2].bias.add_(0.01)
            road.sync_surface_height()
            self.assertEqual((query.call_count, base.call_count), (1, 1))
            with torch.no_grad():
                field.geometry[-1].bias[0].add_(0.1)
            road.sync_surface_height()
            self.assertEqual((query.call_count, base.call_count), (2, 1))
            torch.testing.assert_close(road._means[:, 2], field.height(road._means[:, :2])[0])
            base.reset_mock()
            with torch.no_grad():
                road._means[0, 0].add_(0.1)
            road.sync_surface_height()
            self.assertEqual((query.call_count, base.call_count), (3, 1))
            with torch.no_grad():
                field.base_heights.add_(0.05)
            road.sync_surface_height()
            self.assertEqual((query.call_count, base.call_count), (4, 2))

    def test_optimizer_bakes_once_then_forward_reuses_and_frozen_field_stops_baking(self):
        trainer, road, image, outputs = self.fixture()
        field = road.surface_field
        field.cfg["align_normals"] = True
        trainer.optimizer = torch.optim.Adam(list(field.parameters()) + [outputs["rgb"]], lr=0.001)
        road.sync_surface_height()
        with patch.object(field, "height_and_normal", wraps=field.height_and_normal) as query:
            trainer.optimizer.zero_grad(set_to_none=True)
            sum(trainer.compute_losses(outputs, image, {}, None).values()).backward()
            trainer.optimizer_step()
            self.assertEqual(query.call_count, 1)
            road.sync_surface_height()  # next render must not bake the same geometry
            self.assertEqual(query.call_count, 1)
            trainer.step = road.step = field.cfg["freeze_after"]
            trainer.optimizer.zero_grad(set_to_none=True)
            sum(trainer.compute_losses(outputs, image, {}, None).values()).backward()
            trainer.optimizer_step()
            road.sync_surface_height()
            self.assertEqual(query.call_count, 1)

    def test_replacing_points_and_reloading_checkpoint_invalidates_runtime_cache(self):
        _, road, _, _ = self.fixture()
        field = road.surface_field
        field.cfg["align_normals"] = True
        road.sync_surface_height()
        with patch.object(field, "height_and_normal", wraps=field.height_and_normal) as query:
            road._means = nn.Parameter(road._means.detach().clone(), requires_grad=False)
            road.sync_surface_height()
            self.assertEqual(query.call_count, 1)
            field.load_state_dict({k: v.clone() for k, v in field.state_dict().items()})
            road.sync_surface_height()
            self.assertEqual(query.call_count, 2)

    def test_frozen_field_stops_auxiliary_training(self):
        trainer, road, image, outputs = self.fixture()
        trainer.step = road.surface_field.cfg["freeze_after"]
        self.assertEqual(set(trainer.compute_losses(outputs, image, {}, None)), {"rgb_loss"})


if __name__ == "__main__":
    unittest.main()
