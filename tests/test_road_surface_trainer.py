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

    def test_frozen_field_stops_auxiliary_training(self):
        trainer, road, image, outputs = self.fixture()
        trainer.step = road.surface_field.cfg["freeze_after"]
        self.assertEqual(set(trainer.compute_losses(outputs, image, {}, None)), {"rgb_loss"})


if __name__ == "__main__":
    unittest.main()
