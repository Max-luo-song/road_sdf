"""CPU contract checks for the paper RSG field, independent of CUDA gsplat."""

from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

from models.extrags_road import ExtraGSRoadField


class ExtraGSRoadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)
        self.field = ExtraGSRoadField(dict(
            levels=2, features=2, table_size=64, min_resolution=4,
            max_resolution=8, hidden_dim=16, rays_per_image=16,
            samples_per_ray=48, eikonal_points=16))
        axis = torch.linspace(-2, 2, 17)
        x, y = torch.meshgrid(axis, axis, indexing="ij")
        road = torch.stack((x.flatten(), y.flatten(), torch.zeros(x.numel())), -1)
        self.field.initialize(road)

    def test_sdf_matches_paper_form_and_baked_normal(self):
        xy = torch.tensor([[0.11, 0.21], [0.57, -0.42]])
        height, normal, supported = self.field.height_and_normal(xy)
        self.assertTrue(supported.all())
        points = torch.cat((xy, height[:, None]), -1).requires_grad_(True)
        sdf, _ = self.field.sdf(points)
        torch.testing.assert_close(sdf, torch.zeros_like(sdf), atol=1e-6, rtol=0)
        derivative, = torch.autograd.grad(sdf.sum(), points)
        torch.testing.assert_close(normal, F.normalize(derivative, dim=-1), atol=1e-5, rtol=1e-5)

    def test_road_mask_l1_and_eikonal_update_field(self):
        c2w = torch.eye(4)
        c2w[:3, :3] = torch.diag(torch.tensor([1., -1., -1.]))
        c2w[2, 3] = 2.
        camera = SimpleNamespace(camtoworlds=c2w, Ks=torch.tensor([
            [8., 0., 2.], [0., 8., 2.], [0., 0., 1.]]))
        image = dict(road_masks=torch.ones(4, 4, dtype=torch.bool),
                     pixels=torch.zeros(4, 4, 3))
        losses = self.field.training_losses(image, camera, step=0)
        self.assertEqual(set(losses), {"road_field_rgb", "road_field_eikonal"})
        sum(losses.values()).backward()
        for parameter in (self.field.elevation[-1].bias, self.field.slope[-1].bias,
                          self.field.color[-2].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
