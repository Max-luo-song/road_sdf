import unittest

import torch

from models.gaussians.basics import dup_in_optim, remove_from_optim


class OptimizerRefinementResumeTest(unittest.TestCase):
    def test_densify_and_prune_with_uninitialized_adam_state(self):
        means = torch.nn.Parameter(torch.arange(3.0).reshape(3, 1))
        opacity = torch.nn.Parameter(torch.ones(3, 1))
        optimizer = torch.optim.Adam(
            [{"name": "means", "params": [means]}, {"name": "opacity", "params": [opacity]}],
            lr=0.01,
        )

        # A model-only resume can leave one parameter without any Adam moments.
        means.grad = torch.ones_like(means)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.assertNotIn(opacity, optimizer.state)

        split = torch.tensor([1])
        new_means = torch.nn.Parameter(torch.cat([means.detach(), means.detach()[split].repeat(2, 1)]))
        new_opacity = torch.nn.Parameter(torch.cat([opacity.detach(), opacity.detach()[split].repeat(2, 1)]))
        dup_in_optim(optimizer, split, {"means": [new_means], "opacity": [new_opacity]}, n=2)
        self.assertEqual(optimizer.state[new_means]["exp_avg"].shape, (5, 1))
        self.assertTrue(torch.all(optimizer.state[new_means]["exp_avg"][3:] == 0))
        self.assertNotIn(new_opacity, optimizer.state)

        deleted = torch.tensor([False, True, False, False, True])
        kept_means = torch.nn.Parameter(new_means.detach()[~deleted].clone())
        kept_opacity = torch.nn.Parameter(new_opacity.detach()[~deleted].clone())
        remove_from_optim(optimizer, deleted, {"means": [kept_means], "opacity": [kept_opacity]})
        self.assertEqual(optimizer.state[kept_means]["exp_avg"].shape, (3, 1))
        self.assertNotIn(kept_opacity, optimizer.state)

        kept_opacity.grad = torch.ones_like(kept_opacity)
        optimizer.step()
        self.assertEqual(optimizer.state[kept_opacity]["exp_avg"].shape, (3, 1))


if __name__ == "__main__":
    unittest.main()
