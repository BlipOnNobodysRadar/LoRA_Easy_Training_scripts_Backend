"""Objective direction and weighting, independent of the full SDXL model."""
import math
import unittest

import torch
from backend.preference.loss import diffusion_dpo_loss


class ObjectiveTests(unittest.TestCase):
    def test_initial_equal_policy_has_log_two_loss_and_correct_gradient(self):
        policy = torch.tensor([[0.2, 0.8]], requires_grad=True)
        reference = policy.detach().clone().requires_grad_()
        loss, metrics = diffusion_dpo_loss(policy, reference, 4)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        loss.backward()
        # Gradient descent reduces winner error and increases rejected error.
        torch.testing.assert_close(policy.grad, torch.tensor([[1., -1.]]))
        self.assertIsNone(reference.grad)
        self.assertEqual(metrics['preference_accuracy'], 0.5)

    def test_reference_relative_improvement_reduces_loss(self):
        reference = torch.tensor([[.2, .8]])
        better, _ = diffusion_dpo_loss(torch.tensor([[.1, .9]]), reference, 4)
        worse, _ = diffusion_dpo_loss(torch.tensor([[.3, .7]]), reference, 4)
        self.assertLess(better.item(), math.log(2))
        self.assertGreater(worse.item(), math.log(2))

    def test_strength_survives_single_pair_accumulation(self):
        base = torch.tensor([[.2, .8]])
        gradients = []
        for weight in (.5, 2):
            policy = base.clone().requires_grad_()
            loss, _ = diffusion_dpo_loss(policy, base, 4, [weight])
            loss.backward()
            gradients.append(policy.grad)
        torch.testing.assert_close(gradients[1], gradients[0] * 4)

    def test_accumulation_matches_batch_mean(self):
        p = torch.tensor([[.3, .8], [.2, .4]], requires_grad=True)
        ref = torch.tensor([[.2, .8], [.2, .5]])
        loss, _ = diffusion_dpo_loss(p, ref, 4, [.5, 2])
        loss.backward()
        q = p.detach().clone().requires_grad_()
        for i, weight in enumerate([.5, 2]):
            micro, _ = diffusion_dpo_loss(q[i:i+1], ref[i:i+1], 4, [weight])
            (micro / 2).backward()
        torch.testing.assert_close(p.grad, q.grad)

    def test_invalid_values_cannot_update_optimizer(self):
        for weights in ([0], [float('nan')], [-1]):
            with self.assertRaises(ValueError):
                diffusion_dpo_loss(torch.ones(1, 2), torch.ones(1, 2), 4, weights)
        with self.assertRaises(FloatingPointError):
            diffusion_dpo_loss(torch.tensor([[float('nan'), 0]]), torch.zeros(1, 2), 4)


if __name__ == '__main__':
    unittest.main()
