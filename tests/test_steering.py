import importlib.util
import threading
import unittest

from ntweak.engine import calibrate, steering_hooks, validate_targets

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.down_proj = nn.Linear(3, 2, bias=False)
            with torch.no_grad():
                self.mlp.down_proj.weight.copy_(torch.tensor([[3., 0., 1.], [4., 2., 0.]]))
        def forward(self, x):
            return self.mlp.down_proj(x)

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block()])
        def forward(self, input_ids, attention_mask=None, use_cache=False):
            return self.layers[0](input_ids.float())


@unittest.skipUnless(HAS_TORCH, "Install PyTorch to run numerical intervention tests")
class SteeringTests(unittest.TestCase):
    def setUp(self):
        self.block = Block()
        self.rows = [{"layer": 0, "neuron": 0, "dose": 0.5, "mode": "add", "enabled": True}]

    def test_exact_residual_push_norm_every_prefill_and_decode_token(self):
        prompt, token = torch.zeros(1, 4, 3), torch.zeros(1, 1, 3)
        with steering_hooks([self.block], self.rows, {0: 20.}) as metrics:
            # Column norm is 5, delta is .5 * 20 / 5 = 2, residual write is [6, 8].
            self.assertEqual(metrics[0]["activation_delta"], 2.)
            torch.testing.assert_close(self.block(prompt), torch.tensor([[[6., 8.]] * 4]))
            torch.testing.assert_close(self.block(token), torch.tensor([[[6., 8.]]]))
            self.assertAlmostEqual(float(self.block(token).norm()), 10.)
        torch.testing.assert_close(self.block(token), torch.zeros(1, 1, 2))

    def test_hooks_are_removed_on_error_and_weights_are_unchanged(self):
        original = self.block.mlp.down_proj.weight.clone()
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with steering_hooks([self.block], self.rows, {0: 20.}):
                raise RuntimeError("intentional")
        self.assertFalse(self.block.mlp.down_proj._forward_pre_hooks)
        torch.testing.assert_close(original, self.block.mlp.down_proj.weight)

    def test_multiple_neurons_negative_dose_and_multiplier_compose(self):
        rows = [*self.rows, {"layer": 0, "neuron": 1, "dose": -0.2, "mode": "add"}, {"layer": 0, "neuron": 0, "dose": 0, "mode": "mul"}]
        with steering_hooks([self.block], rows, {0: 20.}):
            torch.testing.assert_close(self.block(torch.zeros(1, 1, 3)), torch.tensor([[[0., -4.]]]))

    def test_calibration_uses_outgoing_layer_norms_skips_first_and_padding(self):
        backbone = Backbone()
        # Writes [300,400] for token 0 (excluded), then [3,4], [6,8], [9,12]; padding excluded.
        inputs = {"input_ids": torch.tensor([[[100., 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.], [999., 0., 0.]]]),
                  "attention_mask": torch.tensor([[1, 1, 1, 1, 0]])}
        medians = calibrate(backbone, backbone.layers, inputs, self.rows, threading.Event())
        self.assertEqual(medians[0], 10.)
        self.assertFalse(backbone.layers[0]._forward_hooks)
        inputs["input_ids"][:, 1:4] *= 2
        self.assertEqual(calibrate(backbone, backbone.layers, inputs, self.rows, threading.Event())[0], 20.)

    def test_target_bounds_and_zero_columns_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_targets([self.block], [{**self.rows[0], "neuron": 3}])
        with torch.no_grad():
            self.block.mlp.down_proj.weight[:, 0] = 0
        with self.assertRaisesRegex(ValueError, "zero"):
            with steering_hooks([self.block], self.rows, {0: 20.}):
                pass


if __name__ == "__main__":
    unittest.main()
