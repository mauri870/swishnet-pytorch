import pytest
import torch
from swishnet import CausalBlock, CausalConv1D, CausalGatedConv1D, SwishNet, SwishNetWide

BATCH = 2
TIME = 32
IN_CHANNELS = 20


class TestCausalConv1D:
    def test_output_shape(self):
        conv = CausalConv1D(in_channels=4, out_channels=8, kernel_size=3)
        y = conv(torch.randn(BATCH, 4, TIME))
        assert y.shape == (BATCH, 8, TIME)

    def test_output_length_preserved_with_dilation(self):
        conv = CausalConv1D(in_channels=4, out_channels=8, kernel_size=3, dilation=2)
        y = conv(torch.randn(BATCH, 4, TIME))
        assert y.shape == (BATCH, 8, TIME)

    def test_causal_property(self):
        # Output at time t must not depend on inputs at time t+1 or later.
        conv = CausalConv1D(in_channels=1, out_channels=1, kernel_size=3)
        x_past = torch.zeros(1, 1, TIME)
        x_future = x_past.clone()
        x_future[0, 0, TIME // 2 :] = 1.0
        with torch.no_grad():
            y_past = conv(x_past)
            y_future = conv(x_future)
        assert torch.allclose(y_past[0, 0, : TIME // 2], y_future[0, 0, : TIME // 2])


class TestCausalGatedConv1D:
    def test_output_shape(self):
        # Output channels = filters // 2 (sigmoid branch × tanh branch).
        layer = CausalGatedConv1D(in_channels=8, filters=16)
        y = layer(torch.randn(BATCH, 8, TIME))
        assert y.shape == (BATCH, 8, TIME)

    def test_output_shape_with_dilation(self):
        layer = CausalGatedConv1D(in_channels=8, filters=16, dilation=2)
        y = layer(torch.randn(BATCH, 8, TIME))
        assert y.shape == (BATCH, 8, TIME)

    def test_gating_bounds(self):
        # sigmoid ∈ (0,1) × tanh ∈ (-1,1) → product ∈ (-1,1)
        layer = CausalGatedConv1D(in_channels=4, filters=8)
        with torch.no_grad():
            y = layer(torch.randn(BATCH, 4, TIME))
        assert y.min() >= -1.0
        assert y.max() <= 1.0


class TestCausalBlock:
    def test_output_shape(self):
        # Two branches (kernel 3 and 6) concatenated → output channels == filters.
        block = CausalBlock(in_channels=8, filters=16)
        y = block(torch.randn(BATCH, 8, TIME))
        assert y.shape == (BATCH, 16, TIME)


class TestSwishNet:
    def test_output_shape_default(self):
        model = SwishNet(in_channels=IN_CHANNELS, out_channels=2)
        y = model(torch.randn(BATCH, IN_CHANNELS, TIME))
        assert y.shape == (BATCH, 2)

    def test_output_shape_custom_classes(self):
        model = SwishNet(in_channels=IN_CHANNELS, out_channels=4)
        y = model(torch.randn(BATCH, IN_CHANNELS, TIME))
        assert y.shape == (BATCH, 4)

    def test_output_shape_wide(self):
        model = SwishNet(in_channels=IN_CHANNELS, out_channels=2, width_multiply=2)
        y = model(torch.randn(BATCH, IN_CHANNELS, TIME))
        assert y.shape == (BATCH, 2)

    def test_output_is_finite(self):
        model = SwishNet(in_channels=IN_CHANNELS, out_channels=2)
        with torch.no_grad():
            y = model(torch.randn(BATCH, IN_CHANNELS, TIME))
        assert torch.isfinite(y).all()

    def test_batch_size_one(self):
        model = SwishNet(in_channels=IN_CHANNELS, out_channels=2)
        y = model(torch.randn(1, IN_CHANNELS, TIME))
        assert y.shape == (1, 2)

    def test_compatible_with_cross_entropy_loss(self):
        # Output must be raw logits so CrossEntropyLoss works without prior softmax.
        model = SwishNet(in_channels=IN_CHANNELS, out_channels=3)
        x = torch.randn(BATCH, IN_CHANNELS, TIME)
        targets = torch.randint(0, 3, (BATCH,))
        loss = torch.nn.CrossEntropyLoss()(model(x), targets)
        assert torch.isfinite(loss)


class TestSwishNetWide:
    def test_output_shape(self):
        model = SwishNetWide(classes=3)
        y = model(torch.randn(BATCH, IN_CHANNELS, TIME))
        assert y.shape == (BATCH, 3)

    def test_output_shape_binary(self):
        model = SwishNetWide(classes=2)
        y = model(torch.randn(BATCH, IN_CHANNELS, TIME))
        assert y.shape == (BATCH, 2)
