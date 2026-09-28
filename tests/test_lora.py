"""LoRA recovery layer on top of a frozen quaternion base (Stage 6)."""

import pytest
import torch

from quaternion import QuaternionLinear
from quaternion.lora import (
    LoRAQuaternionLinear,
    count_trainable,
    inject_lora,
    lora_parameters,
    mark_only_lora_trainable,
)


def make_base(seed=0):
    torch.manual_seed(seed)
    return QuaternionLinear(24, 16, bias=True)


def test_initial_output_equals_base():
    base = make_base()
    layer = LoRAQuaternionLinear(base, rank=8)
    x = torch.randn(5, 24)
    assert torch.allclose(layer(x), base(x), atol=0, rtol=0)


def test_base_is_frozen():
    base = make_base()
    layer = LoRAQuaternionLinear(base, rank=4)
    for name, p in layer.named_parameters():
        assert p.requires_grad == ("lora_" in name)


def test_scale_is_alpha_over_rank():
    layer = LoRAQuaternionLinear(make_base(), rank=8, alpha=32)
    assert layer.scale == 4.0
    layer2 = LoRAQuaternionLinear(make_base(), rank=16, alpha=None)
    assert layer2.scale == 2.0


def test_forward_shape_and_gradients():
    layer = LoRAQuaternionLinear(make_base(), rank=4)
    x = torch.randn(3, 7, 24, requires_grad=True)
    out = layer(x)
    assert out.shape == (3, 7, 16)
    out.sum().backward()
    assert layer.lora_A.grad is not None
    assert layer.lora_B.grad is not None
    assert layer.base.wr.grad is None


def test_inject_lora_wraps_all_quaternion_layers():
    import torch.nn as nn

    model = nn.Sequential(
        QuaternionLinear(24, 16),
        nn.ReLU(),
        QuaternionLinear(16, 8),
    )
    wrapped = inject_lora(model, rank=4)
    assert len(wrapped) == 2
    assert all(isinstance(m, LoRAQuaternionLinear) for m in model if isinstance(m, LoRAQuaternionLinear))
    x = torch.randn(2, 24)
    assert model(x).shape == (2, 8)


def test_mark_only_lora_trainable_and_count():
    import torch.nn as nn

    model = nn.Sequential(QuaternionLinear(24, 16), QuaternionLinear(16, 8))
    inject_lora(model, rank=4)
    mark_only_lora_trainable(model)
    # rank 4: (4*24 + 16*4) + (4*16 + 8*4) = 160 + 96
    assert count_trainable(model) == (4 * 24 + 16 * 4) + (4 * 16 + 8 * 4)
    assert len(list(lora_parameters(model))) == 4


def test_invalid_rank_raises():
    with pytest.raises(ValueError):
        LoRAQuaternionLinear(make_base(), rank=0)
