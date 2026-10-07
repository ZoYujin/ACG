from types import SimpleNamespace

import torch

from acg.attention import _orthogonalized_guidance, apply_acg, remove_acg


def test_guidance_is_orthogonal_to_masked_output():
    conditional = torch.tensor([[[[3.0, 4.0]]]])
    image_masked = torch.tensor([[[[1.0, 2.0]]]])

    guidance = _orthogonalized_guidance(conditional, image_masked)

    dot_product = (guidance * image_masked).sum(dim=-1)
    assert torch.allclose(dot_product, torch.zeros_like(dot_product), atol=1e-5)


def test_apply_and_remove_restore_original_forward():
    class Attention:
        def forward(self):
            return "original"

    layers = [
        SimpleNamespace(self_attn=Attention()),
        SimpleNamespace(self_attn=Attention()),
    ]
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    original = [layer.self_attn.forward for layer in layers]

    apply_acg(model, image_start=2, image_end=5, gamma=2.0)
    assert all(
        layer.self_attn.forward.__func__.__name__ == "_acg_attention_forward"
        for layer in layers
    )

    remove_acg(model)
    assert all(
        layer.self_attn.forward == expected
        for layer, expected in zip(layers, original)
    )


def test_invalid_layer_range_is_rejected():
    layers = [SimpleNamespace(self_attn=SimpleNamespace())]
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))

    try:
        apply_acg(model, image_start=1, image_end=2, end_layer=2)
    except ValueError as error:
        assert "Invalid layer range" in str(error)
    else:
        raise AssertionError("Expected invalid layer range to raise ValueError.")
