"""ACG intervention for the eager Llama attention used by LLaVA-NeXT.

The implementation targets transformers 4.37.2 and Vicuna-based LLaVA 1.6.
"""

from __future__ import annotations

import math
import types
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def _orthogonalized_guidance(
    conditional: torch.Tensor,
    image_masked: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return the component of the conditional delta orthogonal to image_masked."""
    delta = conditional - image_masked
    direction = image_masked / (image_masked.norm(dim=-1, keepdim=True) + eps)
    projection = (delta * direction).sum(dim=-1, keepdim=True) * direction
    return delta - projection


def _acg_attention_forward(
    self: Any,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Any] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs: Any,
):
    """LlamaAttention forward with an ACG correction on the current query."""
    del use_cache, kwargs

    if self.num_key_value_heads != self.num_heads:
        raise NotImplementedError(
            "This reference implementation supports standard multi-head attention only."
        )
    if getattr(self.config, "pretraining_tp", 1) != 1:
        raise NotImplementedError("pretraining_tp values other than 1 are unsupported.")

    batch_size, query_length, _ = hidden_states.size()
    query_states = (
        self.q_proj(hidden_states)
        .view(batch_size, query_length, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(batch_size, query_length, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(batch_size, query_length, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )

    key_value_length = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError("A layer index is required when using the KV cache.")
        key_value_length += past_key_value.get_usable_length(
            key_value_length, self.layer_idx
        )

    cos, sin = self.rotary_emb(value_states, seq_len=key_value_length)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    attention_scores = torch.matmul(
        query_states, key_states.transpose(2, 3)
    ) / math.sqrt(self.head_dim)

    expected_shape = (
        batch_size,
        self.num_heads,
        query_length,
        key_value_length,
    )
    if attention_scores.size() != expected_shape:
        raise ValueError(
            f"Attention scores should have shape {expected_shape}, "
            f"but got {attention_scores.size()}."
        )

    if attention_mask is not None:
        mask_shape = (batch_size, 1, query_length, key_value_length)
        if attention_mask.size() != mask_shape:
            raise ValueError(
                f"Attention mask should have shape {mask_shape}, "
                f"but got {attention_mask.size()}."
            )
        attention_scores = attention_scores + attention_mask
        attention_scores = attention_scores.clamp_min(
            torch.finfo(attention_scores.dtype).min
        )

    conditional_weights = F.softmax(
        attention_scores, dim=-1, dtype=torch.float32
    ).to(query_states.dtype)
    conditional_output = torch.matmul(conditional_weights, value_states)

    masked_scores = attention_scores.clone()
    masked_scores[
        :, :, -1, self.acg_image_start : self.acg_image_end
    ] = torch.finfo(masked_scores.dtype).min
    masked_weights = F.softmax(masked_scores, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    image_masked_output = torch.matmul(masked_weights, value_states)

    guidance = _orthogonalized_guidance(
        conditional_output,
        image_masked_output,
        eps=self.acg_eps,
    )
    attention_output = conditional_output + self.acg_gamma * guidance

    attention_output = attention_output.transpose(1, 2).contiguous()
    attention_output = attention_output.reshape(
        batch_size, query_length, self.hidden_size
    )
    attention_output = self.o_proj(attention_output)

    if not output_attentions:
        conditional_weights = None

    return attention_output, conditional_weights, past_key_value


def _decoder_layers(model: Any):
    """Resolve the Llama decoder layers from a LLaVA model or its language model."""
    candidates = (
        model,
        getattr(model, "model", None),
        getattr(getattr(model, "model", None), "model", None),
    )
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate.layers
    raise TypeError("Could not locate Llama decoder layers on the supplied model.")


def apply_acg(
    model: Any,
    image_start: int,
    image_end: int,
    gamma: float = 2.0,
    start_layer: int = 0,
    end_layer: Optional[int] = None,
    eps: float = 1e-6,
) -> Any:
    """Enable ACG on a contiguous range of decoder layers.

    Image indices use Python slice semantics: image_start is inclusive and
    image_end is exclusive.
    """
    layers = _decoder_layers(model)
    end_layer = len(layers) if end_layer is None else end_layer

    if not 0 <= start_layer <= end_layer <= len(layers):
        raise ValueError(
            f"Invalid layer range [{start_layer}, {end_layer}) "
            f"for a {len(layers)}-layer model."
        )
    if not 0 <= image_start < image_end:
        raise ValueError("Expected 0 <= image_start < image_end.")

    for layer in layers[start_layer:end_layer]:
        attention = layer.self_attn
        if not hasattr(attention, "_acg_original_forward"):
            attention._acg_original_forward = attention.forward
        attention.acg_image_start = image_start
        attention.acg_image_end = image_end
        attention.acg_gamma = float(gamma)
        attention.acg_eps = float(eps)
        attention.forward = types.MethodType(_acg_attention_forward, attention)

    return model


def remove_acg(model: Any) -> Any:
    """Restore every attention layer previously patched by apply_acg."""
    for layer in _decoder_layers(model):
        attention = layer.self_attn
        original_forward = getattr(attention, "_acg_original_forward", None)
        if original_forward is not None:
            attention.forward = original_forward
            del attention._acg_original_forward
        for attribute in (
            "acg_image_start",
            "acg_image_end",
            "acg_gamma",
            "acg_eps",
        ):
            if hasattr(attention, attribute):
                delattr(attention, attribute)
    return model
