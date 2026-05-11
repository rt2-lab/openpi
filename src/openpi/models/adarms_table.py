"""Precomputes adaRMS modulation vectors for all discrete timesteps.

At inference with K discrete denoising steps, the Dense projection inside each
adaptive RMSNorm produces the same output for a given (timestep, layer, norm_slot).
This module tabulates those outputs once at model load, eliminating
depth * 2 * num_steps redundant matmuls per sample_actions call.
"""

from __future__ import annotations

import flax.nnx.bridge.variables as bridge_vars
import jax
import jax.numpy as jnp


def _get_linen_params(bridged_module):
    """Extract the Linen params dict from an nnx_bridge.ToNNX module."""
    nnx_attrs = {name: getattr(bridged_module, name) for name in bridged_module.linen_attributes}
    variables = bridge_vars.nnx_attrs_to_linen_vars(nnx_attrs)
    return variables["params"]


def build_adarms_table(model, num_steps: int = 10) -> dict:
    """Build lookup table of pre-computed adaRMS modulations.

    Must be called after model weights are loaded. Only applies to pi0.5
    models (pi05=True) that use adaptive RMSNorm.

    Args:
        model: Pi0 model instance with pi05=True.
        num_steps: Number of discrete denoising steps used at inference.

    Returns:
        Table dict with keys "blocks", "final", "num_steps".
    """
    from openpi.models.pi0 import posemb_sincos  # late import to avoid circular dep

    width = model.time_mlp_in.kernel.value.shape[0]

    # Discrete timesteps: in the denoising loop, time starts at 1.0 and
    # decreases by dt = -1/num_steps each step. Step i → time = 1 - i/num_steps.
    timesteps = jnp.linspace(1.0, 1.0 / num_steps, num_steps)  # [num_steps]

    # Compute time MLP: posemb → Linear → swish → Linear → swish
    time_embs = posemb_sincos(timesteps, width, min_period=4e-3, max_period=4.0)
    conds = time_embs @ model.time_mlp_in.kernel.value + model.time_mlp_in.bias.value
    conds = jnp.float32(conds)
    conds = conds * jax.nn.sigmoid(conds)  # swish in float32
    conds = conds @ jnp.float32(model.time_mlp_out.kernel.value) + jnp.float32(model.time_mlp_out.bias.value)
    conds = conds * jax.nn.sigmoid(conds)  # swish
    # conds: [num_steps, D] in float32

    # Extract Dense weights for action expert norms (expert index 1)
    params = _get_linen_params(model.PaliGemma.llm)
    layers_p = params["layers"]

    attn_k = jnp.float32(layers_p["pre_attention_norm_1"]["Dense_0"]["kernel"])  # [depth, D, 3D]
    attn_b = jnp.float32(layers_p["pre_attention_norm_1"]["Dense_0"]["bias"])    # [depth, 3D]
    ffw_k = jnp.float32(layers_p["pre_ffw_norm_1"]["Dense_0"]["kernel"])         # [depth, D, 3D]
    ffw_b = jnp.float32(layers_p["pre_ffw_norm_1"]["Dense_0"]["bias"])           # [depth, 3D]
    final_k = jnp.float32(params["final_norm_1"]["Dense_0"]["kernel"])           # [D, 3D]
    final_b = jnp.float32(params["final_norm_1"]["Dense_0"]["bias"])             # [3D]

    # Compute modulations: cond @ kernel + bias
    attn_mod = jnp.einsum("sd,ldm->slm", conds, attn_k) + attn_b[None]   # [steps, depth, 3D]
    ffw_mod = jnp.einsum("sd,ldm->slm", conds, ffw_k) + ffw_b[None]      # [steps, depth, 3D]
    final_mod = conds @ final_k + final_b[None]                            # [steps, 3D]

    # Cast to the model's computation dtype (embed_dtype, typically bf16).
    # Note: param storage dtype is float32 but computation happens in embed_dtype.
    dtype = jnp.dtype(model.PaliGemma.llm.module.embed_dtype)
    attn_mod = attn_mod.astype(dtype)
    ffw_mod = ffw_mod.astype(dtype)
    final_mod = final_mod.astype(dtype)

    # Split into (scale, shift, gate) and add broadcast dims for [batch, seq]
    D = width
    a_s, a_h, a_g = attn_mod[..., :D], attn_mod[..., D:2*D], attn_mod[..., 2*D:]
    f_s, f_h, f_g = ffw_mod[..., :D], ffw_mod[..., D:2*D], ffw_mod[..., 2*D:]
    fn_s, fn_h, fn_g = final_mod[..., :D], final_mod[..., D:2*D], final_mod[..., 2*D:]

    # blocks: 6-tuple each [steps, depth, 1, 1, D] (batch, seq broadcast)
    blocks = (
        a_s[:, :, None, None, :], a_h[:, :, None, None, :], a_g[:, :, None, None, :],
        f_s[:, :, None, None, :], f_h[:, :, None, None, :], f_g[:, :, None, None, :],
    )
    # final: 3-tuple each [steps, 1, 1, D]
    final = (fn_s[:, None, None, :], fn_h[:, None, None, :], fn_g[:, None, None, :])

    return {"blocks": blocks, "final": final, "num_steps": num_steps}


def tabulated_adarms_cond(table: dict, step_idx):
    """Index the table for a given denoising step.

    Args:
        table: Output of build_adarms_table.
        step_idx: Integer step index (0-based). May be a JAX traced value.

    Returns:
        Dict suitable for use as adarms_cond[1] in gemma.Module.__call__.
    """
    blocks = tuple(b[step_idx] for b in table["blocks"])
    final = tuple(f[step_idx] for f in table["final"])
    return {"blocks": blocks, "final": final}
