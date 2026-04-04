"""FK steering loop for flow-matching policies (JAX / Pi0.5).

Strategy: split the denoising into JIT-compiled segments (fori_loop chunks)
between resampling points.  Each segment runs at full XLA speed with the
shared KV cache.  Between segments we evaluate rewards and resample — ideally
staying entirely on-device when jax_output_params is provided (pure-JAX
unnormalization), falling back to a host-sync path otherwise.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from openpi.fk_steering.rewards import BaseReward, build_reward
from openpi.fk_steering.potentials import compute_potentials_and_resample

logger = logging.getLogger(__name__)


def fk_sample_actions(
    model: Any,
    sample_actions_batch_jit: Callable,
    denoise_segment_jit: Callable | None,
    encode_prefix_jit: Callable | None,
    observation: Any,
    rng: jax.Array,
    fk_config: dict,
    raw_state: np.ndarray,
    model_state: np.ndarray,
    output_transform_fn: Callable[[dict], dict] | None = None,
    num_steps: int = 10,
    jax_output_params: dict | None = None,
) -> np.ndarray:
    """Run FK steering on a flow-matching model.

    Returns:
        (k, action_horizon, D_physical) physical-space action trajectories.
    """
    k = fk_config["num_particles"]
    resample_steps = sorted(fk_config["resampling_schedule"])
    resample_steps = [s for s in resample_steps if 0 <= s < num_steps]

    if not resample_steps or denoise_segment_jit is None:
        rng, sample_rng = jax.random.split(rng)
        raw_actions = sample_actions_batch_jit(sample_rng, observation, k, num_steps=num_steps)
        return _to_physical(raw_actions, model_state, output_transform_fn, k)

    return _segmented_fk(
        model=model,
        denoise_segment_jit=denoise_segment_jit,
        encode_prefix_jit=encode_prefix_jit,
        observation=observation,
        rng=rng,
        fk_config=fk_config,
        raw_state=raw_state,
        model_state=model_state,
        output_transform_fn=output_transform_fn,
        num_steps=num_steps,
        resample_steps=resample_steps,
        k=k,
        jax_output_params=jax_output_params,
    )


# ── On-device output transform ───────────────────────────────────────────

def _to_physical_jax(
    x_t: jnp.ndarray,
    raw_state: jnp.ndarray,
    params: dict,
) -> jnp.ndarray:
    """Unnormalize + delta→absolute + slice, entirely in JAX (no host sync).

    Replicates the Unnormalize → AbsoluteActions → CollabOutputs chain
    using the fixed constants extracted at init time.
    """
    action_dim = params.get("action_dim", x_t.shape[-1])
    x = x_t[..., :action_dim]

    if params.get("use_quantiles", False):
        q01 = params["q01"][:action_dim]
        q99 = params["q99"][:action_dim]
        x = (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    elif "mean" in params:
        mean = params["mean"][:action_dim]
        std = params["std"][:action_dim]
        x = x * (std + 1e-6) + mean

    if "delta_mask" in params:
        mask = params["delta_mask"][:action_dim]
        x = x + jnp.where(mask, raw_state[:action_dim], 0.0)

    return x


# ── Fallback host-sync path ──────────────────────────────────────────────

def _to_physical(
    raw_actions: jnp.ndarray,
    model_state_np: np.ndarray,
    output_transform_fn: Callable | None,
    k: int,
) -> np.ndarray:
    """Convert model-space actions to physical space via Python transforms (host sync)."""
    actions_np = np.asarray(raw_actions)
    if output_transform_fn is None:
        return actions_np
    result = []
    for i in range(k):
        out = output_transform_fn({"state": model_state_np.copy(), "actions": actions_np[i].copy()})
        result.append(out["actions"])
    return np.stack(result, axis=0)


# ── Segmented FK loop ────────────────────────────────────────────────────

def _segmented_fk(
    model: Any,
    denoise_segment_jit: Callable,
    encode_prefix_jit: Callable | None,
    observation: Any,
    rng: jax.Array,
    fk_config: dict,
    raw_state: np.ndarray,
    model_state: np.ndarray,
    output_transform_fn: Callable | None,
    num_steps: int,
    resample_steps: list[int],
    k: int,
    jax_output_params: dict | None = None,
) -> np.ndarray:
    """FK steering using JIT-compiled denoising segments.

    When encode_prefix_jit is provided, prefix encoding runs as a single
    fused XLA program.  When jax_output_params is provided, intermediate
    reward evaluation stays entirely on-device (no host sync).
    """
    lambda_ = fk_config["lambda_"]
    potential_type = fk_config["potential_type"]
    adaptive = fk_config.get("adaptive_resampling", True)
    ess_threshold = fk_config.get("ess_threshold", 0.5)

    reward_fn = build_reward(fk_config["reward"])
    raw_state_j = jnp.array(raw_state, dtype=jnp.float32)
    model_state_np = np.asarray(model_state)

    dt = -1.0 / num_steps

    # Encode prefix once at batch=1 — JIT'd when available
    if encode_prefix_jit is not None:
        prefix_mask, kv_cache = encode_prefix_jit(observation)
    else:
        from openpi.models import model as _model
        from openpi.models.pi0 import make_attn_mask
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = model.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions,
        )

    # Broadcast to k particles
    kv_cache_k = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (x.shape[0], k, *x.shape[2:])),
        kv_cache,
    )
    prefix_mask_k = jnp.broadcast_to(prefix_mask, (k, prefix_mask.shape[1]))
    observation_k = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (k, *x.shape[1:])),
        observation,
    )

    rng, noise_rng = jax.random.split(rng)
    x_t = jax.random.normal(noise_rng, (k, model.action_horizon, model.action_dim))
    time = 1.0

    reward_history: list[jnp.ndarray] = []
    prev_reward: jnp.ndarray | None = None

    use_jax_path = jax_output_params is not None

    prev_end = -1
    for seg_idx, resample_at in enumerate(resample_steps):
        seg_steps = resample_at - prev_end
        if seg_steps <= 0:
            continue

        x_t, time = denoise_segment_jit(
            observation_k, prefix_mask_k, kv_cache_k,
            x_t, time, seg_steps, dt, k,
        )
        prev_end = resample_at

        # Reward evaluation
        if use_jax_path:
            phys_j = _to_physical_jax(x_t, raw_state_j, jax_output_params)
        else:
            phys_np = _to_physical(x_t, model_state_np, output_transform_fn, k)
            phys_j = jnp.array(phys_np)

        current_reward = reward_fn(phys_j, raw_state_j)
        reward_history.append(current_reward)

        rng, resample_rng = jax.random.split(rng)
        indices = compute_potentials_and_resample(
            rng=resample_rng,
            reward_history=reward_history,
            prev_reward=prev_reward,
            current_reward=current_reward,
            potential_type=potential_type,
            lambda_=lambda_,
            adaptive=adaptive,
            ess_threshold=ess_threshold,
        )

        if indices is not None:
            x_t = x_t[indices]
            reward_history = [r[indices] for r in reward_history]
            prev_reward = current_reward[indices]
        else:
            prev_reward = current_reward

    # Remaining steps after last resampling point
    remaining = num_steps - 1 - prev_end
    if remaining > 0:
        x_t, time = denoise_segment_jit(
            observation_k, prefix_mask_k, kv_cache_k,
            x_t, time, remaining, dt, k,
        )

    # Final conversion — one host sync here is fine
    if use_jax_path:
        return np.asarray(_to_physical_jax(x_t, raw_state_j, jax_output_params))
    return _to_physical(x_t, model_state_np, output_transform_fn, k)
