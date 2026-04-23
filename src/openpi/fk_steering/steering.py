"""FK steering loop for flow-matching policies (JAX / Pi0.5).

Fused implementation: the entire steered denoising (all segments + resampling)
runs as a single JIT-compiled XLA program with zero Python round-trips between
denoising steps.  Uses difference_potential only for resampling weights.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.fk_steering.potentials import (
    difference_potential,
    effective_sample_size,
    multinomial_resample,
)
from openpi.fk_steering.rewards import build_reward
from openpi.models.pi0 import make_attn_mask

logger = logging.getLogger(__name__)

_fused_fn_cache: dict[int, Callable] = {}


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
    output_transform_fn: Callable | None = None,
    num_steps: int = 10,
    jax_output_params: dict | None = None,
) -> np.ndarray:
    """Run FK steering.  Returns (k, action_horizon, D_physical) trajectories."""
    k = fk_config["num_particles"]
    resample_steps = sorted(s for s in fk_config["resampling_schedule"] if 0 <= s < num_steps)

    if not resample_steps or encode_prefix_jit is None:
        rng, sample_rng = jax.random.split(rng)
        raw = sample_actions_batch_jit(sample_rng, observation, k, num_steps=num_steps)
        return _to_physical_host(raw, model_state, output_transform_fn, k)

    if jax_output_params is None:
        raise ValueError(
            "FK steering requires jax_output_params (on-device output transform). "
            "Ensure the policy's output_transform is a supported CompositeTransform."
        )

    return _fused_fk(
        model, encode_prefix_jit, observation, rng, fk_config,
        raw_state, num_steps, resample_steps, k, jax_output_params,
    )


# ── On-device output transform ───────────────────────────────────────────

def _to_physical_jax(x_t, raw_state, params):
    """Unnormalize + delta→absolute, entirely in JAX (no host sync)."""
    action_dim = params.get("action_dim", x_t.shape[-1])
    x = x_t[..., :action_dim]

    if params.get("use_quantiles", False):
        q01, q99 = params["q01"][:action_dim], params["q99"][:action_dim]
        x = (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    elif "mean" in params:
        x = x * (params["std"][:action_dim] + 1e-6) + params["mean"][:action_dim]

    if "delta_mask" in params:
        x = x + jnp.where(params["delta_mask"][:action_dim], raw_state[:action_dim], 0.0)
    return x


# ── Host-side output transform (fallback) ────────────────────────────────

def _to_physical_host(raw_actions, model_state_np, output_transform_fn, k):
    actions_np = np.asarray(raw_actions)
    if output_transform_fn is None:
        return actions_np
    out = []
    for i in range(k):
        o = output_transform_fn({"state": np.asarray(model_state_np).copy(),
                                  "actions": actions_np[i].copy()})
        out.append(o["actions"])
    return np.stack(out, axis=0)


# ── Fused FK loop (single XLA dispatch for all segments + resampling) ────

def _build_fused_fn(model, fk_config, jax_output_params, num_steps):
    """Build and return a JIT-compiled callable for the full FK loop.

    Follows the same nnx.split / nnx.merge pattern as module_jit:
    graphdef is captured by closure, state is passed as the first argument
    to the jitted function and forwarded by the wrapper.
    """
    graphdef, frozen_state = nnx.split(model)

    lambda_       = fk_config["lambda_"]
    adaptive      = fk_config.get("adaptive_resampling", True)
    ess_threshold = fk_config.get("ess_threshold", 0.5)
    reward_fn     = build_reward(fk_config["reward"])
    k             = fk_config["num_particles"]
    dt            = -1.0 / num_steps
    ah            = model.action_horizon

    def _raw(state, x_t, rng, raw_state_j,
             obs_k, pfx_mask_k, kv_cache_k, resample_mask_j):
        module = nnx.merge(graphdef, state)

        def scan_body(carry, should_resample):
            x_t, time, prev_rew, rng, has_prev = carry

            # ── one denoising step (action expert) ──
            suf_tok, suf_mask, suf_ar, adarms = module.embed_suffix(
                obs_k, x_t, jnp.broadcast_to(time, (k,)),
            )
            suf_attn = make_attn_mask(suf_mask, suf_ar)
            pfx_attn = einops.repeat(pfx_mask_k, "b p -> b s p",
                                     s=suf_tok.shape[1])
            full_attn = jnp.concatenate([pfx_attn, suf_attn], axis=-1)
            pos = (jnp.sum(pfx_mask_k, axis=-1)[:, None]
                   + jnp.cumsum(suf_mask, axis=-1) - 1)

            (_, suf_out), _ = module.PaliGemma.llm(
                [None, suf_tok],
                mask=full_attn,
                positions=pos,
                kv_cache=kv_cache_k,
                adarms_cond=[None, adarms],
            )
            v_t = module.action_out_proj(suf_out[:, -ah:])
            x_t = x_t + dt * v_t
            time = time + dt

            # ── conditional reward eval + resample ──
            rng, resample_rng = jax.random.split(rng)

            def _resample_branch(args):
                x_t, prev_rew, has_prev, rrng = args
                phys = _to_physical_jax(x_t, raw_state_j, jax_output_params)
                cur = reward_fn(phys, raw_state_j)

                # has_prev=False zeros prev_rew, giving absolute potential
                # exp(λ R_cur) at the first checkpoint. Subsequent checkpoints
                # use the standard difference exp(λ(R_cur - R_prev)).
                # Product telescopes exactly to exp(λ R_N).
                w = difference_potential(cur, prev_rew * has_prev, lambda_)
                if adaptive:
                    ess = effective_sample_size(w)
                    def _do(b):
                        idx = multinomial_resample(b[2], b[1])
                        return b[0][idx], b[3][idx]
                    def _skip(b):
                        return b[0], b[3]
                    x_t, cur = jax.lax.cond(
                        ess <= ess_threshold * k, _do, _skip,
                        (x_t, w, rrng, cur))
                else:
                    idx = multinomial_resample(rrng, w)
                    x_t, cur = x_t[idx], cur[idx]

                return x_t, cur, jnp.bool_(True)

            def _no_resample_branch(args):
                return args[0], args[1], args[2]   # x_t, prev_rew, has_prev

            x_t, prev_rew, has_prev = jax.lax.cond(
                should_resample,
                _resample_branch, _no_resample_branch,
                (x_t, prev_rew, has_prev, resample_rng))

            return (x_t, time, prev_rew, rng, has_prev), None

        init = (x_t, jnp.float32(1.0), jnp.zeros(k, dtype=jnp.float32),
                rng, jnp.bool_(False))
        (x_final, _, _, _, _), _ = jax.lax.scan(
            scan_body, init, resample_mask_j)
        return _to_physical_jax(x_final, raw_state_j, jax_output_params)

    jitted = jax.jit(_raw)

    def wrapper(x_t, rng, raw_state_j, obs_k, pfx_mask_k, kv_cache_k,
                resample_mask_j):
        return jitted(frozen_state, x_t, rng, raw_state_j,
                      obs_k, pfx_mask_k, kv_cache_k, resample_mask_j)

    return wrapper


def _fused_fk(model, encode_prefix_jit, observation, rng, fk_config,
              raw_state, num_steps, resample_steps, k, jax_output_params):
    cache_key = id(model)
    if cache_key not in _fused_fn_cache:
        logger.info("Building fused FK JIT function (first call, will compile on use)")
        _fused_fn_cache[cache_key] = _build_fused_fn(
            model, fk_config, jax_output_params, num_steps)
    call_fn = _fused_fn_cache[cache_key]

    raw_state_j = jnp.array(raw_state, dtype=jnp.float32)

    resample_mask = np.zeros(num_steps, dtype=np.bool_)
    for s in resample_steps:
        resample_mask[s] = True
    resample_mask_j = jnp.array(resample_mask)

    prefix_mask, kv_cache = encode_prefix_jit(observation)
    kv_cache_k = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (x.shape[0], k, *x.shape[2:])), kv_cache)
    pfx_mask_k = jnp.broadcast_to(prefix_mask, (k, prefix_mask.shape[1]))
    obs_k = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (k, *x.shape[1:])), observation)

    rng, noise_rng = jax.random.split(rng)
    x_t = jax.random.normal(
        noise_rng, (k, model.action_horizon, model.action_dim))

    return np.asarray(
        call_fn(x_t, rng, raw_state_j, obs_k, pfx_mask_k, kv_cache_k,
                resample_mask_j))
