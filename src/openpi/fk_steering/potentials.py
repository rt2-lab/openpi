"""Potential functions and resampling utilities for FK steering (JAX)."""
from __future__ import annotations

import jax
import jax.numpy as jnp


# ── Potential functions ───────────────────────────────────────────────────

def max_potential(
    reward_history: list[jnp.ndarray],
    lambda_: float,
) -> jnp.ndarray:
    stacked = jnp.stack(reward_history, axis=0)  # (T, k)
    max_r = stacked.max(axis=0)                   # (k,)
    return jnp.exp(lambda_ * max_r)


def difference_potential(
    current_reward: jnp.ndarray,
    prev_reward: jnp.ndarray | None,
    lambda_: float,
) -> jnp.ndarray:
    if prev_reward is None:
        return jnp.ones_like(current_reward)
    return jnp.exp(lambda_ * (current_reward - prev_reward))


def sum_potential(
    reward_history: list[jnp.ndarray],
    lambda_: float,
) -> jnp.ndarray:
    stacked = jnp.stack(reward_history, axis=0)
    total = stacked.sum(axis=0)
    return jnp.exp(lambda_ * total)


# ── Resampling ────────────────────────────────────────────────────────────

def effective_sample_size(weights: jnp.ndarray) -> float:
    w = weights / weights.sum()
    return float(1.0 / (w * w).sum())


def multinomial_resample(
    rng: jax.Array,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """Sample k indices from normalized weights. Returns (k,) int32 array."""
    k = weights.shape[0]
    log_probs = jnp.log(jnp.clip(weights / weights.sum(), 1e-8, None))
    return jax.random.categorical(rng, log_probs, shape=(k,))


def compute_potentials_and_resample(
    rng: jax.Array,
    reward_history: list[jnp.ndarray],
    prev_reward: jnp.ndarray | None,
    current_reward: jnp.ndarray,
    potential_type: str,
    lambda_: float,
    adaptive: bool,
    ess_threshold: float,
) -> jnp.ndarray | None:
    """Compute potentials and optionally resample. Returns indices or None."""
    k = current_reward.shape[0]

    if potential_type == "max":
        weights = max_potential(reward_history, lambda_)
    elif potential_type == "difference":
        weights = difference_potential(current_reward, prev_reward, lambda_)
    elif potential_type == "sum":
        weights = sum_potential(reward_history, lambda_)
    else:
        raise ValueError(f"Unknown potential type: {potential_type}")

    if adaptive:
        ess = effective_sample_size(weights)
        if ess > ess_threshold * k:
            return None

    return multinomial_resample(rng, weights)
