"""Difference-potential resampling for FK steering (JAX, fully JIT-compatible)."""
from __future__ import annotations

import jax
import jax.numpy as jnp


def difference_potential(
    current_reward: jnp.ndarray,
    prev_reward: jnp.ndarray,
    lambda_: float,
) -> jnp.ndarray:
    return jnp.exp(lambda_ * (current_reward - prev_reward))


def effective_sample_size(weights: jnp.ndarray) -> jnp.ndarray:
    w = weights / weights.sum()
    return 1.0 / (w * w).sum()


def multinomial_resample(
    rng: jax.Array,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """Sample k indices from normalized weights. Returns (k,) int32 array."""
    log_probs = jnp.log(jnp.clip(weights / weights.sum(), 1e-8, None))
    return jax.random.categorical(rng, log_probs, shape=(weights.shape[0],))
