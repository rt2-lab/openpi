"""Reward functions for FK steering (JAX).

All rewards operate on *unnormalized* (physical-space) action trajectories.
Signature: __call__(x0_hat, current_state) -> (k,) reward array.
    x0_hat:       (k, H, D)  predicted action trajectories
    current_state: (D,)       current robot state

Reward shape: sigmoid  1 / (1 + exp(steepness * (dist - midpoint)))
    midpoint  — distance at which reward = 0.5 (region-of-attraction radius)
    steepness — controls transition sharpness (higher = sharper boundary)
"""
from __future__ import annotations

import abc
import logging
from typing import Any

import math

import jax.numpy as jnp

logger = logging.getLogger(__name__)

DEFAULT_REWARD_EPSILON = 5e-4


def _sigmoid_reward(dist: jnp.ndarray, steepness: float, midpoint: float,
                    reward_epsilon: float = DEFAULT_REWARD_EPSILON) -> jnp.ndarray:
    """1 / (1 + exp(steepness * (dist - midpoint))), clamped to 0 below reward_epsilon."""
    r = 1.0 / (1.0 + jnp.exp(steepness * (dist - midpoint)))
    return jnp.where(r < reward_epsilon, 0.0, r)


def attraction_radius(steepness: float, midpoint: float,
                      reward_epsilon: float = DEFAULT_REWARD_EPSILON) -> float:
    """Distance from a basin at which the sigmoid reward hits reward_epsilon (→ 0).

    Solves  1/(1+exp(k*(d-m))) = eps  ⟹  d = m + ln(1/eps - 1)/k
    """
    return midpoint + math.log(1.0 / reward_epsilon - 1.0) / steepness


def _parse_sigmoid_params(
    attraction_radius: float = 0.15,
    steepness: float = 60.0,
    reward_epsilon: float = DEFAULT_REWARD_EPSILON,
) -> tuple[float, float]:
    """Derive (midpoint, steepness) from attraction_radius and steepness.

    midpoint is placed so that the sigmoid hits reward_epsilon at exactly
    attraction_radius.
    """
    m = attraction_radius - math.log(1.0 / reward_epsilon - 1.0) / steepness
    return m, steepness


class BaseReward(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        x0_hat: jnp.ndarray,
        current_state: jnp.ndarray,
    ) -> jnp.ndarray:
        ...


class ConservativeSoftReward(BaseReward):
    """Sigmoid reward on position-space L2 distance to current state.

    Defaults to position-only (dims 0:3) because including quaternion and
    gripper in the norm makes the distance dominated by orientation
    (quaternion diffs are ~1.0 in magnitude) and the reward collapses to
    zero for any reasonable sigma.

    trajectory_reduction controls how distances across the action horizon
    are aggregated:
      "endpoint" — only the last timestep (default, original behavior)
      "mean"     — average distance across all timesteps
      "max"      — worst-case distance across all timesteps (strictest)
    """

    def __init__(self, dims: tuple[int, int] = (0, 3),
                 trajectory_reduction: str = "endpoint",
                 reward_epsilon: float = DEFAULT_REWARD_EPSILON,
                 attraction_radius: float = 0.15,
                 steepness: float = 60.0, **_: Any):
        self.reward_epsilon = reward_epsilon
        self.midpoint, self.steepness = _parse_sigmoid_params(
            attraction_radius, steepness, reward_epsilon)
        self.dim_slice = slice(*dims)
        self.reduction = trajectory_reduction

    def _dist(self, traj: jnp.ndarray, state: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.norm(traj[..., self.dim_slice] - state[self.dim_slice], axis=-1)

    def __call__(self, x0_hat: jnp.ndarray, current_state: jnp.ndarray) -> jnp.ndarray:
        if self.reduction == "endpoint":
            dist = self._dist(x0_hat[:, -1, :], current_state)
        else:
            per_step = self._dist(x0_hat, current_state)  # (k, H)
            dist = per_step.max(axis=-1) if self.reduction == "max" else per_step.mean(axis=-1)
        return _sigmoid_reward(dist, self.steepness, self.midpoint, self.reward_epsilon)


class ConservativeHardReward(BaseReward):
    """1.0 if position-space distance within epsilon, else 0.0.

    Same position-only default and trajectory_reduction options as
    ConservativeSoftReward.
    """

    def __init__(self, epsilon: float = 0.05, dims: tuple[int, int] = (0, 3),
                 trajectory_reduction: str = "endpoint", **_: Any):
        self.epsilon = epsilon
        self.dim_slice = slice(*dims)
        self.reduction = trajectory_reduction

    def __call__(self, x0_hat: jnp.ndarray, current_state: jnp.ndarray) -> jnp.ndarray:
        state_slice = current_state[self.dim_slice]
        if self.reduction == "endpoint":
            dist = jnp.linalg.norm(x0_hat[:, -1, self.dim_slice] - state_slice, axis=-1)
        else:
            per_step = jnp.linalg.norm(x0_hat[:, :, self.dim_slice] - state_slice, axis=-1)
            dist = per_step.max(axis=-1) if self.reduction == "max" else per_step.mean(axis=-1)
        return jnp.where(dist < self.epsilon, 1.0, 0.0)


class NeutralBasinReward(BaseReward):
    """Sigmoid reward anchored to the basin **closest to the robot's current EE**.

    The basin nearest the current end-effector position determines the target
    for *all* trajectory particles.  This prevents different particles from
    latching onto different basins and pulling the robot in conflicting
    directions.  As the robot moves between basins, ``j_cur`` naturally flips
    and trajectories begin tracking the new basin.

    reward = r_pos * r_traj * drift_factor

    r_pos and r_traj both use  1/(1 + exp(k*(dist - d))).
    r_pos is a scalar (same for all particles) — when the arm is far from any
    basin (r_pos -> 0), all trajectory rewards collapse to ~0 and FK
    resampling becomes uniform, recovering the original unsteered distribution.

    drift_penalty (>= 0): asymmetric retention.  Trajectories that move further
    from the basin than the arm's current position are penalized by
    exp(-drift_penalty * max(0, d_traj - d_current)).
    Approaching or staying incurs no penalty.

    basin_points: list of {"position": [x,y,z], optional "attraction_radius": float}.
        Any basin omitting attraction_radius uses the top-level attraction_radius
        argument as default.
    """

    def __init__(
        self,
        basin_points: list[dict],
        dims: tuple[int, int] = (0, 3),
        trajectory_reduction: str = "endpoint",
        reward_epsilon: float = DEFAULT_REWARD_EPSILON,
        drift_penalty: float = 0.0,
        attraction_radius: float = 0.15,
        steepness: float = 60.0,
        **_: Any,
    ):
        if not basin_points:
            raise ValueError("basin_points must be a non-empty list")
        self.reward_epsilon = reward_epsilon
        self.steepness = steepness
        self.drift_penalty = drift_penalty
        self.dim_slice = slice(*dims)
        self.reduction = trajectory_reduction
        self._basin_pos = jnp.array([bp["position"] for bp in basin_points], dtype=jnp.float32)
        radii = [
            float(bp.get("attraction_radius", attraction_radius)) for bp in basin_points
        ]
        self._basin_midpoints = jnp.array(
            [_parse_sigmoid_params(r, steepness, reward_epsilon)[0] for r in radii],
            dtype=jnp.float32,
        )

    def _nearest_basin_dist_and_idx(self, points: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Closest basin per row: L2 distance to argmin center and basin index.

        points: (..., 3); each row picks the basin with minimum distance.
        """
        shape = points.shape[:-1]
        flat = points.reshape(-1, points.shape[-1])  # (N, 3)
        d_pos = jnp.linalg.norm(flat[:, None, :] - self._basin_pos[None, :, :], axis=-1)  # (N, M)
        j = jnp.argmin(d_pos, axis=-1)
        d = jnp.min(d_pos, axis=-1)
        return d.reshape(shape), j.reshape(shape)

    def _dist_to_basin(self, points: jnp.ndarray, basin_idx: jnp.ndarray) -> jnp.ndarray:
        """L2 distance from each point in (..., 3) to a single basin center.

        basin_idx must be a scalar JAX array (not a Python int) so this
        remains traceable inside jax.jit / jax.lax.scan.
        """
        center = self._basin_pos[basin_idx]  # (3,) via dynamic indexing
        return jnp.linalg.norm(points - center, axis=-1)

    def _sigmoid_pair(self, dist: jnp.ndarray, midpoint: jnp.ndarray) -> jnp.ndarray:
        eps = self.reward_epsilon
        r = 1.0 / (1.0 + jnp.exp(self.steepness * (dist - midpoint)))
        return jnp.where(r < eps, 0.0, r)

    def __call__(self, x0_hat: jnp.ndarray, current_state: jnp.ndarray) -> jnp.ndarray:
        # Anchor everything to the basin closest to the robot's current EE.
        dist_cur, j_cur = self._nearest_basin_dist_and_idx(
            current_state[self.dim_slice][None])  # (1,), (1,)
        mid_cur = self._basin_midpoints[j_cur]
        r_pos = self._sigmoid_pair(dist_cur, mid_cur)

        j = j_cur.squeeze()  # scalar JAX array, safe inside jit
        if self.reduction == "endpoint":
            d_traj = self._dist_to_basin(x0_hat[:, -1, self.dim_slice], j)
            r_traj = self._sigmoid_pair(d_traj, mid_cur)
            dist_for_drift = d_traj
        else:
            d_step = self._dist_to_basin(x0_hat[:, :, self.dim_slice], j)  # (k, H)
            if self.reduction == "max":
                dist_for_drift = d_step.max(axis=-1)
            else:
                dist_for_drift = d_step.mean(axis=-1)
            r_traj = self._sigmoid_pair(dist_for_drift, mid_cur)

        reward = r_pos * r_traj

        if self.drift_penalty > 0:
            drift = jnp.maximum(0.0, dist_for_drift - dist_cur.squeeze())
            reward = reward * jnp.exp(-self.drift_penalty * drift)

        return reward


REWARD_REGISTRY: dict[str, type[BaseReward]] = {
    "conservative_soft": ConservativeSoftReward,
    "conservative_hard": ConservativeHardReward,
    "neutral_basin": NeutralBasinReward,
}


def build_reward(cfg: dict) -> BaseReward:
    rtype = cfg["type"]
    if rtype not in REWARD_REGISTRY:
        raise ValueError(f"Unknown reward type '{rtype}'. Available: {list(REWARD_REGISTRY)}")
    return REWARD_REGISTRY[rtype](**cfg.get("params", {}))
