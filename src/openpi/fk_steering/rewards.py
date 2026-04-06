"""Reward functions for FK steering (JAX).

All rewards operate on *unnormalized* (physical-space) action trajectories.
Signature: __call__(x0_hat, current_state) -> (k,) reward array.
    x0_hat:       (k, H, D)  predicted action trajectories
    current_state: (D,)       current robot state
"""
from __future__ import annotations

import abc
from typing import Any

import jax.numpy as jnp
import numpy as np


class BaseReward(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        x0_hat: jnp.ndarray,
        current_state: jnp.ndarray,
    ) -> jnp.ndarray:
        ...


class ConservativeSoftReward(BaseReward):
    """exp(-dist / sigma) where dist is the position-space L2 distance
    between the trajectory and the current end-effector position.

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

    def __init__(self, sigma: float = 0.05, dims: tuple[int, int] = (0, 3),
                 trajectory_reduction: str = "endpoint",
                 gripper_weight: float = 0.0, gripper_idx: int = 7, **_: Any):
        self.sigma = sigma
        self.dim_slice = slice(*dims)
        self.reduction = trajectory_reduction
        self.gripper_weight = gripper_weight
        self.gripper_idx = gripper_idx

    def _dist(self, traj: jnp.ndarray, state: jnp.ndarray) -> jnp.ndarray:
        """Compute per-timestep distance. traj: (k, ..., D), returns (k, ...)."""
        d = jnp.linalg.norm(traj[..., self.dim_slice] - state[self.dim_slice], axis=-1)
        if self.gripper_weight > 0:
            d = d + self.gripper_weight * jnp.abs(traj[..., self.gripper_idx] - state[self.gripper_idx])
        return d

    def __call__(self, x0_hat: jnp.ndarray, current_state: jnp.ndarray) -> jnp.ndarray:
        if self.reduction == "endpoint":
            dist = self._dist(x0_hat[:, -1, :], current_state)
        else:
            per_step = self._dist(x0_hat, current_state)  # (k, H)
            dist = per_step.max(axis=-1) if self.reduction == "max" else per_step.mean(axis=-1)
        return jnp.exp(-dist / self.sigma)


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
    """exp(-dist/sigma) where dist is the distance to the nearest basin
    point.  Reward is in [0, 1]: 1.0 on a basin, decaying toward 0 away.
    Far-away trajectories get ~0 reward (uniform resampling, no steering).

    drift_penalty (>= 0): extra multiplicative cost for trajectories that
    move *further* from the basin than the arm's current position.  When > 0
    the reward becomes  base * exp(-drift_penalty * max(0, d_traj - d_cur) / sigma).
    Trajectories that approach or stay get no penalty (factor = 1).

    basin_points: list of {"position": [x,y,z]} or
                  {"position": [x,y,z], "gripper": g} when gripper_weight > 0.
    """

    def __init__(
        self,
        basin_points: list[dict],
        sigma: float = 0.05,
        dims: tuple[int, int] = (0, 3),
        trajectory_reduction: str = "endpoint",
        gripper_weight: float = 0.0,
        gripper_idx: int = 7,
        drift_penalty: float = 0.0,
        **_: Any,
    ):
        if not basin_points:
            raise ValueError("basin_points must be a non-empty list")
        self.sigma = sigma
        self.dim_slice = slice(*dims)
        self.reduction = trajectory_reduction
        self.gripper_weight = gripper_weight
        self.gripper_idx = gripper_idx
        self.drift_penalty = drift_penalty
        self._basin_pos = jnp.array([bp["position"] for bp in basin_points], dtype=jnp.float32)
        if gripper_weight > 0:
            self._basin_grip = jnp.array([bp["gripper"] for bp in basin_points], dtype=jnp.float32)

    def _min_basin_dist(self, points: jnp.ndarray, gripper: jnp.ndarray | None = None) -> jnp.ndarray:
        """Min combined distance from each point to any basin. points: (..., 3) -> (...)."""
        shape = points.shape[:-1]
        flat = points.reshape(-1, points.shape[-1])  # (N, 3)
        d_pos = jnp.linalg.norm(flat[:, None, :] - self._basin_pos[None, :, :], axis=-1)  # (N, M)
        if self.gripper_weight > 0 and gripper is not None:
            flat_g = gripper.reshape(-1)  # (N,)
            d_grip = jnp.abs(flat_g[:, None] - self._basin_grip[None, :])  # (N, M)
            d_pos = d_pos + self.gripper_weight * d_grip
        return d_pos.min(axis=-1).reshape(shape)

    def __call__(self, x0_hat: jnp.ndarray, current_state: jnp.ndarray) -> jnp.ndarray:
        if self.reduction == "endpoint":
            grip = x0_hat[:, -1, self.gripper_idx] if self.gripper_weight > 0 else None
            dist = self._min_basin_dist(x0_hat[:, -1, self.dim_slice], grip)
        else:
            grip = x0_hat[:, :, self.gripper_idx] if self.gripper_weight > 0 else None
            per_step = self._min_basin_dist(x0_hat[:, :, self.dim_slice], grip)  # (k, H)
            dist = per_step.max(axis=-1) if self.reduction == "max" else per_step.mean(axis=-1)
        reward = jnp.exp(-dist / self.sigma)
        if self.drift_penalty > 0:
            cur_grip = current_state[self.gripper_idx:self.gripper_idx + 1] if self.gripper_weight > 0 else None
            dist_cur = self._min_basin_dist(current_state[self.dim_slice][None], cur_grip)  # (1,)
            drift = jnp.maximum(0.0, dist - dist_cur.squeeze())
            reward = reward * jnp.exp(-self.drift_penalty * drift / self.sigma)
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
