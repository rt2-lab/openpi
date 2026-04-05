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
    """Clamped negative weighted distance to nearest basin point.

    r = clamp(-min_m d(endpoint, basin_m), min_reward, 0)
    """

    def __init__(
        self,
        basin_points: list[dict],
        w_pos: float = 1.0,
        w_ori: float = 1.0,
        w_grip: float = 1.0,
        min_reward: float = -10.0,
        pos_slice: tuple[int, int] = (0, 3),
        ori_slice: tuple[int, int] = (3, 7),
        grip_idx: int = 7,
        **_: Any,
    ):
        if not basin_points:
            raise ValueError("basin_points must be a non-empty list")
        self.w_pos = w_pos
        self.w_ori = w_ori
        self.w_grip = w_grip
        self.min_reward = min_reward
        self.pos_slice = slice(*pos_slice)
        self.ori_slice = slice(*ori_slice)
        self.grip_idx = grip_idx

        self._basin_pos = jnp.array([bp["position"] for bp in basin_points], dtype=jnp.float32)
        self._basin_ori = jnp.array([bp["orientation"] for bp in basin_points], dtype=jnp.float32)
        self._basin_grip = jnp.array([bp["gripper"] for bp in basin_points], dtype=jnp.float32)

    def __call__(self, x0_hat: jnp.ndarray, current_state: jnp.ndarray) -> jnp.ndarray:
        endpoints = x0_hat[:, -1, :]  # (k, D)
        ep_pos = endpoints[:, self.pos_slice]    # (k, 3)
        ep_ori = endpoints[:, self.ori_slice]    # (k, 4)
        ep_grip = endpoints[:, self.grip_idx]    # (k,)

        # (k, 1, 3) - (1, M, 3) -> (k, M)
        d_pos = jnp.linalg.norm(
            ep_pos[:, None, :] - self._basin_pos[None, :, :], axis=-1
        )

        dot = jnp.abs(jnp.einsum("kd,md->km", ep_ori, self._basin_ori))
        dot = jnp.clip(dot, 0.0, 1.0)
        d_ori = 1.0 - dot

        d_grip = jnp.abs(ep_grip[:, None] - self._basin_grip[None, :])

        total = self.w_pos * d_pos + self.w_ori * d_ori + self.w_grip * d_grip
        min_dist = total.min(axis=-1)
        return jnp.clip(-min_dist, self.min_reward, 0.0)


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
