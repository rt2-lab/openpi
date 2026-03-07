from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        self._hardgate_enabled = getattr(model, "hardgate_enabled", False)

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            if self._hardgate_enabled:
                self._sample_actions_with_gate = nnx_utils.module_jit(model.sample_actions_with_gate)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()

        gate_prob = None
        if self._hardgate_enabled and not self._is_pytorch_model:
            actions, gate_prob_arr = self._sample_actions_with_gate(
                sample_rng_or_pytorch_device, observation, **sample_kwargs
            )
            gate_prob = float(np.asarray(gate_prob_arr[0]))
        else:
            actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)

        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        if gate_prob is not None:
            outputs["gate_prob"] = gate_prob
        return outputs

    def infer_batch(self, obs: dict, num_samples: int) -> dict:
        """Run batched inference: N action samples from a single observation.

        For PyTorch models, encodes the prefix once and tiles the KV cache.
        For JAX models, tiles the observation and calls sample_actions with batch=N.
        """
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)

        if self._is_pytorch_model:
            inputs = jax.tree.map(
                lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs
            )
            observation = _model.Observation.from_dict(inputs)
            start_time = time.monotonic()
            actions = self._model.sample_actions_batch(
                self._pytorch_device, observation, num_samples, **self._sample_kwargs
            )
            model_time = time.monotonic() - start_time
            actions_np = np.asarray(actions.detach().cpu())
            state_np = np.asarray(inputs["state"][0].detach().cpu())
        else:
            inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
            inputs = jax.tree.map(
                lambda x: jnp.broadcast_to(x[np.newaxis, ...], (num_samples, *x.shape)), inputs
            )
            self._rng, sample_rng = jax.random.split(self._rng)
            observation = _model.Observation.from_dict(inputs)
            start_time = time.monotonic()
            actions = self._sample_actions(sample_rng, observation, **self._sample_kwargs)
            model_time = time.monotonic() - start_time
            actions_np = np.asarray(actions)
            state_np = np.asarray(inputs["state"][0])

        all_actions = []
        for i in range(num_samples):
            sample_out = {"state": state_np, "actions": actions_np[i]}
            sample_out = self._output_transform(sample_out)
            all_actions.append(sample_out["actions"])

        return {
            "actions": np.stack(all_actions, axis=0),
            "policy_timing": {"infer_ms": model_time * 1000},
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
