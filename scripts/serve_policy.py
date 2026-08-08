import dataclasses
import enum
import logging
import os
import signal
import socket

import jax
import numpy as np
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None
    # Path to a text file whose contents will always be used as the prompt, overriding any other source.
    prompt_file: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Precompute adaRMS modulations for pi0.5 (faster inference, same output).
    tabulate_adarms: bool = True

    # Optional wait/go gate (directory containing gate_head.pt + embed_norm.json).
    gate_checkpoint: str | None = None
    # Override P(move) HOLD threshold; default comes from gate_head.pt.
    gate_threshold: float | None = None

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(
    env: EnvMode, *, default_prompt: str | None = None, tabulate_adarms: bool = True
) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir,
            default_prompt=default_prompt, tabulate_adarms=tabulate_adarms,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir,
                default_prompt=args.default_prompt, tabulate_adarms=args.tabulate_adarms,
                gate_checkpoint=args.gate_checkpoint, gate_threshold=args.gate_threshold,
            )
        case Default():
            return create_default_policy(
                args.env, default_prompt=args.default_prompt, tabulate_adarms=args.tabulate_adarms,
            )


class PromptOverridePolicy(_policy.BasePolicy):
    """Wraps a policy to always override the prompt."""

    def __init__(self, policy: _policy.BasePolicy, prompt: str):
        self._policy = policy
        self._prompt = np.asarray(prompt)

    def infer(self, obs: dict) -> dict:
        obs["prompt"] = self._prompt
        return self._policy.infer(obs)

    @property
    def metadata(self) -> dict:
        return self._policy.metadata


_TRACE_DIR = os.environ.get("JAX_PROFILE_DIR", "/tmp/jax-profile")
_profiling_active = False


def _start_profile(signum, frame):
    global _profiling_active
    if _profiling_active:
        logging.warning("Profile already active, ignoring SIGUSR1")
        return
    logging.info("SIGUSR1 received — starting JAX trace to %s", _TRACE_DIR)
    jax.profiler.start_trace(_TRACE_DIR)
    _profiling_active = True


def _stop_profile(signum, frame):
    global _profiling_active
    if not _profiling_active:
        logging.warning("No active profile, ignoring SIGUSR2")
        return
    jax.profiler.stop_trace()
    _profiling_active = False
    logging.info("SIGUSR2 received — JAX trace saved to %s", _TRACE_DIR)


def main(args: Args) -> None:
    signal.signal(signal.SIGUSR1, _start_profile)
    signal.signal(signal.SIGUSR2, _stop_profile)
    logging.info(
        "JAX profiling ready  (PID %d).  kill -USR1 %d  to start,  kill -USR2 %d  to stop.  "
        "Traces go to %s  (override with JAX_PROFILE_DIR env var).",
        os.getpid(), os.getpid(), os.getpid(), _TRACE_DIR,
    )

    policy = create_policy(args)
    policy_metadata = policy.metadata

    if args.prompt_file is not None:
        with open(args.prompt_file) as f:
            prompt = f.read().strip()
        logging.info("Overriding prompt with: %s", prompt)
        policy = PromptOverridePolicy(policy, prompt)

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
