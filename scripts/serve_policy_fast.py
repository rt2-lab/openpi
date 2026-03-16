"""Serve an OpenPI pi0.5 policy using realtime-vla's Triton-optimized inference.

Drop-in replacement for serve_policy.py that swaps the model backend to
realtime-vla's Pi05Inference (CUDA-graph + custom Triton kernels), giving
~3-4x speedup over the standard PyTorch/JAX inference.

The WebSocket interface is identical, so inference_watcher.py / openpi_client
work unchanged.

Usage:
    python scripts/serve_policy_fast.py \
        --config pi05_all_handover_derisk \
        --checkpoint-dir checkpoints/pi05_handover_derisk/pi05_kerimcan/24999 \
        --port 8000
"""

import argparse
import contextlib
import logging
import os
import pathlib
import pickle
import socket
import sys
import time

import numpy as np
import sentencepiece
import torch
import transformers

# ---------------------------------------------------------------------------
# Resolve repo roots so we can import both openpi and realtime-vla
# ---------------------------------------------------------------------------
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_OPENPI_ROOT = _SCRIPT_DIR.parent
_REALTIME_VLA_ROOT = _OPENPI_ROOT.parent / "realtime-vla"

if str(_OPENPI_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_OPENPI_ROOT / "src"))
if str(_REALTIME_VLA_ROOT) not in sys.path:
    sys.path.insert(0, str(_REALTIME_VLA_ROOT))

from openpi_client import base_policy as _base_policy
from openpi.serving import websocket_policy_server
from openpi.training import config as _config
from openpi.training import checkpoints as _checkpoints
import openpi.transforms as _transforms
import openpi.shared.download as _download

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SentencePiece wrapper that mimics HuggingFace AutoTokenizer.__call__
# ---------------------------------------------------------------------------
class _SPTokenizerWrapper:
    """Thin wrapper around SentencePiece that matches the HuggingFace tokenizer
    __call__ interface used by realtime-vla (Pi05Inference and prepare_prompt).

    OpenPI already auto-downloads the PaliGemma SentencePiece model from
    gs://big_vision/paligemma_tokenizer.model, so we reuse that instead of
    requiring a separate --tokenizer-path argument.
    """

    def __init__(self):
        path = _download.maybe_download(
            "gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            self._sp = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def __call__(
        self,
        text,
        return_tensors=None,
        truncation=False,
        max_length=None,
        padding=False,
        **_kw,
    ):
        texts = text if isinstance(text, list) else [text]
        all_tokens = []
        for t in texts:
            tokens = self._sp.encode(t, add_bos=True)
            if truncation and max_length is not None and len(tokens) > max_length:
                tokens = tokens[:max_length]
            all_tokens.append(tokens)

        if return_tensors == "pt":
            max_len = max(len(t) for t in all_tokens)
            padded = [t + [0] * (max_len - len(t)) for t in all_tokens]
            return {"input_ids": torch.tensor(padded)}
        return {"input_ids": all_tokens}


_sp_tokenizer: _SPTokenizerWrapper | None = None


def _get_sp_tokenizer() -> _SPTokenizerWrapper:
    global _sp_tokenizer
    if _sp_tokenizer is None:
        logger.info("Downloading PaliGemma SentencePiece tokenizer (same as OpenPI)...")
        _sp_tokenizer = _SPTokenizerWrapper()
    return _sp_tokenizer


@contextlib.contextmanager
def _patched_auto_tokenizer():
    """Temporarily replace AutoTokenizer.from_pretrained so that realtime-vla
    code picks up our SP wrapper instead of requiring a HuggingFace path."""
    wrapper = _get_sp_tokenizer()
    original = transformers.AutoTokenizer.from_pretrained
    transformers.AutoTokenizer.from_pretrained = staticmethod(lambda *a, **kw: wrapper)
    try:
        yield wrapper
    finally:
        transformers.AutoTokenizer.from_pretrained = original


# ---------------------------------------------------------------------------
# State quantization (matches OpenPI's PaligemmaTokenizer.tokenize)
# ---------------------------------------------------------------------------
_BINS = np.linspace(-1, 1, 257)[:-1]  # 256 bins over [-1, 1]


def quantize_state(state: np.ndarray) -> np.ndarray:
    """Quantize normalized state to discrete 0-255 tokens."""
    return (np.digitize(state, _BINS) - 1).astype(np.int64)


# ---------------------------------------------------------------------------
# FastTritonPolicy
# ---------------------------------------------------------------------------
class FastTritonPolicy(_base_policy.BasePolicy):
    """Policy backed by realtime-vla Pi05Inference.

    Applies the same input/output transforms as the standard OpenPI policy,
    then routes the actual neural-network forward pass through Pi05Inference
    with CUDA-graph replay and fused Triton kernels.
    """

    def __init__(
        self,
        pi05_infer,
        input_transforms,
        output_transforms,
        num_views: int,
        default_prompt: str,
        metadata: dict | None = None,
    ):
        self._infer = pi05_infer
        self._input_transform = _transforms.compose(input_transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._num_views = num_views
        self._default_prompt = default_prompt
        self._metadata = metadata or {}

    @property
    def metadata(self) -> dict:
        return self._metadata

    def _prepare_inputs(self, obs: dict):
        """Run input transforms and extract model-ready tensors.

        Returns (state_norm, state_tokens, prompt, images_tensor) where
        state_norm is the pre-padded normalized state (for output transforms),
        and images_tensor is on CUDA in bfloat16.
        """
        inputs = {k: v for k, v in obs.items()}
        inputs = self._input_transform(inputs)

        state_norm = np.asarray(inputs["state"], dtype=np.float32)
        if state_norm.shape[-1] < 32:
            state_padded = np.zeros(32, dtype=np.float32)
            state_padded[: state_norm.shape[-1]] = state_norm
        else:
            state_padded = state_norm
        state_tokens = quantize_state(state_padded)

        prompt = inputs.get("prompt", self._default_prompt)
        if not isinstance(prompt, str):
            prompt = str(prompt.item()) if hasattr(prompt, "item") else str(prompt)

        image_dict = inputs["image"]
        image_mask = inputs.get("image_mask", {})
        images = []
        for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
            img = np.asarray(image_dict[key])
            mask_val = image_mask.get(key, np.True_)
            if not np.bool_(mask_val):
                continue
            if img.dtype == np.uint8:
                img = img.astype(np.float32)
            elif img.max() <= 1.0:
                img = (img * 255.0).astype(np.float32)
            images.append(img)

        if len(images) != self._num_views:
            raise ValueError(
                f"Expected {self._num_views} active image views, got {len(images)}. "
                f"Check image_mask configuration."
            )

        images_tensor = torch.from_numpy(np.stack(images)).to(
            dtype=torch.bfloat16, device="cuda"
        )

        return state_norm, state_tokens, prompt, images_tensor

    def _run_once(self, state_tokens, prompt, images_tensor):
        """Run one forward pass with fresh noise. Returns raw actions as numpy."""
        noise = torch.randn(
            self._infer.chunk_size, 32, dtype=torch.bfloat16, device="cuda"
        )
        raw_actions = self._infer.forward(
            observation_images_normalized=images_tensor,
            diffusion_noise=noise,
            task_prompt=prompt,
            state_tokens=state_tokens,
        )
        return raw_actions.float().cpu().numpy()

    def _postprocess(self, state_norm, actions_np):
        """Run output transforms on a single sample."""
        outputs = {"state": state_norm, "actions": actions_np}
        return self._output_transform(outputs)

    def infer(self, obs: dict) -> dict:
        start = time.monotonic()

        state_norm, state_tokens, prompt, images_tensor = self._prepare_inputs(obs)

        infer_start = time.monotonic()
        actions_np = self._run_once(state_tokens, prompt, images_tensor)
        infer_ms = (time.monotonic() - infer_start) * 1000

        outputs = self._postprocess(state_norm, actions_np)
        outputs["policy_timing"] = {"infer_ms": infer_ms}

        total_ms = (time.monotonic() - start) * 1000
        logger.debug(f"Total policy time: {total_ms:.1f}ms (model: {infer_ms:.1f}ms)")

        return outputs

    def infer_batch(self, obs: dict, num_samples: int) -> dict:
        """Run N action samples from a single observation.

        Preprocesses once, then runs the Triton backend N times with
        independent noise. Output shape matches the standard OpenPI contract:
        actions is (num_samples, chunk_size, action_dim).
        """
        start = time.monotonic()

        state_norm, state_tokens, prompt, images_tensor = self._prepare_inputs(obs)

        infer_start = time.monotonic()
        all_actions = []
        for _ in range(num_samples):
            actions_np = self._run_once(state_tokens, prompt, images_tensor)
            sample_out = self._postprocess(state_norm, actions_np)
            all_actions.append(sample_out["actions"])
        infer_ms = (time.monotonic() - infer_start) * 1000

        total_ms = (time.monotonic() - start) * 1000
        logger.debug(
            f"Batch({num_samples}) policy time: {total_ms:.1f}ms "
            f"(model: {infer_ms:.1f}ms, ~{infer_ms / num_samples:.1f}ms/sample)"
        )

        return {
            "actions": np.stack(all_actions, axis=0),
            "policy_timing": {"infer_ms": infer_ms},
        }


# ---------------------------------------------------------------------------
# Checkpoint conversion
# ---------------------------------------------------------------------------
def convert_checkpoint(jax_checkpoint_dir: str, output_pkl: str, prompt: str):
    """Convert a JAX checkpoint to the realtime-vla pickle format."""
    from convert_from_jax_pi05 import (
        load_jax_weights,
        convert_weights_pi05,
        prepare_adarms_cond,
        prepare_prompt,
    )

    logger.info(f"Converting JAX checkpoint: {jax_checkpoint_dir} -> {output_pkl}")
    dump_weights = load_jax_weights(jax_checkpoint_dir)
    embedding_weight = dump_weights["PaliGemma"]["llm"]["embedder"]["input_embedding"]["value"]
    embedding_weight_torch = torch.tensor(embedding_weight, dtype=torch.bfloat16, device="cpu")

    with _patched_auto_tokenizer():
        language_embeds, prompt_len = prepare_prompt(prompt, embedding_weight, "unused")

    time_embeds = prepare_adarms_cond(num_steps=10)

    weights = {
        "embedding_weight":                   torch.zeros(257152, 2048,          dtype=torch.bfloat16, device="cpu"),
        "vision_patch_embedding_w":           torch.zeros(14, 14, 3, 1152,      dtype=torch.bfloat16, device="cpu"),
        "vision_patch_embedding_b":           torch.zeros(1152,                 dtype=torch.bfloat16, device="cpu"),
        "vision_position_embedding":          torch.zeros(256, 1152,            dtype=torch.bfloat16, device="cpu"),
        "vision_attn_qkv_w":                  torch.zeros(27, 1152, 3 * 1152,   dtype=torch.bfloat16, device="cpu"),
        "vision_attn_qkv_b":                  torch.zeros(27, 3 * 1152,         dtype=torch.bfloat16, device="cpu"),
        "vision_attn_o_w":                    torch.zeros(27, 1152, 1152,       dtype=torch.bfloat16, device="cpu"),
        "vision_attn_o_b":                    torch.zeros(27, 1152,             dtype=torch.bfloat16, device="cpu"),
        "vision_ffn_up_w":                    torch.zeros(27, 1152, 4304,       dtype=torch.bfloat16, device="cpu"),
        "vision_ffn_up_b":                    torch.zeros(27, 4304,             dtype=torch.bfloat16, device="cpu"),
        "vision_ffn_down_w":                  torch.zeros(27, 4304, 1152,       dtype=torch.bfloat16, device="cpu"),
        "vision_ffn_down_b":                  torch.zeros(27, 1152,             dtype=torch.bfloat16, device="cpu"),
        "vision_pre_attn_norm_w":             torch.zeros(27, 1152,             dtype=torch.bfloat16, device="cpu"),
        "vision_pre_attn_norm_b":             torch.zeros(27, 1152,             dtype=torch.bfloat16, device="cpu"),
        "vision_pre_ffn_norm_w":              torch.zeros(27, 1152,             dtype=torch.bfloat16, device="cpu"),
        "vision_pre_ffn_norm_b":              torch.zeros(27, 1152,             dtype=torch.bfloat16, device="cpu"),
        "vision_final_norm_w":                torch.zeros(1152,                 dtype=torch.bfloat16, device="cpu"),
        "vision_final_norm_b":                torch.zeros(1152,                 dtype=torch.bfloat16, device="cpu"),
        "encoder_multi_modal_projector_w":    torch.zeros(1152, 2048,           dtype=torch.bfloat16, device="cpu"),
        "encoder_multi_modal_projector_b":    torch.zeros(2048,                 dtype=torch.bfloat16, device="cpu"),
        "encoder_attn_qkv_w":                 torch.zeros(18, 2048, 2560,       dtype=torch.bfloat16, device="cpu"),
        "encoder_attn_o_w":                   torch.zeros(18, 2048, 2048,       dtype=torch.bfloat16, device="cpu"),
        "encoder_ffn_gate_w":                 torch.zeros(18, 2048, 16384,      dtype=torch.bfloat16, device="cpu"),
        "encoder_ffn_up_w":                   torch.zeros(18, 2048, 16384,      dtype=torch.bfloat16, device="cpu"),
        "encoder_ffn_down_w":                 torch.zeros(18, 16384, 2048,      dtype=torch.bfloat16, device="cpu"),
        "decoder_time_embeds":                torch.zeros(10, 1024,             dtype=torch.bfloat16, device="cpu"),
        "decoder_time_mlp_in_w":              torch.zeros(1024, 1024,           dtype=torch.bfloat16, device="cpu"),
        "decoder_time_mlp_in_b":              torch.zeros(1024,                 dtype=torch.bfloat16, device="cpu"),
        "decoder_time_mlp_out_w":             torch.zeros(1024, 1024,           dtype=torch.bfloat16, device="cpu"),
        "decoder_time_mlp_out_b":             torch.zeros(1024,                 dtype=torch.bfloat16, device="cpu"),
        "decoder_pre_attn_norm_mod_w":        torch.zeros(18, 1024, 3 * 1024,   dtype=torch.bfloat16, device="cpu"),
        "decoder_pre_attn_norm_mod_b":        torch.zeros(18, 3 * 1024,         dtype=torch.bfloat16, device="cpu"),
        "decoder_pre_ffn_norm_mod_w":         torch.zeros(18, 1024, 3 * 1024,   dtype=torch.bfloat16, device="cpu"),
        "decoder_pre_ffn_norm_mod_b":         torch.zeros(18, 3 * 1024,         dtype=torch.bfloat16, device="cpu"),
        "decoder_final_norm_mod_w":           torch.zeros(1024, 3 * 1024,       dtype=torch.bfloat16, device="cpu"),
        "decoder_final_norm_mod_b":           torch.zeros(3 * 1024,             dtype=torch.bfloat16, device="cpu"),
        "decoder_attn_qkv_w":                 torch.zeros(18, 1024, 2560,       dtype=torch.bfloat16, device="cpu"),
        "decoder_attn_o_w":                   torch.zeros(18, 2048, 1024,       dtype=torch.bfloat16, device="cpu"),
        "decoder_ffn_gate_w":                 torch.zeros(18, 1024, 4096,       dtype=torch.bfloat16, device="cpu"),
        "decoder_ffn_up_w":                   torch.zeros(18, 1024, 4096,       dtype=torch.bfloat16, device="cpu"),
        "decoder_ffn_down_w":                 torch.zeros(18, 4096, 1024,       dtype=torch.bfloat16, device="cpu"),
        "decoder_action_in_proj_w":           torch.zeros(32, 1024,             dtype=torch.bfloat16, device="cpu"),
        "decoder_action_in_proj_b":           torch.zeros(1024,                 dtype=torch.bfloat16, device="cpu"),
        "decoder_action_out_proj_w":          torch.zeros(1024, 32,             dtype=torch.bfloat16, device="cpu"),
        "decoder_action_out_proj_b":          torch.zeros(32,                   dtype=torch.bfloat16, device="cpu"),
        "language_embeds":                    torch.zeros(prompt_len, 2048,     dtype=torch.bfloat16, device="cpu"),
    }

    convert_weights_pi05(weights, dump_weights)
    weights["embedding_weight"].copy_(embedding_weight_torch.cpu())
    weights["language_embeds"].copy_(language_embeds)
    weights["decoder_time_embeds"].copy_(time_embeds.cpu())

    os.makedirs(os.path.dirname(output_pkl) or ".", exist_ok=True)
    with open(output_pkl, "wb") as f:
        pickle.dump(weights, f)
    logger.info(f"Saved converted checkpoint to {output_pkl}")
    return weights


# ---------------------------------------------------------------------------
# Build the policy from config + checkpoint
# ---------------------------------------------------------------------------
def build_policy(
    config_name: str,
    checkpoint_dir: str,
    default_prompt: str | None,
    num_views: int,
    chunk_size: int,
) -> tuple[FastTritonPolicy, dict]:
    """Build a FastTritonPolicy from an OpenPI config and checkpoint."""
    from pi05_infer import Pi05Inference

    train_config = _config.get_config(config_name)
    checkpoint_dir = pathlib.Path(_download.maybe_download(checkpoint_dir))

    # -- Load norm stats --
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    norm_stats = _checkpoints.load_norm_stats(
        checkpoint_dir / "assets", data_config.asset_id
    )

    # -- Resolve prompt --
    if default_prompt is None:
        if hasattr(train_config.data, "default_prompt"):
            default_prompt = train_config.data.default_prompt
        else:
            default_prompt = ""

    # -- Convert or load checkpoint --
    pkl_path = checkpoint_dir / "realtime_vla_checkpoint.pkl"
    if pkl_path.exists():
        logger.info(f"Loading pre-converted checkpoint: {pkl_path}")
        with open(pkl_path, "rb") as f:
            checkpoint = pickle.load(f)
    else:
        logger.info("No pre-converted checkpoint found, converting from JAX...")
        checkpoint = convert_checkpoint(
            str(checkpoint_dir), str(pkl_path), default_prompt
        )

    # -- Build Pi05Inference --
    # Patch AutoTokenizer.from_pretrained so Pi05Inference picks up OpenPI's
    # auto-downloaded PaliGemma SentencePiece model instead of needing a path.
    logger.info(
        f"Initializing Pi05Inference (views={num_views}, chunk={chunk_size})..."
    )
    with _patched_auto_tokenizer():
        pi05 = Pi05Inference(
            checkpoint=checkpoint,
            num_views=num_views,
            chunk_size=chunk_size,
            tokenizer_path="unused",
            discrete_state_input=True,
            max_prompt_text=default_prompt,
            state_dim_for_max_prompt=32,
        )
    logger.info("Pi05Inference ready (CUDA graph captured)")

    # -- Build transform chain --
    # Skip data_config.repack_transforms.inputs — those include training-time
    # keys like "actions" that don't exist in inference observations.
    input_transforms = [
        _transforms.InjectDefaultPrompt(default_prompt),
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        _transforms.ResizeImages(224, 224),
    ]

    output_transforms = [
        _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.data_transforms.outputs,
    ]

    policy = FastTritonPolicy(
        pi05_infer=pi05,
        input_transforms=input_transforms,
        output_transforms=output_transforms,
        num_views=num_views,
        default_prompt=default_prompt,
        metadata=train_config.policy_metadata,
    )

    return policy, train_config.policy_metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Serve pi0.5 with realtime-vla Triton backend"
    )
    parser.add_argument(
        "--config", required=True,
        help="OpenPI training config name (e.g. pi05_all_handover_derisk)",
    )
    parser.add_argument(
        "--checkpoint-dir", required=True,
        help="Path to OpenPI JAX checkpoint directory",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--default-prompt", type=str, default=None,
        help="Override the default task prompt",
    )
    parser.add_argument(
        "--num-views", type=int, default=2,
        help="Number of active camera views (default: 2 for collab/xArm)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=16,
        help="Action chunk size / prediction horizon (default: 16)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    policy, metadata = build_policy(
        config_name=args.config,
        checkpoint_dir=args.checkpoint_dir,
        default_prompt=args.default_prompt,
        num_views=args.num_views,
        chunk_size=args.chunk_size,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logger.info(f"Creating server (host: {hostname}, ip: {local_ip})")

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=metadata,
    )
    logger.info(f"Serving on port {args.port} with realtime-vla Triton backend")
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
