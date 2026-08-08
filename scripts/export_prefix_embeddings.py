"""Export mean-pooled pi0.5 prefix embeddings + turn labels for wait/go training.

Loads a frozen OpenPI checkpoint and the turn-labelled LeRobot dataset, runs
``_encode_prefix`` per frame, mean-pools over the prefix mask, and writes:

  embeddings.npy     (N, D) float32
  turn.npy           (N,) uint8   1=move, 0=wait
  episode_index.npy  (N,) int32
  meta.json          {emb_dim, n, checkpoint, config, repo_id, move_frac}

The policy is built from the *unmodified* train config so it loads the norm
stats baked into the checkpoint; only the dataset is repointed at the
turn-labelled repo (asset_id is pinned to the original so lookup still works).

Usage:
  cd openpi
  uv run scripts/export_prefix_embeddings.py \
      --config-name pi05_paper_ready_handover \
      --checkpoint-dir checkpoints/pi05_paper_ready_handover/24999 \
      --repo-id local/paper_ready_handover_turn \
      --out-dir assets/rebuttal_handover_gate
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import tyro

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import nnx_utils
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


@dataclasses.dataclass
class Args:
    config_name: str = "pi05_paper_ready_handover"
    checkpoint_dir: str = "checkpoints/pi05_paper_ready_handover/24999"
    # Turn-labelled LeRobot dataset. Must contain the per-frame `turn` feature.
    repo_id: str = "local/paper_ready_handover_turn"
    out_dir: str = "assets/rebuttal_handover_gate"
    # Take every Nth frame (waiting frames are highly redundant at 10 Hz).
    stride: int = 1
    max_frames: int | None = None
    # Background workers for dataset reads (~34 ms/sample of PNG decode) so they
    # overlap with the ~44 ms prefix encode on the GPU.
    num_workers: int = 4


def _mean_pool_jax(prefix_out, prefix_mask):
    """Mean-pool (B, S, D) over valid prefix tokens -> (B, D), on device.

    Pooling before the host transfer moves ~8 MB/sample of prefix tokens down to
    8 KB, and must match Policy._gate_prob_from_prefix exactly.
    """
    mask = prefix_mask.astype(jnp.float32)[..., None]
    return (prefix_out.astype(jnp.float32) * mask).sum(axis=1) / jnp.maximum(mask.sum(axis=1), 1.0)


def _scalar(x, default=-1) -> int:
    if x is None:
        return default
    return int(np.asarray(x).reshape(-1)[0])


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    train_cfg = _config.get_config(args.config_name)

    # Policy from the untouched config so checkpoint norm stats resolve.
    policy = _policy_config.create_trained_policy(
        train_cfg, args.checkpoint_dir, tabulate_adarms=False,
    )
    encode_prefix = nnx_utils.module_jit(policy._model._encode_prefix)

    # Dataset from the turn-labelled repo, keeping the original asset_id.
    ds_factory = dataclasses.replace(
        train_cfg.data,
        repo_id=args.repo_id,
        assets=_config.AssetsConfig(asset_id=train_cfg.data.repo_id),
    )
    ds_cfg = ds_factory.create(train_cfg.assets_dirs, train_cfg.model)
    dataset = _data_loader.create_torch_dataset(
        ds_cfg, action_horizon=1, model_config=train_cfg.model,
    )
    # Repack maps raw dataset keys -> the observation/* keys CollabInputs wants.
    # PadWaitingChunks (if present) is harmless here; we read `turn` beforehand.
    repack = list(ds_cfg.repack_transforms.inputs)
    logging.info("Dataset size: %d  repack: %s",
                 len(dataset), [type(t).__name__ for t in repack])

    indices = list(range(0, len(dataset), max(1, args.stride)))
    if args.max_frames is not None:
        indices = indices[: args.max_frames]

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.num_workers > 0:
        import torch.utils.data as _tud

        stream = _tud.DataLoader(
            _tud.Subset(dataset, indices),
            batch_size=None,          # yield individual samples, no collation
            num_workers=args.num_workers,
            shuffle=False,
        )
    else:
        stream = (dataset[i] for i in indices)

    embs, turns, ep_idxs = [], [], []
    for sample in tqdm(stream, total=len(indices), desc="export embeddings"):
        if "turn" not in sample:
            raise KeyError(
                "Sample is missing 'turn'. Rebuild the LeRobot dataset with the "
                "updated convert_collab_data_to_lerobot.py."
            )
        turn_scalar = _scalar(sample["turn"])
        ep_scalar = _scalar(sample.get("episode_index"), default=-1)

        packed = dict(sample)
        for t in repack:
            packed = t(packed)
        inputs = policy._input_transform(packed)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs)

        prefix_out, prefix_mask, _kv = encode_prefix(observation)
        pooled = np.asarray(_mean_pool_jax(prefix_out, prefix_mask), dtype=np.float32)[0]

        embs.append(pooled.astype(np.float32))
        turns.append(turn_scalar)
        ep_idxs.append(ep_scalar)

    emb_arr = np.stack(embs, axis=0)
    turn_arr = np.asarray(turns, dtype=np.uint8)
    ep_arr = np.asarray(ep_idxs, dtype=np.int32)

    np.save(out_dir / "embeddings.npy", emb_arr)
    np.save(out_dir / "turn.npy", turn_arr)
    np.save(out_dir / "episode_index.npy", ep_arr)
    meta = {
        "emb_dim": int(emb_arr.shape[1]),
        "n": int(emb_arr.shape[0]),
        "stride": args.stride,
        "checkpoint": args.checkpoint_dir,
        "config": args.config_name,
        "repo_id": args.repo_id,
        "move_frac": float(turn_arr.mean()),
        "n_episodes": int(len(np.unique(ep_arr))),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logging.info("Wrote %s  shape=%s  move_frac=%.3f  episodes=%d",
                 out_dir, emb_arr.shape, meta["move_frac"], meta["n_episodes"])


if __name__ == "__main__":
    main(tyro.cli(Args))
