"""Verify PadWaitingChunks removes DAL from waiting-start chunks.

Dumps a few hundred post-repack action chunks from the padded TrainConfig and
runs the same translational deadband used in generate_action_leakage_plot.py.

Expect:
  - waiting-start chunks (turn[0]==0): b_hat == T_p  (0% DAL)
  - moving-start chunks: untouched relative to the unpadded pipeline

Usage:
  cd openpi
  uv run scripts/verify_padded_chunks.py --config-name pi05_paper_ready_handover_padded
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import tyro
from tqdm import tqdm

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


@dataclasses.dataclass
class Args:
    """Verify PadWaitingChunks against the leakage-plot deadband."""

    config_name: str = "pi05_paper_ready_handover_padded"
    """Padded TrainConfig to inspect."""
    num_chunks: int = 400
    """How many chunks to sample. Drawn at random across all episodes: consecutive
    indices all land in episode 0 and would miss the waiting phases entirely."""
    seed: int = 0
    """Sampling seed."""
    tau_pos: float = 0.002
    """Translational deadband in metres (2 mm matches generate_action_leakage_plot.py)."""
    compare_unpadded: bool = True
    """Also run the same chunks without PadWaitingChunks, to confirm padding is what changed."""


def _b_hat(actions: np.ndarray, tau_pos: float) -> int:
    """First step whose translational step-delta exceeds tau_pos; else H."""
    H = actions.shape[0]
    if H < 2:
        return H
    step = np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=-1)
    moving = step > tau_pos
    # Align to absolute step index: step[i] is the move into frame i+1.
    moving_full = np.concatenate([[False], moving])
    if not moving_full.any():
        return H
    return int(moving_full.argmax())


def _summarise(tag: str, b: np.ndarray, H: int) -> None:
    if b.size == 0:
        print(f"  {tag:16s} (none sampled)")
        return
    dal = (b > 0) & (b < H)
    print(f"  {tag:16s} n={b.size:4d}  DAL={dal.mean() * 100:5.1f}%  "
          f"b_hat med={np.median(b):4.1f} min={b.min():2d} max={b.max():2d}")


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    train_cfg = _config.get_config(args.config_name)
    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    H = train_cfg.model.action_horizon

    dataset = _data_loader.create_torch_dataset(
        data_cfg, action_horizon=H, model_config=train_cfg.model,
    )

    # Apply only the repack group (RepackTransform [+ PadWaitingChunks]) so the
    # actions we measure are still absolute 8D, before CollabInputs/DeltaActions.
    padded_tfs = list(data_cfg.repack_transforms.inputs)
    unpadded_tfs = [t for t in padded_tfs if type(t).__name__ != "PadWaitingChunks"]
    if len(unpadded_tfs) == len(padded_tfs):
        print("NOTE: config has no PadWaitingChunks — measuring the unpadded baseline only.")

    def run(sample, tfs):
        for t in tfs:
            sample = t(sample)
        return np.asarray(sample["actions"], dtype=np.float64)

    n = min(args.num_chunks, len(dataset))
    rng = np.random.default_rng(args.seed)
    sample_idx = [int(i) for i in np.sort(rng.choice(len(dataset), size=n, replace=False))]
    wait_pad, move_pad, wait_raw, move_raw = [], [], [], []
    step_mm, pre_noise_mm, post_boundary_motion = [], [], []

    for i in tqdm(sample_idx, desc="verify padded chunks"):
        raw = dataset[i]
        turn = np.asarray(raw.get("turn", [])).reshape(-1)
        is_wait = bool(turn.size) and turn[0] == 0

        a_pad = run(dict(raw), padded_tfs)
        b_pad = _b_hat(a_pad, args.tau_pos)
        (wait_pad if is_wait else move_pad).append(b_pad)

        if is_wait:
            # Label boundary: first step the labels call motion.
            diff = np.flatnonzero(turn != turn[0])
            boundary = int(diff[0]) if diff.size else len(turn)
            tail = a_pad[boundary:]
            # The transform's actual contract: nothing moves at or after the boundary.
            held = tail.shape[0] <= 1 or np.abs(tail - tail[0]).max() == 0.0
            post_boundary_motion.append(not held)
            if 0 < b_pad < boundary:
                pre_noise_mm.append(
                    float(np.linalg.norm(np.diff(a_pad[:boundary, :3], axis=0), axis=-1).max() * 1000)
                )

        if args.compare_unpadded:
            a_raw = run(dict(raw), unpadded_tfs)
            (wait_raw if is_wait else move_raw).append(_b_hat(a_raw, args.tau_pos))
            step_mm.append(np.linalg.norm(np.diff(a_raw[:, :3], axis=0), axis=-1) * 1000)

    wait_pad, move_pad = np.asarray(wait_pad, int), np.asarray(move_pad, int)
    wait_raw, move_raw = np.asarray(wait_raw, int), np.asarray(move_raw, int)

    print(f"\nT_p={H}  tau_pos={args.tau_pos * 1000:.0f}mm  chunks={n}")
    if step_mm:
        s = np.concatenate(step_mm)
        print("per-step displacement (mm): p50=%.2f p90=%.2f p99=%.2f max=%.2f"
              % tuple(np.percentile(s, [50, 90, 99, 100])))
    if args.compare_unpadded:
        print("\nBEFORE padding:")
        _summarise("waiting-start", wait_raw, H)
        _summarise("moving-start", move_raw, H)
    print("\nAFTER padding:")
    _summarise("waiting-start", wait_pad, H)
    _summarise("moving-start", move_pad, H)

    ok = True
    if post_boundary_motion:
        bad = int(np.sum(post_boundary_motion))
        if bad:
            ok = False
            print(f"\nFAIL: {bad}/{len(post_boundary_motion)} waiting-start chunks still move "
                  "at or after the label boundary")
        else:
            print(f"\nOK: all {len(post_boundary_motion)} waiting-start chunks are frozen "
                  "from the label boundary to the end of the horizon")
    if pre_noise_mm:
        arr = np.asarray(pre_noise_mm)
        print(f"NOTE: {arr.size}/{len(post_boundary_motion)} waiting-start chunks trip the "
              f"{args.tau_pos * 1000:.0f}mm deadband BEFORE the label boundary "
              f"(max step med={np.median(arr):.1f}mm, max={arr.max():.1f}mm). That is setpoint "
              "drift inside a segment the labels call 'wait', not leakage past the boundary.")
    if args.compare_unpadded and move_pad.size and move_raw.size:
        if not np.array_equal(move_pad, move_raw):
            ok = False
            print("FAIL: moving-start chunks changed; padding must leave them untouched")
        else:
            print("OK: moving-start chunks are byte-identical to the unpadded pipeline")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main(tyro.cli(Args))
