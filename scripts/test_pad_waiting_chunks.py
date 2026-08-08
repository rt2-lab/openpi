"""Unit test for PadWaitingChunks (no dataset / GPU required)."""

import numpy as np

from openpi.transforms import PadWaitingChunks


def test_waiting_chunk_is_padded():
    pad = PadWaitingChunks()
    actions = np.arange(16 * 8, dtype=np.float32).reshape(16, 8)
    turn = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    out = pad({"actions": actions.copy(), "turn": turn})
    assert "turn" not in out
    # hold action is the last waiting frame (index 4)
    hold = actions[4]
    np.testing.assert_array_equal(out["actions"][:5], actions[:5])
    for i in range(5, 16):
        np.testing.assert_array_equal(out["actions"][i], hold)


def test_moving_chunk_untouched():
    pad = PadWaitingChunks()
    actions = np.arange(16 * 8, dtype=np.float32).reshape(16, 8)
    turn = np.ones(16, dtype=np.float32)
    out = pad({"actions": actions.copy(), "turn": turn})
    np.testing.assert_array_equal(out["actions"], actions)


def test_inference_noop_without_turn():
    pad = PadWaitingChunks()
    actions = np.ones((16, 8), dtype=np.float32)
    out = pad({"actions": actions.copy()})
    np.testing.assert_array_equal(out["actions"], actions)


if __name__ == "__main__":
    test_waiting_chunk_is_padded()
    test_moving_chunk_untouched()
    test_inference_noop_without_turn()
    print("PadWaitingChunks OK")
