"""Inputs/Outputs mapping for the Collab (xArm) pick-and-place dataset.

Your dataset has:
  - state: 8D  (xyz position + quaternion + gripper_actual_position)
  - action: 8D (xyz position + quaternion + gripper_commanded_position)
  - images: mount camera (third-person) + gripper camera (wrist)

The model expects three image slots (base, left_wrist, right_wrist).
We map mount → base, gripper → left_wrist, and zero-pad right_wrist.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_collab_example() -> dict:
    """Creates a random input example for testing the Collab policy."""
    return {
        "observation/state": np.random.rand(8).astype(np.float32),
        "observation/mount_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/gripper_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the object",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    # LeRobot stores as (C, H, W), model expects (H, W, C)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class CollabInputs(transforms.DataTransformFn):
    """Maps Collab dataset observations → model input format.

    Used during both training (on LeRobot data) and inference (from policy server).
    The key names here must match what the repack_transform produces (training) or
    what your inference client sends (deployment).
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Mount camera = third-person view → base_0_rgb
        mount_image = _parse_image(data["observation/mount_image"])
        # Gripper camera = wrist view → left_wrist_0_rgb
        gripper_image = _parse_image(data["observation/gripper_image"])

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": mount_image,
                "left_wrist_0_rgb": gripper_image,
                "right_wrist_0_rgb": np.zeros_like(mount_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class CollabOutputs(transforms.DataTransformFn):
    """Maps model output actions back to dataset format.

    Action dim = 8: (xyz + quaternion + gripper).
    The model pads actions to its internal dim, so we slice back to 8.
    """

    # 7D pose + 1D gripper = 8D
    action_dim: int = 8

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
