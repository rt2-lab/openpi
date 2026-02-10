"""Convert Collab (xArm) HDF5 episodes to a LeRobot dataset for OpenPI training.

Each episode lives at <db_dir>/ep<NNNNNN>/episode.h5 and contains:
  - timestamps            (N,)      float64
  - robot_current_pose    (N, 7)    float32   [x, y, z, qx, qy, qz, qw]  (meters)
  - robot_desired_pose    (N, 7)    float32   same layout
  - gripper_actual_position   (N,)  float32
  - gripper_commanded_position (N,) float32
  - gripper_image_rgb_compressed  (N,) vlen uint8  (JPEG)
  - mount_image_rgb_compressed    (N,) vlen uint8  (JPEG)
  - state                 (N,)      string    turn-taking label

Usage:
  uv run examples/collab/convert_collab_data_to_lerobot.py \
      --db_dir /home/leo/datasets/new_collab \
      --repo_name your_hf_user/collab \
      --fps 10 \
      --default_task "collaborative handover task"
"""

import glob
import os
import shutil

import cv2
import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
import tyro


def main(
    db_dir: str,
    repo_name: str = "local/collab",
    fps: int = 10,
    downsample_factor: int = 2,
    image_size: tuple[int, int] = (224, 224),
    default_task: str = "perform the collaborative task",
    push_to_hub: bool = False,
):
    """
    Args:
        db_dir: Path to directory containing ep*/episode.h5 files.
        repo_name: LeRobot repo id (local or HuggingFace).
        fps: Target frames per second (after downsampling).
        downsample_factor: Take every Nth frame from the raw data.
        image_size: (width, height) to resize images to.
        default_task: Language instruction to attach to every episode.
        push_to_hub: Whether to push the dataset to HuggingFace Hub.
    """
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    episode_paths = sorted(glob.glob(os.path.join(db_dir, "ep*/episode.h5")))
    assert len(episode_paths) > 0, f"No episodes found in {db_dir}"
    print(f"Found {len(episode_paths)} episodes in {db_dir}")

    # State: 7D pose + 1D gripper = 8D
    # Action: 7D desired pose + 1D gripper cmd = 8D
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="xarm6",
        fps=fps,
        features={
            "mount_image": {
                "dtype": "image",
                "shape": (image_size[1], image_size[0], 3),
                "names": ["height", "width", "channel"],
            },
            "gripper_image": {
                "dtype": "image",
                "shape": (image_size[1], image_size[0], 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for ep_path in episode_paths:
        print(f"Processing {ep_path}...")
        with h5py.File(ep_path, "r") as f:
            ep_len = len(f["timestamps"])
            frame_indices = list(range(0, ep_len, downsample_factor))

            robot_pose = np.array(f["robot_current_pose"][:], dtype=np.float32)
            desired_pose = np.array(f["robot_desired_pose"][:], dtype=np.float32)
            gripper_act = np.array(f["gripper_actual_position"][:], dtype=np.float32)
            gripper_cmd = np.array(f["gripper_commanded_position"][:], dtype=np.float32)

            if gripper_act.ndim == 1:
                gripper_act = gripper_act[:, np.newaxis]
            if gripper_cmd.ndim == 1:
                gripper_cmd = gripper_cmd[:, np.newaxis]

            # State = current pose (7D) + gripper actual (1D) = 8D
            states = np.concatenate([robot_pose, gripper_act], axis=1)
            # Action = desired pose (7D) + gripper cmd (1D) = 8D
            actions = np.concatenate([desired_pose, gripper_cmd], axis=1)

            for frame_idx in frame_indices:
                # Decode gripper image
                gripper_jpeg = np.array(f["gripper_image_rgb_compressed"][frame_idx])
                gripper_img = cv2.imdecode(gripper_jpeg, cv2.IMREAD_COLOR)
                gripper_img = cv2.cvtColor(gripper_img, cv2.COLOR_BGR2RGB)
                gripper_img = cv2.resize(gripper_img, image_size)

                # Decode mount image
                mount_jpeg = np.array(f["mount_image_rgb_compressed"][frame_idx])
                mount_img = cv2.imdecode(mount_jpeg, cv2.IMREAD_COLOR)
                mount_img = cv2.cvtColor(mount_img, cv2.COLOR_BGR2RGB)
                mount_img = cv2.resize(mount_img, image_size)

                dataset.add_frame(
                    {
                        "mount_image": mount_img,
                        "gripper_image": gripper_img,
                        "state": states[frame_idx],
                        "actions": actions[frame_idx],
                        "task": default_task,
                    }
                )

        dataset.save_episode()

    print(f"Dataset saved to {output_path}")
    print(f"Total episodes: {len(episode_paths)}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["collab", "xarm6", "pick-and-place"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
