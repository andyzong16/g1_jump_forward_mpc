"""Convert a WBC motion npz file back to the CSV format used by convert_csv_to_npz.py.

The output CSV columns are:

    base_pos_xyz, base_quat_xyzw, joint_positions

.. code-block:: bash

    python source/isaaclab_tasks/isaaclab_tasks/manager_based/wbc/data/motions/npz/convert_npz_to_csv.py \
        -f source/isaaclab_tasks/isaaclab_tasks/manager_based/wbc/data/motions/npz/g1_jump_forward_traj.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternions from Isaac Lab's wxyz order to the CSV xyzw order."""

    return quat_wxyz[..., [1, 2, 3, 0]]


def convert_npz_to_csv(input_file: str, output_file: str | None = None, root_body_index: int = 0):
    """Convert a motion npz file to CSV.

    If the npz has a ``qpos`` array, this uses it directly because it contains
    floating-base pose plus joint positions. Otherwise, it reconstructs the CSV
    from ``body_pos_w``, ``body_quat_w``, and ``joint_pos``.
    """

    input_path = Path(input_file)
    output_path = Path(output_file) if output_file else input_path.with_suffix(".csv")

    data = np.load(input_path)

    if "qpos" in data:
        qpos = data["qpos"]
        if qpos.ndim != 2 or qpos.shape[1] < 8:
            raise ValueError(f"Expected qpos with shape (frames, >=8), got {qpos.shape}")

        base_pos = qpos[:, :3]
        base_quat_xyzw = _quat_wxyz_to_xyzw(qpos[:, 3:7])
        joint_pos = qpos[:, 7:]
    else:
        required_keys = ("body_pos_w", "body_quat_w", "joint_pos")
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise KeyError(f"Motion file is missing required keys: {missing_keys}")

        body_pos_w = data["body_pos_w"]
        body_quat_w = data["body_quat_w"]
        joint_pos = data["joint_pos"]

        if root_body_index >= body_pos_w.shape[1]:
            raise ValueError(f"root_body_index={root_body_index} is out of range for {body_pos_w.shape[1]} bodies")

        base_pos = body_pos_w[:, root_body_index]
        base_quat_xyzw = _quat_wxyz_to_xyzw(body_quat_w[:, root_body_index])

    csv_motion = np.concatenate([base_pos, base_quat_xyzw, joint_pos], axis=1)
    np.savetxt(output_path, csv_motion, delimiter=",")

    print(f"[INFO]: Loaded {input_path}")
    print(f"[INFO]: Wrote {output_path}")
    print(f"[INFO]: CSV shape: {csv_motion.shape}")


def main():
    parser = argparse.ArgumentParser(description="Convert a WBC motion npz file to CSV.")
    parser.add_argument("--input_file", "-f", type=str, required=True, help="Path to the input npz file.")
    parser.add_argument("--output_file", "-o", type=str, default=None, help="Path to the output CSV file.")
    parser.add_argument(
        "--root_body_index",
        type=int,
        default=0,
        help="Root body index to use when qpos is not available in the npz file.",
    )
    args = parser.parse_args()

    convert_npz_to_csv(args.input_file, args.output_file, args.root_body_index)


if __name__ == "__main__":
    main()