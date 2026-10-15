"""Reference motion for the G1: a retargeted human clip that loops (motions/<name>.npz).

The npz files come from openhe/g1-retargeted-motions (MIT) via tools/convert_motion.py:
  walk.npz       ACCAD B3 walk1, frames 42-155 (one clean loop, 3.77 s)
  bruce_lee.npz  kungfu Bruce_Lee_pose, 330 frames (11 s)
"""
import numpy as np
import torch

from rl_shortcut import MOTION_DIR

# the 23 joints stored in the clip, in order (legs, waist, arms)
CLIP_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_wrist_roll_joint",
]
# names that differ in the Isaac Lab G1 (g1_minimal.usd has no wrist; elbow roll is the closest)
ROBOT_NAME = {"waist_yaw_joint": "torso_joint",
              "left_wrist_roll_joint": "left_elbow_roll_joint",
              "right_wrist_roll_joint": "right_elbow_roll_joint"}


def quat_yaw(q):
    """Yaw angle of wxyz quaternions."""
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class Motion:
    """Per-frame targets of one looping clip, on the GPU."""

    def __init__(self, name, device):
        d = np.load(MOTION_DIR / f"{name}.npz")
        self.name, self.fps = name, float(d["fps"])
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)  # noqa: E731
        self.dof = t(d["dof"])                       # (T, 23) joint angles
        self.root_pos = t(d["root_pos"])             # (T, 3)
        self.root_quat = t(d["root_quat"])           # (T, 4) wxyz
        self.T = self.dof.shape[0]
        # velocities by finite difference; the last frame copies the one before (no jump at the loop)
        self.dof_vel = self._diff(self.dof)
        self.root_vel = self._diff(self.root_pos)
        yaw = torch.tensor(np.unwrap(quat_yaw(self.root_quat).cpu().numpy()), device=device)
        self.yaw = yaw.float()
        self.yaw_rate = self._diff(self.yaw.unsqueeze(1)).squeeze(1)

    def _diff(self, x):
        v = (torch.roll(x, -1, dims=0) - x) * self.fps
        v[-1] = v[-2]
        return v

    @property
    def duration(self):
        return self.T / self.fps

    def joint_ids(self, robot_joint_names):
        """Index of each clip joint in the robot's joint list."""
        return [robot_joint_names.index(ROBOT_NAME.get(n, n)) for n in CLIP_JOINTS]
