"""Terminal captures for the lecture pages. Outputs copied from the real runs on the course PC (long lines cut).

  python3 docs/lecture/render_terminals.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_terminal import render  # noqa: E402

V = "(.venv) pw@laptop:~/rl_shortcut$ "
LAB = "(.venv) pw@laptop:~/rl_shortcut/IsaacLab$ "


def setup():
    render([("prompt", "mkdir -p ~/rl_shortcut"),
            ("prompt", "cd ~/rl_shortcut"),
            ("prompt", "git clone https://github.com/PinkWink/rl_shortcut.git .", "pw@laptop:~/rl_shortcut$ "),
            ("out", "Cloning into '.'..."),
            ("prompt", "ls", "pw@laptop:~/rl_shortcut$ "),
            ("out", "assets  checkpoints  docs  motions  pyproject.toml  README.md  rl_shortcut  scripts  tools")],
           "l02_t_clone.png", prompt="pw@laptop:~$ ", title="pw@laptop: ~ — bash", highlight=("checkpoints",))
    render([("prompt", "python3.11 -m venv .venv && source .venv/bin/activate"),
            ("prompt", "pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128"),
            ("out", "  Downloading nvidia_cudnn_cu12-9.7.1.26-py3-none-manylinux_2_27_x86_64.whl (726.9 MB)"),
            ("out", "  ..."),
            ("out", "Successfully installed ... torch-2.7.0+cu128 torchvision-0.22.0+cu128 triton-3.3.0 ..."),
            ("prompt", 'pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com'),
            ("out", "  Downloading isaacsim_extscache_kit-5.1.0.0-cp311-none-manylinux_2_35_x86_64.whl (3021.3 MB)"),
            ("out", "  ..."),
            ("out", "Successfully installed ... isaacsim-5.1.0.0 isaacsim-app-5.1.0.0 ... isaacsim-rl-5.1.0.0 ..."),
            ("prompt", "du -sh .venv"),
            ("out", "18G\t.venv")],
           "l02_t_pip.png", prompt=V, highlight=("Successfully installed", "18G"))
    render([("prompt", "./isaaclab.sh --install rsl_rl"),
            ("out", "[INFO] Installing extra requirements such as learning frameworks..."),
            ("out", "[INFO] Installing rl-framework: rsl_rl"),
            ("out", "  ..."),
            ("out", "[INFO] Setting up vscode settings..."),
            ("out", ""),
            ("out", "By installing or using Omniverse Kit, I agree to the terms of NVIDIA OMNIVERSE LICENSE AGREEMENT (EULA)"),
            ("out", "in https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html"),
            ("out", ""),
            ("out", "Do you accept the EULA? (Yes/No): Yes"),
            ("out", "The EULA was accepted."),
            ("out", "[WARN] Could not find Isaac Sim VSCode settings: ... (VSCode 자동 완성만 영향, 무시해도 됨)")],
           "l02_t_isaaclab.png", prompt=LAB, highlight=("Do you accept", "The EULA was accepted"))
    render([("prompt", "python scripts/check_install.py --headless"),
            ("out", "  ... (Isaac Sim 시작 로그) ..."),
            ("out", "isaacsim     5.1.0.0"),
            ("out", "isaaclab     0.54.2"),
            ("out", "rsl-rl-lib   3.1.2"),
            ("out", "torch        2.7.0+cu128"),
            ("out", "GPU          NVIDIA GeForce RTX 5060 Ti, cuda True"),
            ("out", "physics: 100 steps of 0.01 s on cuda:0")],
           "l02_t_check.png", prompt=V, highlight=("cuda True", "physics:"))


def physics():
    render([("prompt", "python scripts/physics_drop.py"),
            ("out", "dt = 0.005 s, drop height 1.0 m, theory: fall 0.452 s"),
            ("out", "restitution 0.0: fall 0.440 s, bounce 0.000 m (theory 0.000 m)"),
            ("out", "restitution 0.5: fall 0.440 s, bounce 0.252 m (theory 0.250 m)"),
            ("out", "restitution 0.9: fall 0.440 s, bounce 0.811 m (theory 0.810 m)")],
           "l03_t_drop.png", prompt=V, highlight=("bounce",))
    render([("prompt", "python scripts/physics_drop.py --headless --dt 0.02"),
            ("out", "dt = 0.02 s, drop height 1.0 m, theory: fall 0.452 s"),
            ("out", "restitution 0.0: fall 0.420 s, bounce 0.000 m (theory 0.000 m)"),
            ("out", "restitution 0.5: fall 0.420 s, bounce 0.241 m (theory 0.250 m)"),
            ("out", "restitution 0.9: fall 0.420 s, bounce 0.807 m (theory 0.810 m)")],
           "l03_t_drop_dt.png", prompt=V, highlight=("fall 0.420", "0.241"))


def pendulum_check():
    render([("prompt", "python scripts/pendulum_check.py --headless"),
            ("out", "joints: ['base_to_arm', 'arm_to_pendulum']"),
            ("out", "links: base 500.0 g, arm 36.0 g, pendulum 16.0 g"),
            ("out", "joint limits (rad): [(-2.36, 2.36), (-3.4028234663852886e+38, 3.4028234663852886e+38)]"),
            ("out", "arm held, swing from 30 deg: period 0.607 s (theory with the arm held 0.597 s), arm moved up to 0.0 deg"),
            ("prompt", "python scripts/pendulum_check.py --headless --free_arm"),
            ("out", "arm free, swing from 30 deg: period 0.491 s (theory with the arm held 0.597 s), arm moved up to 24.7 deg")],
           "l04_t_check.png", prompt=V, highlight=("period 0.607", "period 0.491"))


def pendulum():
    render([("out", "                      Learning iteration 149/150"),
            ("out", "                       Computation: 665389 steps/s (collection: 0.066s, learning 0.032s)"),
            ("out", "             Mean action noise std: 0.00"),
            ("out", "                       Mean reward: 4.98"),
            ("out", "               Mean episode length: 500.00"),
            ("out", "              Episode_Reward/alive: 1.0000"),
            ("out", "       Episode_Reward/pole_upright: -0.0008"),
            ("out", "         Episode_Reward/arm_center: -0.0012"),
            ("out", "      Episode_Termination/time_out: 1.0000"),
            ("out", "     Episode_Termination/pole_fell: 0.0000"),
            ("out", "                    Iteration time: 0.10s"),
            ("out", "                      Time elapsed: 00:00:15"),
            ("out", "[train] done in 0.3 min -> /home/pw/rl_shortcut/logs/pendulum/2026-10-15_19-44-27")],
           "l05_t_train.png", prompt=V, highlight=("Mean episode length", "time_out", "[train] done"))
    render([("prompt", "python scripts/play.py --task pendulum --headless --seconds 5"),
            ("out", "[play] /home/pw/rl_shortcut/logs/pendulum/2026-10-15_19-44-27/model_149.pt"),
            ("out", "[play] t=  1 s  fails=  0  pendulum angle from upright 0.027 rad"),
            ("out", "[play] t=  2 s  fails=  0  pendulum angle from upright 0.003 rad"),
            ("out", "[play] t=  3 s  fails=  0  pendulum angle from upright 0.000 rad"),
            ("out", "[play] t=  4 s  fails=  0  pendulum angle from upright 0.000 rad"),
            ("out", "[play] t=  5 s  fails=  0  pendulum angle from upright 0.001 rad"),
            ("out", "[play] 5 s x 16 envs: fails=0, pendulum angle from upright 0.006 rad")],
           "l05_t_play.png", prompt=V, highlight=("5 s x 16 envs",))


def go2():
    render([("prompt", "python scripts/go2_inspect.py --headless"),
            ("out", "USD: https://omniverse-content-production.s3-us-west-2.amazonaws.com/.../IsaacLab/Robots/Unitree/Go2/go2.usd"),
            ("out", ""),
            ("out", "19 links, total 15.02 kg"),
            ("out", "  base              6.921 kg"),
            ("out", "  FL_hip            0.678 kg"),
            ("out", "  FL_thigh          1.152 kg"),
            ("out", "  FL_calf           0.154 kg"),
            ("out", "  FL_foot           0.040 kg"),
            ("out", "  FR_hip            0.678 kg"),
            ("out", "  ...  (FR · RL · RR 다리도 같음)"),
            ("out", "  Head_upper        0.001 kg"),
            ("out", "  Head_lower        0.001 kg")],
           "l06_t_links.png", prompt=V, highlight=("19 links", "base "))
    render([("out", "12 revolute joints (limits in degrees, from the USD)"),
            ("out", "  FL_hip_joint       -60.0 ..   60.0"),
            ("out", "  FR_hip_joint       -60.0 ..   60.0"),
            ("out", "  RL_hip_joint       -60.0 ..   60.0"),
            ("out", "  RR_hip_joint       -60.0 ..   60.0"),
            ("out", "  FL_thigh_joint     -90.0 ..  200.0"),
            ("out", "  FL_calf_joint     -156.0 ..  -48.0"),
            ("out", "  FR_thigh_joint     -90.0 ..  200.0"),
            ("out", "  FR_calf_joint     -156.0 ..  -48.0"),
            ("out", "  RL_thigh_joint     -30.0 ..  260.0"),
            ("out", "  RL_calf_joint     -156.0 ..  -48.0"),
            ("out", "  RR_thigh_joint     -30.0 ..  260.0"),
            ("out", "  RR_calf_joint     -156.0 ..  -48.0")],
           "l06_t_joints.png", prompt=V, highlight=("thigh_joint",))
    order = [("FL_hip_joint", "+0.10"), ("FR_hip_joint", "-0.10"), ("RL_hip_joint", "+0.10"), ("RR_hip_joint", "-0.10"),
             ("FL_thigh_joint", "+0.80"), ("FR_thigh_joint", "+0.80"), ("RL_thigh_joint", "+1.00"),
             ("RR_thigh_joint", "+1.00"), ("FL_calf_joint", "-1.50"), ("FR_calf_joint", "-1.50"),
             ("RL_calf_joint", "-1.50"), ("RR_calf_joint", "-1.50")]
    render([("out", "Isaac Lab joint order (actions and observations use this order) and the standing pose:")]
           + [("out", f"  {i:2d} {n:16s} {q} rad") for i, (n, q) in enumerate(order)],
           "l06_t_order.png", prompt=V, highlight=("Isaac Lab joint order",))
    render([("prompt", "python scripts/go2_stand.py"),
            ("out", "kp 25.0, kd 0.5"),
            ("out", "t=0.5 s  body height 0.246 m  max joint error 0.613 rad  max torque  14.0 Nm"),
            ("out", "t=1.0 s  body height 0.273 m  max joint error 0.338 rad  max torque   8.4 Nm"),
            ("out", "t=1.5 s  body height 0.284 m  max joint error 0.226 rad  max torque   5.6 Nm"),
            ("out", "t=2.0 s  body height 0.283 m  max joint error 0.239 rad  max torque   6.1 Nm"),
            ("out", "t=2.5 s  body height 0.276 m  max joint error 0.303 rad  max torque   7.5 Nm"),
            ("out", "t=3.0 s  body height 0.283 m  max joint error 0.236 rad  max torque   6.0 Nm")],
           "l06_t_stand.png", prompt=V, highlight=("t=3.0",))
    render([("out", "                      Learning iteration 299/300"),
            ("out", "                       Computation: 183948 steps/s (collection: 0.454s, learning 0.081s)"),
            ("out", "                       Mean reward: 34.95"),
            ("out", "               Mean episode length: 1000.00"),
            ("out", "   Episode_Reward/track_lin_vel_xy: 1.4318"),
            ("out", "    Episode_Reward/track_ang_vel_z: 0.6581"),
            ("out", "      Episode_Reward/feet_air_time: -0.0340"),
            ("out", "      Episode_Reward/joint_torques: -0.0861"),
            ("out", "Metrics/base_velocity/error_vel_xy: 0.1793"),
            ("out", "      Episode_Termination/time_out: 0.9927"),
            ("out", "  Episode_Termination/base_contact: 0.0073"),
            ("out", "                    Iteration time: 0.53s"),
            ("out", "                      Time elapsed: 00:02:43")],
           "l08_t_train.png", prompt=V, highlight=("track_lin_vel_xy", "error_vel_xy", "base_contact"))


def g1_motion():
    render([("prompt", "python scripts/g1_motion.py --motion walk"),
            ("out", "walk: 113 frames @ 30 fps = 3.77 s, pelvis height 0.75..0.84 m, travel 3.98 m"),
            ("prompt", "python scripts/g1_motion.py --motion bruce_lee"),
            ("out", "bruce_lee: 330 frames @ 30 fps = 11.00 s, pelvis height 0.31..0.93 m, travel 0.13 m")],
           "l09_t_motion.png", prompt=V, highlight=("walk:", "bruce_lee:"))


def g1_walk():
    render([("out", "                     Learning iteration 1999/2000"),
            ("out", "                       Mean reward: 140.27"),
            ("out", "               Mean episode length: 600.00"),
            ("out", "             Episode_Reward/joints: 4.3967"),
            ("out", "               Episode_Reward/arms: 1.9727"),
            ("out", "           Episode_Reward/velocity: 1.9203"),
            ("out", "                Episode_Reward/yaw: 1.9827"),
            ("out", "             Episode_Reward/height: 1.3371"),
            ("out", "       Episode_Reward/joint_limits: -0.5911"),
            ("out", "      Episode_Termination/time_out: 1.0000"),
            ("out", "                    Iteration time: 0.78s"),
            ("out", "[train] done in 28.9 min -> /home/pw/rl_shortcut/logs/g1_walk/2026-10-15_16-15-17")],
           "l09_t_train.png", prompt=V, highlight=("joints", "time_out", "[train] done"))
    render([("prompt", "python scripts/play.py --task g1 --motion walk --headless --seconds 12 --num_envs 64"),
            ("out", "[play] /home/pw/rl_shortcut/logs/g1_walk/2026-10-15_16-15-17/model_1999.pt"),
            ("out", "[play] t=  1 s  fails=  0  joint error 0.078 rad"),
            ("out", "[play] t=  2 s  fails=  0  joint error 0.074 rad"),
            ("out", "  ..."),
            ("out", "[play] t= 12 s  fails=  0  joint error 0.073 rad"),
            ("out", "[play] 12 s x 64 envs: fails=0, joint error 0.074 rad")],
           "l09_t_play.png", prompt=V, highlight=("12 s x 64 envs",))
    render([("prompt", "python scripts/play.py --task g1 --motion bruce_lee --checkpoint g1_walk --headless --seconds 12 --num_envs 64"),
            ("out", "[play] /home/pw/rl_shortcut/logs/g1_walk/2026-10-15_16-15-17/model_1999.pt"),
            ("out", "[play] t=  1 s  fails= 62  joint error 0.437 rad"),
            ("out", "[play] t=  2 s  fails=137  joint error 0.449 rad"),
            ("out", "[play] t=  3 s  fails=216  joint error 0.450 rad"),
            ("out", "  ..."),
            ("out", "[play] t= 12 s  fails=930  joint error 0.436 rad"),
            ("out", "[play] 12 s x 64 envs: fails=930, joint error 0.441 rad")],
           "l10_t_walk_on_bruce.png", prompt=V, highlight=("12 s x 64 envs",))


def g1_bruce():
    render([("out", "                     Learning iteration 2999/3000"),
            ("out", "                       Mean reward: 106.88"),
            ("out", "               Mean episode length: 597.63"),
            ("out", "             Episode_Reward/joints: 3.1277"),
            ("out", "               Episode_Reward/arms: 1.0370"),
            ("out", "           Episode_Reward/velocity: 1.8272"),
            ("out", "                Episode_Reward/yaw: 1.9582"),
            ("out", "             Episode_Reward/height: 1.2019"),
            ("out", "      Episode_Termination/time_out: 0.9898"),
            ("out", "        Episode_Termination/tilted: 0.0078"),
            ("out", "                    Iteration time: 0.79s"),
            ("out", "[train] done in 43.6 min -> /home/pw/rl_shortcut/logs/g1_bruce_lee/2026-10-15_16-44-17")],
           "l10_t_train.png", prompt=V, highlight=("joints", "time_out", "[train] done"))
    render([("prompt", "python scripts/play.py --task g1 --motion bruce_lee --headless --seconds 12 --num_envs 64"),
            ("out", "[play] /home/pw/rl_shortcut/logs/g1_bruce_lee/2026-10-15_16-44-17/model_2999.pt"),
            ("out", "[play] t=  1 s  fails=  0  joint error 0.191 rad"),
            ("out", "[play] t=  2 s  fails=  0  joint error 0.175 rad"),
            ("out", "[play] t=  3 s  fails=  0  joint error 0.182 rad"),
            ("out", "  ..."),
            ("out", "[play] 12 s x 64 envs: fails=0, joint error 0.198 rad"),
            ("prompt", "python scripts/play.py --task g1 --motion walk --checkpoint g1_bruce_lee --headless --seconds 12 --num_envs 64"),
            ("out", "  ..."),
            ("out", "[play] 12 s x 64 envs: fails=967, joint error 0.259 rad")],
           "l10_t_play_bruce.png", prompt=V, highlight=("12 s x 64 envs",))


if __name__ == "__main__":
    for f in (setup, physics, pendulum_check, pendulum, go2, g1_motion, g1_walk, g1_bruce):
        f()
