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
    render([("prompt", "sudo apt update && sudo apt install -y git"),
            ("out", "  ..."),
            ("prompt", "git clone https://github.com/PinkWink/rl_shortcut.git"),
            ("out", "Cloning into 'rl_shortcut'..."),
            ("prompt", "cd rl_shortcut"),
            ("prompt", "ls", "pw@laptop:~/rl_shortcut$ "),
            ("out", "assets  checkpoints  constraints.txt  docs  motions  pyproject.toml  README.md  rl_shortcut  scripts  tools")],
           "l02_t_clone.png", prompt="pw@laptop:~$ ", title="pw@laptop: ~ — bash", highlight=("checkpoints",))
    render([("prompt", "./scripts/install.sh"),
            ("out", ""),
            ("out", "== 시스템 확인"),
            ("out", "  GPU: NVIDIA GeForce RTX 5060 Ti, 580.196.07"),
            ("out", ""),
            ("out", "== apt 패키지 (Python 3.11)"),
            ("out", "  설치합니다: python3.11 python3.11-venv python3.11-dev cmake build-essential  (sudo 비밀번호를 물을 수 있습니다)"),
            ("out", "  ... (가상환경 · PyTorch · Isaac Sim · Isaac Lab · 이 과정의 패키지) ..."),
            ("out", ""),
            ("out", "== 버전 확인"),
            ("out", "  torch        2.7.0"),
            ("out", "  numpy        1.26.0"),
            ("out", "  tensordict   0.14.3"),
            ("out", "  rsl-rl-lib   3.1.2"),
            ("out", "  isaacsim     5.1.0.0"),
            ("out", "  ultralytics  8.4.218"),
            ("out", ""),
            ("out", "설치 완료. 이어서 확인합니다:"),
            ("out", "  cd /home/pw/rl_shortcut"),
            ("out", "  source scripts/env.sh"),
            ("out", "  python scripts/check_install.py --headless      # course code ... import OK 가 나오면 준비 끝")],
           "l02_t_install.png", prompt="pw@laptop:~/rl_shortcut$ ", title="pw@laptop: ~/rl_shortcut — bash",
           highlight=("== ", "설치 완료"))
    render([("prompt", "cd ~/rl_shortcut"),
            ("prompt", "source scripts/env.sh", "pw@laptop:~/rl_shortcut$ "),
            ("out", "rl_shortcut 환경: Python 3.11.18 · torch 2.7.0+cu128"),
            ("out", "GPU: NVIDIA GeForce RTX 5060 Ti, 580.196.07"),
            ("out", "작업 폴더: /home/pw/rl_shortcut  (명령은 모두 이 폴더에서 실행합니다)"),
            ("prompt", "python scripts/check_install.py --headless", V)],
           "l02_t_env.png", prompt="pw@laptop:~$ ", title="pw@laptop: ~ — bash", highlight=("rl_shortcut 환경", "GPU:", "작업 폴더"))
    render([("prompt", "python scripts/check_install.py --headless"),
            ("out", "Do you accept the EULA? (Yes/No): Yes"),
            ("out", "  ... (Isaac Sim 시작 로그) ..."),
            ("out", "isaacsim     5.1.0.0"),
            ("out", "isaaclab     0.54.2"),
            ("out", "rsl-rl-lib   3.1.2"),
            ("out", "tensordict   0.14.3"),
            ("out", "torch        2.7.0+cu128"),
            ("out", "ultralytics  8.4.218"),
            ("out", "GPU          NVIDIA GeForce RTX 5060 Ti, cuda True"),
            ("out", "course code  pendulum, pinky, go2, g1: import OK"),
            ("out", "physics: 100 steps of 0.01 s on cuda:0")],
           "l02_t_check.png", prompt=V, highlight=("Do you accept", "cuda True", "import OK", "physics:"))


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


def pinky():
    render([("prompt", "python scripts/pinky_dataset.py --headless --enable_cameras"),
            ("out", "  ... (Isaac Sim 시작 로그) ..."),
            ("out", "[dataset] 400 images -> /home/pw/rl_shortcut/datasets/pinky_road"),
            ("out", "[dataset] 16 / 400"),
            ("out", "  ..."),
            ("out", "[dataset] 400 / 400"),
            ("out", "[dataset] pixels: background 41.6 %, road 43.0 %, sidewalk 15.4 %"),
            ("out", "[dataset] road view from the map vs from the camera labels: mean difference 0.010 (0 = same)")],
           "l11_t_dataset.png", prompt=V, highlight=("400 / 400", "mean difference"))
    render([("prompt", "python scripts/pinky_yolo.py"),
            ("out", "  ..."),
            ("out", "      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size"),
            ("out", "      40/40      3.22G     0.3961      0.475     0.3136     0.8695         17        640: 100% 23/23 16.9it/s 1.4s"),
            ("out", "40 epochs completed in 0.019 hours."),
            ("out", "YOLO11n-seg summary (fused): 114 layers, 2,849,710 parameters, 0 gradients, 9.7 GFLOPs"),
            ("out", "                   all         40        128      0.938      0.924      0.977      0.879       0.95      0.925       0.98      0.835"),
            ("out", "Results saved to /home/pw/rl_shortcut/logs/pinky_yolo/2026-11-24_12-13-15"),
            ("out", "[yolo] mask mAP50 0.977, mAP50-95 0.835"),
            ("out", "[yolo] weights: /home/pw/rl_shortcut/logs/pinky_yolo/2026-11-24_12-13-15/weights/best.pt")],
           "l11_t_yolo.png", prompt=V, highlight=("40 epochs completed", "[yolo] mask"))
    render([("prompt", "python scripts/pinky_yolo.py --check"),
            ("out", "[yolo] /home/pw/rl_shortcut/logs/pinky_yolo/2026-11-24_12-13-15/weights/best.pt"),
            ("out", "[yolo] road IoU on 6 validation images: 0.980"),
            ("out", "[yolo] saved /home/pw/rl_shortcut/outputs/pinky_yolo_check.jpg  (camera | road mask | road view 8 x 16)")],
           "l11_t_check.png", prompt=V, highlight=("road IoU",))
    render([("out", "[train] pinky: 1024 envs, observation 130, action 2"),
            ("out", "  ..."),
            ("out", "                      Learning iteration 299/300"),
            ("out", "                       Computation: 55804 steps/s (collection: 0.552s, learning 0.035s)"),
            ("out", "             Mean action noise std: 0.30"),
            ("out", "                       Mean reward: 82.93"),
            ("out", "               Mean episode length: 600.00"),
            ("out", "           Episode_Reward/progress: 1.9164"),
            ("out", "            Episode_Reward/in_lane: 0.8752"),
            ("out", "        Episode_Reward/action_rate: -0.0271"),
            ("out", "        Episode_Reward/terminating: 0.0000"),
            ("out", "      Episode_Termination/time_out: 0.9973"),
            ("out", "      Episode_Termination/off_road: 0.0027"),
            ("out", "                    Iteration time: 0.59s"),
            ("out", "                      Time elapsed: 00:02:55"),
            ("out", "[train] done in 2.9 min -> /home/pw/rl_shortcut/logs/pinky/2026-11-24_12-21-44")],
           "l12_t_train.png", prompt=V, highlight=("Mean episode length", "progress", "[train] done"))
    render([("prompt", "python scripts/play.py --task pinky --headless --seconds 30"),
            ("out", "[play] /home/pw/rl_shortcut/logs/pinky/2026-11-24_12-21-44/model_299.pt"),
            ("out", "[play] t=  1 s  fails=  0  speed 0.201 m/s  lane error 1.5 cm"),
            ("out", "[play] t=  2 s  fails=  0  speed 0.238 m/s  lane error 1.3 cm"),
            ("out", "[play] t=  3 s  fails=  0  speed 0.246 m/s  lane error 2.3 cm"),
            ("out", "  ..."),
            ("out", "[play] t= 30 s  fails=  0  speed 0.234 m/s  lane error 2.9 cm"),
            ("out", "[play] 30 s x 16 envs: fails=0, speed 0.240 m/s  lane error 2.4 cm")],
           "l12_t_play.png", prompt=V, highlight=("30 s x 16 envs",))
    render([("prompt", "python scripts/play.py --task pinky --perception yolo --enable_cameras --headless --seconds 30"),
            ("out", "[yolo] /home/pw/rl_shortcut/logs/pinky_yolo/2026-11-24_12-13-15/weights/best.pt"),
            ("out", "[play] /home/pw/rl_shortcut/logs/pinky/2026-11-24_12-21-44/model_299.pt"),
            ("out", "[play] t=  1 s  fails=  0  speed 0.165 m/s  lane error 1.4 cm"),
            ("out", "[play] t=  2 s  fails=  0  speed 0.233 m/s  lane error 1.1 cm"),
            ("out", "[play] t=  3 s  fails=  0  speed 0.241 m/s  lane error 2.0 cm"),
            ("out", "  ..."),
            ("out", "[play] t= 30 s  fails=  0  speed 0.230 m/s  lane error 2.5 cm"),
            ("out", "[play] 30 s x 16 envs: fails=0, speed 0.238 m/s  lane error 2.6 cm")],
           "l12_t_play_yolo.png", prompt=V, highlight=("[yolo]", "30 s x 16 envs"))


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
    render([("out", "                      Learning iteration 499/500"),
            ("out", "                       Computation: 188421 steps/s (collection: 0.442s, learning 0.080s)"),
            ("out", "                       Mean reward: 34.57"),
            ("out", "               Mean episode length: 995.57"),
            ("out", "   Episode_Reward/track_lin_vel_xy: 1.4421"),
            ("out", "    Episode_Reward/track_ang_vel_z: 0.6688"),
            ("out", "      Episode_Reward/feet_air_time: -0.0287"),
            ("out", "      Episode_Reward/joint_torques: -0.0749"),
            ("out", "Metrics/base_velocity/error_vel_xy: 0.1615"),
            ("out", "      Episode_Termination/time_out: 0.9859"),
            ("out", "  Episode_Termination/base_contact: 0.0141"),
            ("out", "                    Iteration time: 0.52s"),
            ("out", "                      Time elapsed: 00:04:26")],
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


def models():
    """scripts/get_model.sh: a model received from the instructor, then play.py / pinky_yolo.py with it."""
    def got(name, size):
        return [("prompt", f"./scripts/get_model.sh {name}"),
                ("out", f"가져왔습니다: /home/pw/Downloads/{name}"),
                ("out", f"        →  /home/pw/rl_shortcut/models/{name}  ({size})"),
                ("out", "  ...")]
    pages = [("l05_t_model.png", got("pendulum.pt", "48K"),
              "python scripts/play.py --task pendulum --checkpoint models/pendulum.pt", "[play] models/pendulum.pt"),
             ("l11_t_model.png", got("pinky_road_seg.pt", "5.8M"),
              "python scripts/pinky_yolo.py --check --weights models/pinky_road_seg.pt", "[yolo] models/pinky_road_seg.pt"),
             ("l12_t_model.png", got("pinky.pt", "612K"),
              "python scripts/play.py --task pinky --checkpoint models/pinky.pt", "[play] models/pinky.pt"),
             ("l08_t_model.png", got("go2.pt", "968K"),
              "python scripts/play.py --task go2 --checkpoint models/go2.pt", "[play] models/go2.pt"),
             ("l09_t_model.png", got("g1_walk.pt", "1.7M"),
              "python scripts/play.py --task g1 --motion walk --checkpoint models/g1_walk.pt", "[play] models/g1_walk.pt"),
             ("l10_t_model.png", got("g1_walk.pt", "1.7M") + got("g1_bruce_lee.pt", "1.7M"),
              "python scripts/play.py --task g1 --motion bruce_lee --checkpoint models/g1_bruce_lee.pt",
              "[play] models/g1_bruce_lee.pt")]
    for name, lines, command, first in pages:
        render(lines + [("prompt", command), ("out", "  ... (Isaac Sim 시작 로그) ..." if "play.py" in command else "  ..."),
                        ("out", first)],
               name, prompt=V, highlight=("가져왔습니다", "→", "[play]", "[yolo]"))


if __name__ == "__main__":
    for f in (setup, physics, pendulum_check, pendulum, pinky, go2, g1_motion, g1_walk, g1_bruce, models):
        f()
