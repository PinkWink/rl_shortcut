# rl_shortcut

A two-day reinforcement learning course with NVIDIA Isaac Sim 5.1 and Isaac Lab 2.3:
PinkLAB rotary inverted pendulum kit, Pinky Pro driving the City of Pink ring road from YOLO road segmentation, Unitree Go2 (from its USD to walking), Unitree G1 (motion imitation: walk, then Bruce Lee kung fu).

The lecture (Korean) is on Confluence; its sources are in `docs/lecture/`.

## Install (Ubuntu 24.04, NVIDIA driver 580+)

```bash
sudo apt update && sudo apt install -y git
mkdir -p ~/rl_shortcut && cd ~/rl_shortcut
git clone https://github.com/PinkWink/rl_shortcut.git .
sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev git cmake build-essential
python3.11 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
git clone --depth 1 --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git
export PIP_CONSTRAINT=~/rl_shortcut/constraints.txt PIP_BUILD_CONSTRAINT=~/rl_shortcut/constraints.txt   # tested versions
(cd IsaacLab && ./isaaclab.sh --install rsl_rl)      # answer Yes to the EULA
pip install -e . -c constraints.txt
pip install --no-deps -c constraints.txt ultralytics==8.4.218 ultralytics-thop polars polars-runtime-32   # YOLO; --no-deps keeps numpy 1.26
python scripts/check_install.py --headless
```

## Use

| Script | What |
|---|---|
| `scripts/physics_drop.py`, `scripts/physics_parallel.py` | PhysX experiments: restitution and dt, many robots on the GPU |
| `scripts/pendulum_check.py [--free_arm]` | PinkLAB pendulum kit: masses, free swing period vs theory |
| `scripts/pinky_dataset.py --headless --enable_cameras` | Pinky Pro camera images + automatic road labels (YOLO format) into `datasets/pinky_road/` |
| `scripts/pinky_yolo.py [--check]` | YOLO11n segmentation training (road, sidewalk) into `logs/pinky_yolo/` |
| `scripts/go2_inspect.py`, `scripts/go2_stand.py` | Go2 USD contents, PD motors holding the standing pose |
| `scripts/g1_motion.py --motion walk\|bruce_lee` | play a G1 reference clip (no physics control) |
| `scripts/train.py --task pendulum\|pinky\|go2\|g1 [--motion ...]` | PPO (RSL-RL) training into `logs/<experiment>/<date_time>/` |
| `scripts/play.py --task ... [--checkpoint name\|path]` | run a policy, print falls and tracking |
| `scripts/play.py --task pinky --perception yolo --enable_cameras` | the Pinky policy driven by its camera + YOLO instead of the floor map |

`checkpoints/` holds the policies trained for the lecture (pendulum, pinky, go2, g1_walk, g1_bruce_lee);
`play.py` uses them when `logs/` has no run of that experiment.

## Data

`assets/pinky_pro/` is the Pinky Pro USD converted by `tools/convert_pinky.py` from the URDF of the official repository
[pinklab-art/pinky_pro](https://github.com/pinklab-art/pinky_pro). `assets/pink_city/` is the City of Pink (3 x 3 m):
textures made by `16_pink_city/pink_city.py` of [PinkWink/nvidia-dli-lecture](https://github.com/PinkWink/nvidia-dli-lecture),
USD built from them by `tools/make_pink_city.py`. The road segmentation weights are not in the repository: train them with `scripts/pinky_yolo.py` (about a minute),
or save the file attached to the lecture page as `checkpoints/pinky_road_seg.pt`. Ultralytics models and code are AGPL-3.0.

`assets/pinklab_pendulum/` is the PinkLAB rotary inverted pendulum kit (URDF, meshes, and the USD made from them by
`tools/convert_pendulum.py`; colliders come from the visual meshes, since PhysX starts thousands of shape-less
articulations very slowly).

`motions/*.npz` are converted with `tools/convert_motion.py` from
[openhe/g1-retargeted-motions](https://huggingface.co/datasets/openhe/g1-retargeted-motions) (MIT):
`walk.npz` = ACCAD `B3_-_walk1_stageii` frames 42-155, `bruce_lee.npz` = kungfu `Bruce_Lee_pose`.
