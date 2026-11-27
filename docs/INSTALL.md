# 강화학습 단기 과정: 수업 전 설치 안내

수업은 설치가 끝난 PC 에서 바로 시작합니다. **수업 전에 아래 순서대로 설치하고, 마지막 확인 화면까지 보고 오세요.**
명령 세 묶음을 터미널에 붙여 넣으면 끝납니다. 걸리는 시간은 약 20~40분(인터넷 속도에 따라 다름)이고, 대부분은 기다리는 시간입니다.

터미널은 `Ctrl + Alt + T` 로 엽니다. 붙여 넣기는 `Ctrl + Shift + V` 입니다.

## 0. 준비물 확인

| 항목 | 필요한 것 | 확인 명령 |
|---|---|---|
| 운영체제 | PC 에 직접 설치한 Ubuntu 24.04 (이 환경에서 검증) | `lsb_release -d` |
| 그래픽카드 | NVIDIA RTX 시리즈 · VRAM 8 GB 이상 권장 | `nvidia-smi` |
| 드라이버 | NVIDIA 드라이버 580 이상 | `nvidia-smi` 첫 줄의 Driver Version |
| 디스크 | 홈 폴더에 여유 30 GB 이상 | `df -h ~` 의 Avail |
| 그 밖에 | 인터넷 · sudo 비밀번호(로그인 비밀번호) | |

- `nvidia-smi` 가 실행되지 않거나 드라이버가 580 보다 낮으면 → 드라이버를 580 이상으로 설치하고 재부팅한 뒤 진행합니다.
- 내려받는 양이 9 GB 이상입니다. 테더링보다 유선 · 와이파이를 권장합니다.

## 1. 저장소 받기

```bash
cd ~
sudo apt update
sudo apt install -y git
git clone https://github.com/PinkWink/rl_shortcut.git
```

- `sudo` 가 비밀번호를 물으면 로그인 비밀번호를 입력합니다 (입력해도 화면에 표시되지 않는 것이 정상).
- 홈 폴더에 `rl_shortcut` 폴더가 생깁니다. **수업의 모든 작업은 이 폴더(`~/rl_shortcut`)에서 합니다.**

## 2. 설치 스크립트 실행

```bash
cd ~/rl_shortcut
./scripts/install.sh
```

- 필요한 것을 모두 순서대로 설치합니다: Python 3.11 → 가상환경 → PyTorch → Isaac Sim → Isaac Lab → 수업 코드.
- 시작할 때 sudo 비밀번호를 한 번 묻습니다. 그 뒤로는 기다리면 됩니다.
- 마지막에 아래처럼 **`✅ 설치 완료`** 가 나오면 성공입니다.

```text
== 버전 확인
  torch        2.7.0
  numpy        1.26.0
  tensordict   0.14.3
  rsl-rl-lib   3.1.2
  isaacsim     5.1.0.0
  ultralytics  8.4.218

✅ 설치 완료. 이어서 확인합니다:
```

- **중간에 멈추거나 `❌` 가 나오면 같은 명령(`./scripts/install.sh`)을 다시 실행합니다.** 이미 설치한 것은 건너뛰고 이어서 진행합니다.
- 시스템에는 apt 패키지(Python 3.11 · cmake · build-essential)만 추가하고, 나머지는 모두 `~/rl_shortcut` 폴더 안에 설치합니다. 기존 파이썬 · conda · ROS 2 환경은 그대로입니다.

## 3. 설치 확인

```bash
cd ~/rl_shortcut
source scripts/env.sh
python scripts/check_install.py --headless
```

- 처음 실행할 때 `Do you accept the EULA? (Yes/No):` 를 묻습니다 → `Yes` 를 입력하고 Enter (NVIDIA Omniverse 라이선스 동의 · 한 번만 묻습니다).
- 처음 실행은 시간이 조금 더 걸립니다. 마지막에 아래 두 줄이 나오면 **준비 끝**입니다.

```text
course code  pendulum, pinky, go2, g1: import OK
physics: 100 steps of 0.01 s on cuda:0
```

## 수업에서: 터미널을 새로 열 때마다

```bash
cd ~/rl_shortcut
source scripts/env.sh
```

- 프롬프트 앞에 `(.venv)` 가 붙고, 마지막 줄에 `작업 폴더: /home/<이름>/rl_shortcut` 이 나오면 수업 명령을 실행할 준비가 된 것입니다.
- 이 두 줄을 실행한 터미널은 수업 전용으로 사용합니다 (conda · ROS 2 의 파이썬은 이 터미널에서만 꺼집니다).

## 잘 안 될 때

| 화면에 나오는 메시지 | 해결 |
|---|---|
| `디스크 여유가 ... GB 입니다. 30 GB 이상 필요합니다.` | 공간을 비우고 `./scripts/install.sh` 다시 실행 |
| `fatal: destination path 'rl_shortcut' already exists` | 이미 받은 것입니다 → 2번부터 진행 |
| `503 Service Unavailable` · 내려받다가 멈춤 | 잠시 뒤 `./scripts/install.sh` 다시 실행 |
| `nvidia-smi 가 동작하지 않습니다` | NVIDIA 드라이버(580 이상) 설치 → 재부팅 → 3번 다시 |
| `No module named 'isaaclab'` | `cd ~/rl_shortcut` → `source scripts/env.sh` 를 먼저 실행 |
| `설치 상태가 검증한 것과 다릅니다` · `ImportError` | `cd ~/rl_shortcut` → `rm -rf .venv` → `./scripts/install.sh` 다시 실행 |

- 설치 후에는 `pip install --upgrade ...` 로 패키지를 따로 올리지 마세요. 수업에서 검증한 버전이 바뀌어 실습 코드가 동작하지 않습니다.
- 해결되지 않으면 **터미널 화면의 마지막 20줄 정도를 사진이나 복사한 글**로 보내 주세요. 수업 전에 확인해 드립니다.
- 처음부터 다시 하고 싶으면 `rm -rf ~/rl_shortcut` 으로 폴더를 지우고 1번부터 시작합니다 (파이썬 패키지 · Isaac Sim · Isaac Lab 이 모두 이 폴더 안에 있습니다).
