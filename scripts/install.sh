#!/usr/bin/env bash
# 강화학습 단기 과정 설치: Ubuntu 24.04 + NVIDIA GPU. 이 저장소 폴더 안에 모든 것을 설치한다.
#   ./scripts/install.sh
# apt 패키지(Python 3.11) → .venv → PyTorch → Isaac Sim → Isaac Lab → 이 과정의 패키지 순서로 한 번에 진행한다.
# 여러 번 실행해도 안전하다: 네트워크가 끊겨 멈추면 다시 실행하면 이어서 설치한다.
# 끝나면 python scripts/check_install.py --headless 로 확인한다.
set -eo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WS"
fail() { echo "❌ $*" >&2; exit 1; }
step() { echo; echo "== $*"; }
retry() {      # 강의실 네트워크 · PPA 서버가 잠깐 끊기는 일이 잦다: 세 번까지 다시 시도
  local n
  for n in 1 2 3; do
    "$@" && return 0
    echo "  ⚠️ 실패 ($n/3): $*"; sleep 5
  done
  return 1
}
PYTORCH="torch==2.7.0 torchvision==0.22.0"
ISAACSIM="isaacsim[all,extscache]==5.1.0"
ISAACLAB=v2.3.2
YOLO="ultralytics==8.4.218 ultralytics-thop polars polars-runtime-32"

step "시스템 확인"
[[ "$(uname -s)" == Linux ]] || fail "Ubuntu 24.04 에서 실행하세요."
. /etc/os-release
[[ "${VERSION_ID:-}" == 24.04 ]] || echo "  ⚠️ Ubuntu 24.04 에서 검증했습니다 (지금: ${PRETTY_NAME:-알 수 없음})."
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  echo "  GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
else
  echo "  ⚠️ nvidia-smi 가 동작하지 않습니다. NVIDIA 드라이버(580 이상)를 설치해야 실습을 실행할 수 있습니다."
fi
FREE_GB=$(df -BG --output=avail "$WS" | tail -1 | tr -dc 0-9)
[[ -d .venv/lib/python3.11/site-packages/isaacsim ]] || (( FREE_GB >= 30 )) \
  || fail "디스크 여유가 ${FREE_GB} GB 입니다. 30 GB 이상 필요합니다."
SUDO=""; [[ $EUID -eq 0 ]] || SUDO=sudo
# conda 가 켜져 있어도 venv 는 apt 로 설치한 python3.11 로 만든다. ~/.local 의 pip 패키지도 보지 않는다.
[[ -n "${CONDA_PREFIX:-}" ]] && echo "  ⚠️ conda 환경($CONDA_PREFIX)이 켜져 있습니다. 설치는 /usr/bin/python3.11 로 진행합니다."
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
[[ -n "${VIRTUAL_ENV:-}" ]] && { PATH="${PATH//$VIRTUAL_ENV\/bin:/}"; unset VIRTUAL_ENV; }

step "apt 패키지 (Python 3.11)"
APT="python3.11 python3.11-venv python3.11-dev git cmake build-essential"
MISSING=""
for pkg in $APT; do dpkg -s "$pkg" >/dev/null 2>&1 || MISSING="$MISSING $pkg"; done
if [[ -n "$MISSING" ]]; then
  echo "  설치합니다:$MISSING  (sudo 비밀번호를 물을 수 있습니다)"
  if ! apt-cache show python3.11 >/dev/null 2>&1; then      # Ubuntu 24.04 의 기본은 3.12: deadsnakes PPA 에서 3.11
    retry $SUDO apt-get update || fail "apt update 실패 (네트워크 확인). 다시 실행하세요."
    $SUDO apt-get install -y software-properties-common
    retry $SUDO add-apt-repository -y ppa:deadsnakes/ppa || fail "deadsnakes PPA 추가 실패. 잠시 뒤 다시 실행하세요."
  fi
  retry $SUDO apt-get update || fail "apt update 실패 (네트워크 확인). 다시 실행하세요."
  # shellcheck disable=SC2086
  retry $SUDO apt-get install -y -o Acquire::Retries=5 $MISSING \
    || fail "apt 설치 실패 (503 이면 PPA 서버가 잠시 응답하지 않는 것). 받은 패키지는 남아 있으니 다시 실행하세요."
else
  echo "  이미 설치됨"
fi

step "Python 가상환경 (.venv, Python 3.11)"
if [[ -x .venv/bin/python ]] && ! .venv/bin/python -c 'import sys; sys.exit(sys.version_info[:2] != (3, 11))'; then
  fail ".venv 가 Python 3.11 이 아닙니다. rm -rf .venv 후 다시 실행하세요."
fi
[[ -x .venv/bin/python ]] || /usr/bin/python3.11 -m venv .venv
source .venv/bin/activate
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_RETRIES=10 PIP_DEFAULT_TIMEOUT=60      # 큰 휠이 느린 망에서 끊기지 않게
python -m pip install -q --upgrade pip
echo "  $(python --version), $(pip --version | cut -d' ' -f1-2)"

step "PyTorch (CUDA 12.8, 약 3.9 GB)"
# shellcheck disable=SC2086
retry pip install $PYTORCH --index-url https://download.pytorch.org/whl/cu128 || fail "PyTorch 설치 실패. 다시 실행하세요."

step "Isaac Sim 5.1.0 (약 4.7 GB)"
retry pip install "$ISAACSIM" --extra-index-url https://pypi.nvidia.com || fail "Isaac Sim 설치 실패. 다시 실행하세요."

step "Isaac Lab 2.3.2 + RSL-RL"
# 이 과정에서 검증한 버전으로 고정한다. 고정하지 않으면 최신 tensordict · setuptools 가 설치되어 실습 코드가 깨진다.
# pip 25.3 부터 빌드 환경에는 PIP_BUILD_CONSTRAINT 가 따로 필요하다.
export PIP_CONSTRAINT="$WS/constraints.txt" PIP_BUILD_CONSTRAINT="$WS/constraints.txt"
if [[ ! -x IsaacLab/isaaclab.sh ]]; then
  rm -rf IsaacLab                       # 받다가 끊긴 폴더
  retry git clone --depth 1 --branch $ISAACLAB https://github.com/isaac-sim/IsaacLab.git \
    || fail "Isaac Lab 을 받지 못했습니다 (네트워크 확인). 다시 실행하세요."
fi
[[ -n "${TERM:-}" && "$TERM" != dumb ]] || export TERM=xterm       # isaaclab.sh 는 tput 을 쓴다
echo "  NVIDIA Omniverse 라이선스(EULA) 동의를 물으면 Yes 를 입력하세요."
(cd IsaacLab && ./isaaclab.sh --install rsl_rl)
# isaaclab.sh 는 패키지 설치가 실패해도 0 으로 끝난다: 직접 확인한다.
for pkg in isaaclab isaaclab_rl isaaclab_tasks rsl-rl-lib; do
  python -c "import importlib.metadata as m, sys; m.version(sys.argv[1])" $pkg 2>/dev/null \
    || fail "Isaac Lab 의 $pkg 가 설치되지 않았습니다. 위 출력의 ERROR 줄을 확인하고 다시 실행하세요."
done

step "이 과정의 패키지 (rl_shortcut, YOLO)"
PYTHONWARNINGS=ignore pip install -q -e . -c constraints.txt      # setuptools 의 editable 안내문은 숨김
# shellcheck disable=SC2086
retry pip install -q --no-deps -c constraints.txt $YOLO || fail "ultralytics 설치 실패. 다시 실행하세요."     # --no-deps: numpy 2.x 방지

step "버전 확인"
python - <<'EOF' || fail "검증한 버전과 다른 패키지가 있습니다. rm -rf .venv 후 다시 실행하세요."
import importlib.metadata as m
import sys
want = {"torch": "2.7.0", "numpy": "1.26.0", "tensordict": "0.14.3", "rsl-rl-lib": "3.1.2", "isaacsim": "5.1.0.0",
        "ultralytics": "8.4.218"}
bad = 0
for name, version in want.items():
    have = m.version(name).split("+")[0]
    print(f"  {name:12s} {have}" + ("" if have == version else f"   <- {version} 이어야 함"))
    bad += have != version
sys.exit(bad)
EOF

cat <<EOF

✅ 설치 완료. 이어서 확인합니다:
  cd $WS && source .venv/bin/activate
  python scripts/check_install.py --headless      # course code ... import OK 가 나오면 준비 끝
처음 실행할 때 NVIDIA Omniverse 라이선스(EULA) 동의를 물으면 Yes 를 입력합니다.
새 터미널을 열 때마다 첫 줄(cd · source)을 먼저 실행합니다.
EOF
