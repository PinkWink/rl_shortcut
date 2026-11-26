# 수업 환경 켜기: 터미널을 새로 열 때마다 한 번 실행한다 (bash · zsh).
#   cd ~/rl_shortcut
#   source scripts/env.sh
# 작업 폴더 확인(다른 폴더에서 실행했으면 이동) → 다른 파이썬 환경(conda · 다른 venv · ROS 2 · ~/.local) 정리 → .venv 활성화 → 버전 확인
# 여러 번 실행해도 된다. 설치는 scripts/install.sh, 전체 확인은 python scripts/check_install.py --headless
if [[ -n "${BASH_VERSION:-}" && "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "❌ source 로 실행하세요:  source ${BASH_SOURCE[0]}" >&2
  exit 1
fi
_RL_WS="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
if [[ ! -f "$_RL_WS/.venv/bin/activate" ]]; then
  echo "❌ $_RL_WS/.venv 가 없습니다. 먼저 설치하세요:  $_RL_WS/scripts/install.sh" >&2
  unset _RL_WS
  return 1
fi
cd "$_RL_WS" || return 1

# 다른 파이썬 환경이 섞이면 검증한 버전 대신 그쪽 패키지를 불러온다: 끄고 시작한다.
if [[ "${VIRTUAL_ENV:-}" != "$_RL_WS/.venv" ]]; then
  [[ -n "${VIRTUAL_ENV:-}" ]] && command -v deactivate >/dev/null 2>&1 && deactivate
  while [[ -n "${CONDA_PREFIX:-}" ]]; do                 # conda (base 포함)
    _RL_PREV="$CONDA_PREFIX"
    conda deactivate >/dev/null 2>&1
    [[ "${CONDA_PREFIX:-}" == "$_RL_PREV" ]] && break     # conda 명령을 쓸 수 없는 셸
  done
  source "$_RL_WS/.venv/bin/activate"
fi
if [[ -n "${PYTHONPATH:-}" ]]; then
  echo "  PYTHONPATH 를 비웠습니다 (ROS 2 등 다른 환경의 파이썬 패키지). 이 터미널은 수업 전용으로 사용하세요."
fi
unset PYTHONPATH PYTHONHOME _RL_PREV
export PYTHONNOUSERSITE=1                                # ~/.local 의 pip 패키지를 보지 않는다

# 검증한 버전(constraints.txt)과 같은지, GPU 가 보이는지 확인한다 (0.2초).
python - "$_RL_WS/constraints.txt" <<'EOF'
import importlib.metadata as m
import sys

want = dict(line.strip().split("==") for line in open(sys.argv[1]) if "==" in line)
bad = []
for name in ("torch", "numpy", "tensordict", "rsl-rl-lib", "isaacsim", "isaaclab", "rl_shortcut"):
    try:
        have = m.version(name).split("+")[0]
    except m.PackageNotFoundError:
        bad.append(f"{name}: 설치되지 않음")
        continue
    if name in want and have != want[name]:
        bad.append(f"{name}: {have} (검증한 버전 {want[name]})")
if bad:
    print("  ⚠️ 설치 상태가 검증한 것과 다릅니다: " + " · ".join(bad))
    print("     ./scripts/install.sh 를 다시 실행하세요. 그래도 같으면 rm -rf .venv 후 다시 실행합니다.")
else:
    print(f"rl_shortcut 환경: Python {sys.version.split()[0]} · torch {m.version('torch')}")
EOF
if command -v nvidia-smi >/dev/null 2>&1 && _RL_GPU="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)" \
   && [[ -n "$_RL_GPU" ]]; then
  echo "GPU: $_RL_GPU"
else
  echo "  ⚠️ nvidia-smi 가 동작하지 않습니다. NVIDIA 드라이버를 확인하세요 (실습은 GPU 가 필요합니다)."
fi
echo "작업 폴더: $PWD  (명령은 모두 이 폴더에서 실행합니다)"
unset _RL_WS _RL_GPU
