"""Figures for page ② (setup). Output: docs/lecture/l02_*.png

  python3 docs/lecture/draw_lesson02_figures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import INK, C, arrow, chain, fig_ax, note, rbox, save, tiles  # noqa: E402


def requirements():
    fig, ax = fig_ax(16, 4.6)
    tiles(ax, [("운영체제", ["Ubuntu 24.04 (검증 24.04.4)", "인터넷 · sudo 권한"], C["blue"]),
               ("NVIDIA GPU", ["RTX 시리즈, VRAM 8 GB 이상 권장", "검증: RTX 5060 Ti 16 GB"], C["red"]),
               ("드라이버", ["580 이상 (검증 580.126)", "nvidia-smi 로 확인"], C["yellow"]),
               ("Python", ["3.11 (Isaac Sim 5.1 전용)", "deadsnakes PPA 로 설치"], C["grey"]),
               ("디스크", ["venv 약 18 GB", "에셋 캐시 · 학습 로그 여유 포함 30 GB"], C["teal"]),
               ("내려받기", ["PyTorch 약 3.9 GB", "Isaac Sim 약 4.7 GB"], C["green"])], 3, h=1.85)
    save(fig, "l02_requirements.png")


def install_flow():
    fig, ax = fig_ax(16, 2.6)
    chain(ax, [("① 저장소 받기", ["git 설치 · git clone"]), ("② 설치 스크립트", ["./scripts/install.sh  (약 20분)"]),
               ("③ 설치 확인", ["env.sh · check_install.py"])],
          0.5, 1.35, gap=0.5, colors=[C["blue"], C["teal"], C["green"]], tfs=12.5, fs=10.5)
    note(ax, 0.3, 2.3, "검증: 새로 설치한 Ubuntu 24.04 에서 ① ~ ③ 약 20분 (인터넷 속도에 따라 달라짐)", fs=10)
    save(fig, "l02_install_flow.png")


def install_steps():
    fig, ax = fig_ax(16, 5.3)
    rows = [("1  apt 패키지", "python3.11 (deadsnakes PPA) · venv · dev · cmake · build-essential", C["grey"]),
            ("2  가상환경", ".venv = Python 3.11 (Isaac Sim 5.1 은 3.11 에서만 동작)", C["purple"]),
            ("3  PyTorch", "torch 2.7.0 · torchvision 0.22.0 (CUDA 12.8 → RTX 50 시리즈도 동작)", C["yellow"]),
            ("4  Isaac Sim", "isaacsim[all,extscache] 5.1.0 pip 휠 (소스 빌드 없음)", C["red"]),
            ("5  Isaac Lab", "v2.3.2 소스 + RSL-RL · constraints.txt 로 버전 고정", C["teal"]),
            ("6  이 과정", "pip install -e . (rl_shortcut) · YOLO (ultralytics 8.4.218)", C["blue"]),
            ("7  버전 확인", "torch · numpy · tensordict · rsl-rl-lib · isaacsim · ultralytics", C["green"])]
    for i, (name, what, fc) in enumerate(rows):
        y = 4.5 - i * 0.66
        rbox(ax, 0.3, y, 3.6, 0.54, name, fc=fc, tfs=10.5, lw=1.0)
        rbox(ax, 4.1, y, 11.6, 0.54, what, fc="white", tfs=10, lw=1.0)
    note(ax, 0.3, 0.15, "단계마다 이미 설치된 것은 건너뜀 · 실패하면 그 자리에서 멈추고 무엇을 할지 알려 줌")
    save(fig, "l02_install_steps.png")


def folders():
    fig, ax = fig_ax(16, 6.66)
    rows = [("~/rl_shortcut/", "작업 폴더 = 이 과정의 git 저장소", C["blue"]),
            ("  .venv/", "Python 3.11 venv: PyTorch · Isaac Sim · Isaac Lab (약 18 GB)", C["purple"]),
            ("  IsaacLab/", "Isaac Lab 2.3.2 소스 (install.sh 가 git clone)", C["teal"]),
            ("  rl_shortcut/", "환경 코드: pendulum_env · pinky_env · go2_env · g1_env · ppo", C["yellow"]),
            ("  scripts/", "실행 파일: train · play · 물리 실험 · Pinky · Go2 · G1", C["yellow"]),
            ("  assets/ · motions/", "역진자 키트 · Pinky Pro · 핑크시티 (USD) · G1 레퍼런스 동작 2개", C["green"]),
            ("  checkpoints/ · tools/", "미리 학습한 정책 5개 · 모델 · 동작 변환 도구", C["green"]),
            ("  logs/<실험>/<날짜_시각>/", "학습할 때 생기는 체크포인트 · TensorBoard 기록", C["grey"]),
            ("  datasets/pinky_road/", "⑥ 페이지에서 모으는 도로 인식 데이터 (영상 400장)", C["grey"])]
    for i, (name, what, fc) in enumerate(rows):
        y = 5.81 - i * 0.66
        rbox(ax, 0.3, y, 5.4, 0.54, name, fc=fc, tfs=10.5, lw=1.0)
        rbox(ax, 5.9, y, 9.8, 0.54, what, fc="white", tfs=10, lw=1.0)
    note(ax, 0.3, 0.2, ".venv/ · IsaacLab/ · logs/ · datasets/ 는 .gitignore 에 있어 git 에 올라가지 않음")
    save(fig, "l02_folders.png")


def scripts_map():
    fig, ax = fig_ax(16, 9.72)
    rows = [("install.sh", "한 번에 설치 (apt · venv · pip · Isaac Lab)", "②", C["green"]),
            ("env.sh", "새 터미널마다 source: 가상환경 활성화 · 버전 확인", "②", C["green"]),
            ("check_install.py", "버전 · GPU · 물리 100 스텝 확인", "②", C["green"]),
            ("physics_drop.py", "공 3개 낙하: 반발 계수 · dt", "③", C["blue"]),
            ("physics_parallel.py", "Go2 N 대 병렬 물리 속도", "③", C["blue"]),
            ("pendulum_check.py", "키트 진자 흔들림 주기 vs 이론", "④", C["teal"]),
            ("pinky_dataset.py", "핑크시티에서 영상 + 자동 라벨 400장", "⑥", C["purple"]),
            ("pinky_yolo.py", "YOLO 세그먼테이션 학습 · 확인", "⑥", C["purple"]),
            ("train.py --task pendulum | pinky | go2 | g1", "PPO 학습 → logs/<실험>/", "⑤⑦⑩⑪⑫", C["purple"]),
            ("play.py --task ...", "학습한 정책 실행 · 평가 · 영상", "⑤⑦⑩⑪⑫", C["purple"]),
            ("go2_inspect.py", "go2.usd 의 링크 · 관절 · 순서", "⑧", C["yellow"]),
            ("go2_stand.py", "PD 모터로 서 있기 (kp 비교)", "⑧", C["yellow"]),
            ("g1_motion.py", "G1 레퍼런스 동작 재생", "⑪⑫", C["red"])]
    for i, (name, what, page, fc) in enumerate(rows):
        y = 8.85 - i * 0.68
        rbox(ax, 0.3, y, 6.4, 0.56, name, fc=fc, tfs=10.5, lw=1.0)
        rbox(ax, 6.9, y, 6.9, 0.56, what, fc="white", tfs=10, lw=1.0)
        rbox(ax, 14.0, y, 1.7, 0.56, page, fc=C["grey"], tfs=10.5, lw=1.0)
    note(ax, 0.3, 9.60, "scripts/ 아래 스크립트", bold=True, color=INK, fs=11)
    note(ax, 6.9, 9.60, "하는 일", bold=True, color=INK, fs=11)
    note(ax, 14.0, 9.60, "페이지", bold=True, color=INK, fs=11)
    save(fig, "l02_scripts_map.png")


def troubleshoot():
    fig, ax = fig_ax(16, 6.56)
    rows = [("디스크 여유가 ... GB 입니다. 30 GB 이상 필요합니다.", "설치에 30 GB 필요", "공간을 비우고 install.sh 다시"),
            ("fatal: destination path 'rl_shortcut' already exists", "이미 받은 폴더가 있음", "cd ~/rl_shortcut 후 ② 부터"),
            ("No module named 'isaaclab'", "환경을 켜지 않음 · 설치 중단", "cd ~/rl_shortcut · source scripts/env.sh"),
            ("Do you accept the EULA? (Yes/No)", "첫 실행 라이선스 확인", "Yes 입력 (한 번만)"),
            ("CUDA error: no kernel image is available", "torch 가 CUDA 12.8 빌드가 아님", "rm -rf .venv 후 install.sh 다시"),
            ("ImportError: cannot import name 'is_structseq_instance'", "패키지를 따로 올려 버전이 바뀜", "rm -rf .venv 후 install.sh 다시"),
            ("503 Service Unavailable (apt install)", "PPA 서버가 잠시 응답 없음", "./scripts/install.sh 다시 실행")]
    x = [0.3, 7.6, 11.0]; w = [7.1, 3.2, 4.7]
    for j, t in enumerate(["증상 (메시지)", "원인", "해결"]):
        rbox(ax, x[j], 5.71, w[j], 0.6, t, fc=[C["grey"], C["yellow"], C["green"]][j], tfs=11)
    for i, r in enumerate(rows):
        y = 4.91 - i * 0.78
        for j, t in enumerate(r):
            rbox(ax, x[j], y, w[j], 0.64, t, fc="white", tfs=9.5 if j == 0 else 10, lw=1.0)
    save(fig, "l02_troubleshoot.png")


if __name__ == "__main__":
    for f in (requirements, install_flow, install_steps, folders, scripts_map, troubleshoot):
        f()
