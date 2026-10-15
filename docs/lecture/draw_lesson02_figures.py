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
    fig, ax = fig_ax(16, 3.9)
    chain(ax, [("① 작업 폴더", ["mkdir · git clone"]), ("② apt", ["python3.11 · git · cmake"]),
               ("③ venv", ["python3.11 -m venv"]), ("④ PyTorch", ["2.7.0 · CUDA 12.8"])],
          2.2, 1.15, gap=0.4, colors=[C["blue"], C["grey"], C["purple"], C["yellow"]], tfs=12, fs=10)
    chain(ax, [("⑤ Isaac Sim", ["5.1.0 pip 휠"]), ("⑥ Isaac Lab", ["2.3.2 · rsl_rl"]),
               ("⑦ 이 과정", ["pip install -e ."]), ("⑧ 확인", ["check_install.py"])],
          0.4, 1.15, gap=0.4, colors=[C["red"], C["teal"], C["blue"], C["green"]], tfs=12, fs=10)
    xr, xl = 0.3 + 3 * 3.95 + 1.775, 0.3 + 1.775
    ax.plot([xr, xr, xl], [2.2, 1.85, 1.85], color=INK, lw=1.4)
    arrow(ax, (xl, 1.85), (xl, 1.55))
    note(ax, 0.3, 3.65, "이 PC 실측: ④ 2분 21초  ·  ⑤ 5분 26초  ·  ⑥ 30초  (인터넷 속도에 따라 달라집니다)", fs=10)
    save(fig, "l02_install_flow.png")


def folders():
    fig, ax = fig_ax(16, 6.0)
    rows = [("~/rl_shortcut/", "작업 폴더 = 이 과정의 git 저장소", C["blue"]),
            ("  .venv/", "Python 3.11 venv: PyTorch · Isaac Sim · Isaac Lab (약 18 GB)", C["purple"]),
            ("  IsaacLab/", "Isaac Lab 2.3.2 소스 (⑥ 에서 git clone)", C["teal"]),
            ("  rl_shortcut/", "환경 코드: pendulum_env · go2_env · g1_env · motion · ppo", C["yellow"]),
            ("  scripts/", "실행 파일: train · play · 물리 실험 · Go2 · G1 확인", C["yellow"]),
            ("  assets/ · motions/", "PinkLAB 역진자 키트 (URDF · USD) · G1 레퍼런스 동작 2개", C["green"]),
            ("  checkpoints/ · tools/", "미리 학습한 정책 4개 · 모델 · 동작 변환 도구", C["green"]),
            ("  logs/<실험>/<날짜_시각>/", "학습할 때 생기는 체크포인트 · TensorBoard 기록", C["grey"])]
    for i, (name, what, fc) in enumerate(rows):
        y = 5.15 - i * 0.66
        rbox(ax, 0.3, y, 5.4, 0.54, name, fc=fc, tfs=10.5, lw=1.0)
        rbox(ax, 5.9, y, 9.8, 0.54, what, fc="white", tfs=10, lw=1.0)
    note(ax, 0.3, 0.2, ".venv/ · IsaacLab/ · logs/ 는 .gitignore 에 있어 git 에 올라가지 않습니다.")
    save(fig, "l02_folders.png")


def scripts_map():
    fig, ax = fig_ax(16, 7.0)
    rows = [("check_install.py", "버전 · GPU · 물리 100 스텝 확인", "②", C["green"]),
            ("physics_drop.py", "공 3개 낙하: 반발 계수 · dt", "③", C["blue"]),
            ("physics_parallel.py", "Go2 N 대 병렬 물리 속도", "③", C["blue"]),
            ("pendulum_check.py", "키트 진자 흔들림 주기 vs 이론", "④", C["teal"]),
            ("train.py --task pendulum | go2 | g1", "PPO 학습 → logs/<실험>/", "⑤⑧⑨⑩", C["purple"]),
            ("play.py --task ...", "학습한 정책 실행 · 평가 · 영상", "⑤⑧⑨⑩", C["purple"]),
            ("go2_inspect.py", "go2.usd 의 링크 · 관절 · 순서", "⑥", C["yellow"]),
            ("go2_stand.py", "PD 모터로 서 있기 (kp 비교)", "⑥", C["yellow"]),
            ("g1_motion.py", "G1 레퍼런스 동작 재생", "⑨⑩", C["red"])]
    for i, (name, what, page, fc) in enumerate(rows):
        y = 6.13 - i * 0.68
        rbox(ax, 0.3, y, 6.4, 0.56, name, fc=fc, tfs=10.5, lw=1.0)
        rbox(ax, 6.9, y, 6.9, 0.56, what, fc="white", tfs=10, lw=1.0)
        rbox(ax, 14.0, y, 1.7, 0.56, page, fc=C["grey"], tfs=10.5, lw=1.0)
    note(ax, 0.3, 6.88, "scripts/ 아래 스크립트", bold=True, color=INK, fs=11)
    note(ax, 6.9, 6.88, "하는 일", bold=True, color=INK, fs=11)
    note(ax, 14.0, 6.88, "페이지", bold=True, color=INK, fs=11)
    save(fig, "l02_scripts_map.png")


def troubleshoot():
    fig, ax = fig_ax(16, 5.0)
    rows = [("Unable to locate package python3.11", "deadsnakes PPA 없음", "② add-apt-repository 먼저"),
            ("fatal: destination path '.' already exists ...", "폴더가 비어 있지 않음", "ls -a 확인 후 비우고 다시"),
            ("No module named 'isaaclab'", "venv 활성화 안 됨", "source .venv/bin/activate"),
            ("Do you accept the EULA? (Yes/No)", "첫 실행 라이선스 확인", "Yes 입력 (한 번만)"),
            ("CUDA error: no kernel image is available", "torch 가 CUDA 12.8 빌드가 아님", "④ 명령의 --index-url 확인")]
    x = [0.3, 7.6, 11.0]; w = [7.1, 3.2, 4.7]
    for j, t in enumerate(["증상 (메시지)", "원인", "해결"]):
        rbox(ax, x[j], 4.15, w[j], 0.6, t, fc=[C["grey"], C["yellow"], C["green"]][j], tfs=11)
    for i, r in enumerate(rows):
        y = 3.35 - i * 0.78
        for j, t in enumerate(r):
            rbox(ax, x[j], y, w[j], 0.64, t, fc="white", tfs=9.5 if j == 0 else 10, lw=1.0)
    save(fig, "l02_troubleshoot.png")


if __name__ == "__main__":
    for f in (requirements, install_flow, folders, scripts_map, troubleshoot):
        f()
