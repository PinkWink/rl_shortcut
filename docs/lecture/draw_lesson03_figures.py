"""Figures for page ③ (Isaac Sim physics). Output: docs/lecture/l03_*.png

  python3 docs/lecture/draw_lesson03_figures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import BLUE, GREY, INK, RED, C, arrow, fig_ax, note, rbox, save  # noqa: E402

# measured with scripts/physics_parallel.py --headless (RTX 5060 Ti, 500 steps of 0.005 s)
PARALLEL = [(1, 853), (64, 797), (1024, 626), (4096, 311)]


def stack():
    fig, ax = fig_ax(16, 5.4)
    layers = [("scripts/  ·  rl_shortcut/  (이 과정의 코드)", "환경 설정 · 학습 · 평가", C["yellow"]),
              ("RSL-RL 3.1.2", "PPO 알고리즘 (액터 · 크리틱 신경망, PyTorch)", C["purple"]),
              ("Isaac Lab 2.3.2", "환경 틀(ManagerBasedRLEnv) · 로봇 설정 · 병렬 복제 · GPU 텐서 API", C["teal"]),
              ("Isaac Sim 5.1.0", "USD 장면  ·  PhysX 5 (GPU 물리)  ·  RTX 렌더러  ·  Omniverse Kit", C["blue"]),
              ("NVIDIA GPU + 드라이버 580", "물리 · 렌더 · 신경망 계산이 모두 GPU 에서", C["grey"])]
    for i, (name, what, fc) in enumerate(layers):
        y = 4.3 - i * 0.95
        rbox(ax, 0.3, y, 4.6, 0.8, name, fc=fc, tfs=11.5)
        rbox(ax, 5.1, y, 10.6, 0.8, what, fc="white", tfs=10.5, lw=1.0)
    save(fig, "l03_stack.png")


def vs_mujoco():
    fig, ax = fig_ax(16, 5.0)
    rows = [("", "앞 과정: MuJoCo 3.15", "오늘: Isaac Sim 5.1 + Isaac Lab"),
            ("모델 파일", "MJCF (XML)", "USD (Pixar 장면 형식)"),
            ("물리 엔진", "MuJoCo (CPU)", "PhysX 5 (GPU)"),
            ("한 번에 돌리는 로봇", "1 대 (사람이 조작)", "수천 대 (정책이 연습)"),
            ("화면", "OpenGL 뷰어", "RTX 렌더러 (사진에 가까움)"),
            ("학습 방법", "모방학습: 시연 → ACT", "강화학습: 보상 → PPO"),
            ("파이썬", "3.10 이상 (검증 3.12)", "3.11 (Isaac Sim 5.1)")]
    x, w = [0.3, 4.3, 10.0], [3.8, 5.5, 5.7]
    for i, r in enumerate(rows):
        y = 4.3 - i * 0.66
        for j, t in enumerate(r):
            if i == 0 and j == 0:
                continue
            fc = [C["grey"], C["yellow"], C["blue"]][j] if i == 0 or j == 0 else "white"
            rbox(ax, x[j], y, w[j], 0.56, t, fc=fc, tfs=10.5 if i else 11.5, lw=1.0)
    save(fig, "l03_vs_mujoco.png")


def usd():
    fig, ax = fig_ax(16, 5.6)
    tree = [(0, "/World", "Stage 의 뿌리", C["grey"]),
            (1, "ground", "바닥 (모두 공유)", C["grey"]),
            (1, "light", "조명", C["grey"]),
            (1, "envs", "환경들을 담는 폴더", C["teal"]),
            (2, "env_0", "환경 하나 (자기 원점)", C["teal"]),
            (3, "Robot", "Articulation 루트 (go2.usd 를 참조)", C["blue"]),
            (4, "base · FL_hip · FL_thigh · ...", "링크 = 강체 (질량 · 충돌 모양 · 메시)", C["yellow"]),
            (4, "FL_hip_joint · ...", "관절 (종류 · 한계 · 축)", C["yellow"]),
            (2, "env_1 · env_2 · ... env_4095", "같은 구조를 복사", C["teal"])]
    for i, (d, name, what, fc) in enumerate(tree):
        y = 5.0 - i * 0.58
        x = 0.4 + d * 0.7
        if d:
            ax.plot([x - 0.4, x - 0.4, x], [y + 0.5, y + 0.23, y + 0.23], color=GREY, lw=1.2)
        rbox(ax, x, y, 6.2 - d * 0.3, 0.46, name, fc=fc, tfs=10.5, lw=1.0)
        note(ax, 9.6, y + 0.23, what, fs=10.5)
    save(fig, "l03_usd.png")


def dt():
    fig, ax = fig_ax(16, 3.8)
    x0, step = 0.6, 0.62
    for k in range(21):
        x = x0 + k * step
        ax.plot([x, x], [1.55, 1.95], color=BLUE, lw=1.4)
    for k in range(0, 21, 4):
        x = x0 + k * step
        ax.plot([x, x], [2.35, 2.95], color=RED, lw=2.4)
        if k < 20:
            rbox(ax, x + 0.08, 2.4, 4 * step - 0.16, 0.5, "같은 관절 목표 유지", fc=C["red"], tfs=9.5, lw=0.8)
    ax.plot([x0, x0 + 20 * step], [1.75, 1.75], color=INK, lw=1)
    note(ax, x0 + 20 * step + 0.3, 1.75, "물리 200 Hz (dt 0.005 s)", color=BLUE, fs=11, bold=True)
    note(ax, x0 + 20 * step + 0.3, 2.65, "정책 50 Hz", color=RED, fs=11, bold=True)
    note(ax, 0.6, 0.95, "decimation = 4 : 정책이 행동을 한 번 내면, 물리가 4 스텝 진행한 뒤 다음 관측을 받습니다.", fs=10.5)
    note(ax, 0.6, 0.45, "역진자는 dt 0.005 s · decimation 2 (정책 100 Hz), Go2 · G1 은 dt 0.005 s · decimation 4 (정책 50 Hz).", fs=10.5)
    save(fig, "l03_dt.png")


def parallel_chart():
    fig, ax = fig_ax(16, 5.2)
    n = len(PARALLEL)
    bw, gap, base = 2.1, 0.6, 0.8
    peak = max(c * s for c, s in PARALLEL)
    for i, (count, sps) in enumerate(PARALLEL):
        x = 1.0 + i * (bw + gap)
        h1 = 3.4 * sps / 900
        ax.add_patch(plt_rect(x, base, bw / 2 - 0.05, h1, C["blue"]))
        ax.text(x + bw / 4, base + h1 + 0.12, f"{sps}", ha="center", fontsize=10, color=INK)
        h2 = 3.4 * (count * sps) / peak
        ax.add_patch(plt_rect(x + bw / 2 + 0.05, base, bw / 2 - 0.05, max(h2, 0.02), C["red"]))
        ax.text(x + 3 * bw / 4, base + max(h2, 0.02) + 0.12, f"{count * sps:,}", ha="center", fontsize=10, color=INK)
        ax.text(x + bw / 2, base - 0.35, f"{count} 대", ha="center", fontsize=11.5, fontweight="bold", color=INK)
    ax.plot([0.7, 1.0 + n * (bw + gap)], [base, base], color=INK, lw=1)
    rbox(ax, 12.4, 3.7, 3.3, 0.5, "스텝 / 초 (장면 전체)", fc=C["blue"], tfs=10, lw=0.8)
    rbox(ax, 12.4, 3.05, 3.3, 0.5, "로봇-스텝 / 초", fc=C["red"], tfs=10, lw=0.8)
    note(ax, 12.4, 2.5, "Go2, 서 있는 자세", fs=10)
    note(ax, 12.4, 2.1, "RTX 5060 Ti, 500 스텝", fs=10)
    save(fig, "l03_parallel_chart.png")


def plt_rect(x, y, w, h, fc):
    from matplotlib.patches import Rectangle
    return Rectangle((x, y), w, h, fc=fc, ec=INK, lw=1.0)


if __name__ == "__main__":
    for f in (stack, vs_mujoco, usd, dt, parallel_chart):
        f()
