"""Figures for pages ⑥ ⑦ ⑧ (Go2). Output: docs/lecture/l06_*.png, l07_*.png, l08_*.png

  python3 docs/lecture/draw_lesson06_08_figures.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import BLUE, GREEN, INK, RED, C, arrow, fig_ax, note, rbox, save  # noqa: E402

# play.py --task go2 --cmd ... --seconds 6, 16 envs: (command, measured vx vy wz)
SWEEP = [((0.5, 0, 0), (0.48, -0.02, 0.02)), ((1.0, 0, 0), (0.97, -0.05, -0.03)),
         ((-0.5, 0, 0), (-0.47, 0.02, 0.03)), ((0, 0.5, 0), (-0.01, 0.48, 0.02)),
         ((0, 0, 1.0), (-0.01, -0.03, 0.89)), ((1.0, 0, 0.5), (0.95, -0.06, 0.47)), ((0, 0, 0), (0.00, -0.01, -0.00))]
# play.py --task go2 --cmd 1 0 0 --seconds 10 --num_envs 64 --checkpoint model_<it>.pt: mean vx
PROGRESS = [(0, -0.01), (50, 0.00), (100, 0.65), (150, 0.89), (299, 0.98)]


def pd():
    fig, ax = fig_ax(16, 4.6)
    rbox(ax, 0.3, 1.0, 7.2, 3.2, "PD 위치 제어 (관절마다)", ["토크 = kp × (목표 - 현재 각) - kd × 각속도",
                                                         "kp = 25 Nm/rad,  kd = 0.5 Nm·s/rad",
                                                         "목표에서 0.24 rad 처지면 6 Nm 를 낸다",
                                                         "정책은 '목표 각'만 정하고 토크는 모터가"],
         fc=C["blue"], align="left", dy=0.5)
    ax2 = fig.add_axes([0.55, 0.2, 0.4, 0.68])
    w = np.linspace(0, 30, 50)
    ax2.plot(w, np.clip(23.5 * (1 - w / 30), 0, None), color=RED, lw=2)
    ax2.fill_between(w, 0, np.clip(23.5 * (1 - w / 30), 0, None), color="#FFE0E0")
    ax2.set_xlabel("관절 속도 (rad/s)"); ax2.set_ylabel("낼 수 있는 최대 토크 (Nm)")
    ax2.set_title("DCMotorCfg: 빠를수록 토크가 줄어든다", fontsize=11)
    ax2.spines[["top", "right"]].set_visible(False); ax2.grid(alpha=0.3)
    save(fig, "l06_pd.png")


def task():
    fig, ax = fig_ax(16, 4.2)
    rbox(ax, 0.3, 0.8, 4.6, 2.9, "명령 (10초마다 무작위)", ["vx  앞뒤   -1 ~ 1 m/s", "vy  옆     -1 ~ 1 m/s",
                                                      "wz  회전   -1 ~ 1 rad/s", "2% 는 '서 있기' (0, 0, 0)"],
         fc=C["yellow"], align="left", dy=0.5)
    rbox(ax, 5.7, 0.8, 4.6, 2.9, "정책", ["관측 48개 → 행동 12개", "신경망 128-128-128", "50 Hz 로 관절 목표를 낸다"],
         fc=C["purple"], align="left", dy=0.5)
    rbox(ax, 11.1, 0.8, 4.6, 2.9, "보상", ["실제 몸통 속도가 명령에 가까울수록 +", "발을 들었다 놓으면 +",
                                        "흔들림 · 기울기 · 큰 토크는 -"], fc=C["green"], align="left", dy=0.5)
    arrow(ax, (4.9, 2.25), (5.7, 2.25)); arrow(ax, (10.3, 2.25), (11.1, 2.25))
    save(fig, "l07_task.png")


def vs_cartpole():
    fig, ax = fig_ax(16, 4.6)
    rows = [("", "역진자 (④⑤)", "Go2 (⑦⑧)"),
            ("명령 (Commands)", "없음", "속도 명령 3개, 10초마다 새로"),
            ("행동", "힘 1개 (카트)", "관절 목표 12개 (기본 자세 + 0.25 × 행동)"),
            ("관측", "4개", "48개 (몸 상태 + 명령 + 관절 + 직전 행동, 잡음 포함)"),
            ("보상", "막대 세우기", "명령 따라가기 + 걸음새 + 감점 6가지"),
            ("센서 · 무작위화", "없음", "발 접촉 센서 · 마찰 · 몸통 질량"),
            ("학습", "4096 환경, 150 it, 18초", "4096 환경, 300 it, 2.7분")]
    x, w = [0.3, 4.2, 8.0], [3.7, 3.6, 7.7]
    for i, r in enumerate(rows):
        y = 3.95 - i * 0.62
        for j, t in enumerate(r):
            if i == 0 and j == 0:
                continue
            fc = [C["grey"], C["yellow"], C["blue"]][j] if i == 0 or j == 0 else "white"
            rbox(ax, x[j], y, w[j], 0.52, t, fc=fc, tfs=10.5, lw=1.0)
    save(fig, "l07_vs_cartpole.png")


def obs():
    fig, ax = fig_ax(16, 2.8)
    parts = [("몸통 선속도", 3, C["blue"]), ("각속도", 3, C["blue"]), ("중력 방향", 3, C["blue"]),
             ("명령", 3, C["yellow"]), ("관절 각 (기본 자세 대비)", 12, C["teal"]), ("관절 속도", 12, C["teal"]),
             ("직전 행동", 12, C["purple"])]
    x, scale = 0.3, 15.4 / 48
    for name, n, fc in parts:
        rbox(ax, x, 1.0, n * scale - 0.04, 1.2, f"{n}", fc=fc, tfs=12, lw=1.0)
        ax.text(x + n * scale / 2, 0.75, name, ha="center", va="top", fontsize=9.5 if n < 12 else 10.5, color=INK,
                rotation=0 if n >= 12 else 25)
        x += n * scale
    note(ax, 0.3, 2.5, "관측 48 = 3 + 3 + 3 + 3 + 12 + 12 + 12", bold=True, color=INK, fs=11)
    save(fig, "l07_obs.png")


def rewards():
    fig, ax = fig_ax(16, 6.4)
    rows = [("track_lin_vel_xy", "+1.5", "vx · vy 가 명령에 가까울수록 (exp)", C["green"]),
            ("track_ang_vel_z", "+0.75", "wz 가 명령에 가까울수록 (exp)", C["green"]),
            ("feet_air_time", "+0.25", "발이 0.5초 넘게 떴다 닿으면 (걸음을 유도)", C["green"]),
            ("lin_vel_z", "-2.0", "몸통 위아래 속도²", C["red"]),
            ("ang_vel_xy", "-0.05", "몸통 구르기 · 끄덕임 각속도²", C["red"]),
            ("flat_orientation", "-2.5", "몸통 기울기²", C["red"]),
            ("joint_torques", "-0.0002", "관절 토크²", C["red"]),
            ("joint_acc", "-2.5e-7", "관절 가속도²", C["red"]),
            ("action_rate", "-0.01", "직전 행동과의 차이²", C["red"])]
    for i, (k, wgt, m, fc) in enumerate(rows):
        y = 5.55 - i * 0.62
        rbox(ax, 0.3, y, 4.2, 0.52, k, fc="white", tfs=10.5, lw=1.0)
        rbox(ax, 4.7, y, 1.9, 0.52, wgt, fc=fc, tfs=10.5, lw=1.0)
        rbox(ax, 6.8, y, 8.9, 0.52, m, fc="white", tfs=10.5, lw=1.0)
    note(ax, 0.3, 0.1, "Isaac Lab 평지 Go2 예제 (UnitreeGo2FlatEnvCfg) 와 같은 항목 · 가중치입니다.", fs=10.5)
    save(fig, "l07_rewards.png")


def exp_shape():
    fig, ax = fig_ax(16, 4.0)
    ax2 = fig.add_axes([0.06, 0.18, 0.42, 0.72])
    e = np.linspace(-1.5, 1.5, 200)
    ax2.plot(e, np.exp(-e ** 2 / 0.25), color=GREEN, lw=2.2)
    ax2.set_xlabel("속도 오차 (m/s)"); ax2.set_ylabel("보상 (가중치 곱하기 전)")
    ax2.set_title("exp(-오차² / 0.5²)", fontsize=11)
    ax2.grid(alpha=0.3); ax2.spines[["top", "right"]].set_visible(False)
    for err in (0.1, 0.3, 0.5):
        ax2.plot([err], [np.exp(-err ** 2 / 0.25)], "o", color=INK)
        ax2.annotate(f"{err} → {np.exp(-err ** 2 / 0.25):.2f}", (err, np.exp(-err ** 2 / 0.25)), xytext=(8, 4),
                     textcoords="offset points", fontsize=9.5)
    rbox(ax, 8.4, 0.9, 7.3, 2.5, "왜 exp 모양인가", ["오차 0 에서 최대 1 → 넘치게 빨라도 이득 없음",
                                                   "오차가 커도 0 아래로 내려가지 않음 → 넘어지는 것보다 낫다",
                                                   "std 0.5: 오차 0.5 m/s 에서 0.37 로 줄어든다"],
         fc=C["green"], align="left", dy=0.5)
    save(fig, "l07_exp.png")


def sweep():
    fig, ax = fig_ax(16, 5.6)
    heads = ["명령 vx", "vy", "wz", "실제 vx", "vy", "wz", "넘어짐"]
    xs = [0.3, 2.4, 4.5, 6.9, 9.0, 11.1, 13.5]
    for j, t in enumerate(heads):
        rbox(ax, xs[j], 4.8, 1.9 if j < 6 else 2.2, 0.52, t, fc=C["yellow"] if j < 3 else (C["blue"] if j < 6 else C["grey"]),
             tfs=10.5, lw=1.0)
    for i, (cmd, got) in enumerate(SWEEP):
        y = 4.15 - i * 0.58
        for j, v in enumerate(list(cmd) + list(got)):
            rbox(ax, xs[j], y, 1.9, 0.48, f"{v:+.2f}" if j >= 3 else f"{v:+.1f}", fc="white", tfs=10.5, lw=1.0)
        rbox(ax, xs[6], y, 2.2, 0.48, "0 / 16", fc="white", tfs=10.5, lw=1.0)
    note(ax, 0.3, 0.05, "vx · vy 는 m/s, wz 는 rad/s.  6초 동안 16대 평균 (처음 1초의 가속 포함).", fs=10.5)
    save(fig, "l08_sweep.png")


def progress():
    fig, ax = fig_ax(16, 4.2)
    ax2 = fig.add_axes([0.07, 0.2, 0.55, 0.7])
    its, vx = zip(*PROGRESS)
    ax2.plot(its, vx, "o-", color=BLUE, lw=2)
    ax2.axhline(1.0, color=RED, ls="--", lw=1.2); ax2.text(5, 1.03, "명령 1.0 m/s", color=RED, fontsize=10)
    for a, b in PROGRESS:
        ax2.annotate(f"{b:.2f}", (a, b), xytext=(6, -14), textcoords="offset points", fontsize=10)
    ax2.set_xlabel("iteration (체크포인트)"); ax2.set_ylabel("실제 vx (m/s)")
    ax2.set_ylim(-0.1, 1.15); ax2.grid(alpha=0.3); ax2.spines[["top", "right"]].set_visible(False)
    rbox(ax, 10.6, 0.9, 5.1, 2.6, "같은 명령, 다른 체크포인트", ["64대 · 10초 평균", "0 · 50: 서 있기만 (넘어짐 0)",
                                                               "100 이후 빠르게 명령에 다가감"],
         fc=C["blue"], align="left", dy=0.5)
    save(fig, "l08_progress.png")


if __name__ == "__main__":
    for f in (pd, task, vs_cartpole, obs, rewards, exp_shape, sweep, progress):
        f()
