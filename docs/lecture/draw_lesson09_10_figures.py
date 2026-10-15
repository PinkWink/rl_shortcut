"""Concept figures for pages ⑨ ⑩ (G1 motion imitation). Output: docs/lecture/l09_*.png, l10_*.png

  python3 docs/lecture/draw_lesson09_10_figures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import INK, RED, C, arrow, chain, fig_ax, note, rbox, save  # noqa: E402


def why_mimic():
    fig, ax = fig_ax(16, 4.4)
    rows = [("", "Go2 (4족)", "G1 (휴머노이드)"),
            ("정책이 움직이는 관절", "12", "23"),
            ("땅에 닿는 곳", "발 4개 (넓은 지지면)", "발바닥 2개 (좁다, 늘 균형을 잡아야)"),
            ("무게 중심", "낮다 (약 0.3 m)", "높다 (골반 약 0.8 m)"),
            ("보상만으로 학습하면", "자연스러운 걸음이 잘 나온다", "걷기는 하지만 사람답지 않은 동작이 나오기 쉽다")]
    x, w = [0.3, 4.6, 8.7], [4.1, 3.9, 7.0]
    for i, r in enumerate(rows):
        y = 3.6 - i * 0.72
        for j, t in enumerate(r):
            if i == 0 and j == 0:
                continue
            fc = [C["grey"], C["teal"], C["red"]][j] if i == 0 or j == 0 else "white"
            rbox(ax, x[j], y, w[j], 0.6, t, fc=fc, tfs=10.5, lw=1.0)
    save(fig, "l09_why_mimic.png")


def mimic():
    fig, ax = fig_ax(16, 3.6)
    chain(ax, [("사람 모션 캡처", ["관절 위치 (SMPL)", "ACCAD · LAFAN1 · 쿵푸"]),
               ("리타게팅", ["G1 관절 23개 각도로", "(데이터셋에서 이미 됨)"]),
               ("레퍼런스", ["프레임마다 목표 자세", "· 높이 · 방향 · 속도"]),
               ("강화학습", ["레퍼런스에 가까울수록 보상", "모터 명령은 정책이 찾는다"]),
               ("정책", ["관측 107 → 행동 23", "넘어지지 않고 따라 하기"])],
          1.0, 2.1, colors=[C["grey"], C["grey"], C["yellow"], C["purple"], C["green"]], tfs=11.5, fs=9.5)
    note(ax, 0.3, 0.45, "레퍼런스에는 자세만 있고 모터 명령(행동)은 없습니다. 그래서 모방학습(ACT)처럼 지도 학습할 수 없고, 강화학습이 필요합니다.", fs=10.5)
    save(fig, "l09_mimic.png")


def motions():
    fig, ax = fig_ax(16, 3.6)
    rbox(ax, 0.3, 0.6, 7.5, 2.7, "motions/walk.npz", ["원본: ACCAD B3 walk1 (사람 걷기)", "프레임 42~155 만 사용: 113 프레임 = 3.77초",
                                                     "시작과 끝이 이어져 무한 반복 가능", "3.77초에 3.98 m 앞으로 (약 1.06 m/s)"],
         fc=C["teal"], align="left", dy=0.48)
    rbox(ax, 8.2, 0.6, 7.5, 2.7, "motions/bruce_lee.npz", ["원본: kungfu Bruce_Lee_pose (무술 동작)", "330 프레임 = 11초 전체",
                                                          "자세를 낮췄다 일어서기 (골반 0.31 ~ 0.93 m)", "제자리에서 몸통 회전 · 발차기 (이동 0.13 m)"],
         fc=C["red"], align="left", dy=0.48)
    save(fig, "l09_motions.png")


def obs():
    fig, ax = fig_ax(16, 3.0)
    parts = [("선속도", 3, C["blue"]), ("각속도", 3, C["blue"]), ("중력", 3, C["blue"]), ("관절 각", 23, C["teal"]),
             ("관절 속도", 23, C["teal"]), ("직전 행동", 23, C["purple"]), ("목표 자세", 23, C["red"]),
             ("위상", 2, C["red"]), ("방향 오차", 2, C["red"]), ("목표 속도", 2, C["red"])]
    x, scale = 0.3, 15.4 / 107
    for name, n, fc in parts:
        rbox(ax, x, 1.0, n * scale - 0.03, 1.1, f"{n}", fc=fc, tfs=11 if n > 2 else 9, lw=1.0)
        if n > 2:
            ax.text(x + n * scale / 2, 0.8, name, ha="center", va="top", fontsize=10 if n > 3 else 9, color=INK,
                    rotation=0 if n > 3 else 35)
        x += n * scale
    ax.text(15.7, 0.8, "위상 · 방향 오차 · 목표 속도 (각 2)", ha="right", va="top", fontsize=10, color=INK)
    note(ax, 0.3, 2.55, "관측 107 = 내 몸 상태 77  +  레퍼런스가 지금 원하는 것 30 (빨강)", bold=True, color=INK, fs=11)
    save(fig, "l09_obs.png")


def rewards():
    fig, ax = fig_ax(16, 5.6)
    rows = [("joints", "+5.0", "전신 23관절이 목표 자세에 가까울수록 exp(-오차²)", C["green"]),
            ("arms", "+2.0", "팔 10관절을 한 번 더 (더 엄격하게, 오차 × 3)", C["green"]),
            ("velocity", "+2.0", "몸통 이동 속도(벡터)가 목표에 가까울수록", C["green"]),
            ("yaw", "+2.0", "몸통 방향이 레퍼런스 방향에 가까울수록", C["green"]),
            ("height", "+2.0", "골반 높이가 레퍼런스 높이에 가까울수록", C["green"]),
            ("alive", "+1.0", "넘어지지 않은 매 스텝", C["green"]),
            ("감점 5가지", "-", "기울기 0.2 · 발 미끄러짐 0.1 · 관절 속도 0.005 · 행동 변화 0.05 · 관절 한계 1.0", C["red"])]
    for i, (k, wgt, m, fc) in enumerate(rows):
        y = 4.75 - i * 0.66
        rbox(ax, 0.3, y, 2.8, 0.54, k, fc="white", tfs=10.5, lw=1.0)
        rbox(ax, 3.3, y, 1.6, 0.54, wgt, fc=fc, tfs=10.5, lw=1.0)
        rbox(ax, 5.1, y, 10.6, 0.54, m, fc="white", tfs=10.5, lw=1.0)
    save(fig, "l09_rewards.png")


def rsi():
    fig, ax = fig_ax(16, 3.8)
    ax.plot([0.8, 15.2], [2.6, 2.6], color=INK, lw=2)
    for k in range(8):
        x = 0.8 + k * (14.4 / 7)
        ax.plot([x, x], [2.45, 2.75], color=INK, lw=1.4)
    note(ax, 0.8, 3.15, "레퍼런스 클립 (끝 → 처음으로 이어짐)", bold=True, color=INK, fs=11)
    starts = [1.6, 5.2, 8.9, 12.6]
    for k, sx in enumerate(starts):
        arrow(ax, (sx, 1.3), (sx, 2.4), color=RED)
        rbox(ax, sx - 1.2, 0.5, 2.4, 0.75, f"env {k}: 여기서 시작", fc=C["red"], tfs=10, lw=1.0)
    note(ax, 0.8, 0.15, "리셋마다 무작위 프레임 f 를 골라 그 순간의 관절 각 · 관절 속도 · 골반 높이 · 방향 · 이동 속도로 로봇을 놓습니다.", fs=10.5)
    save(fig, "l09_rsi.png")


def flow10():
    fig, ax = fig_ax(16, 3.4)
    chain(ax, [("① 걷기 정책 (⑨)", ["g1_walk.pt", "걷기 레퍼런스로 학습"]),
               ("② 걷기 정책 + 쿵푸 레퍼런스", ["학습 없이 바로 실행", "어떻게 될까?"]),
               ("③ 쿵푸 학습", ["코드 그대로, --motion 만", "bruce_lee, 3000 it"]),
               ("④ 쿵푸 정책 + 쿵푸 레퍼런스", ["g1_bruce_lee.pt", "추론 · 비교"])],
          0.9, 2.1, colors=[C["teal"], C["yellow"], C["purple"], C["red"]], tfs=11.5, fs=10)
    note(ax, 0.3, 0.4, "네 단계 모두 같은 환경 코드(rl_shortcut/g1_env.py), 같은 관측 107 · 행동 23 입니다. 바뀌는 것은 레퍼런스와 정책뿐.", fs=10.5)
    save(fig, "l10_flow.png")


# play.py --task g1 --motion walk --seconds 12 --num_envs 64 --checkpoint model_<it>.pt (falls were 0 for all)
WALK_PROGRESS = [(100, 0.275), (200, 0.110), (300, 0.096), (500, 0.085), (1000, 0.080), (1999, 0.074)]


def walk_progress():
    import numpy as np
    fig, ax = fig_ax(16, 4.2)
    ax2 = fig.add_axes([0.07, 0.2, 0.55, 0.7])
    its, err = zip(*WALK_PROGRESS)
    ax2.plot(its, err, "o-", color="#1565C0", lw=2)
    for a, b in WALK_PROGRESS:
        ax2.annotate(f"{b:.3f}", (a, b), xytext=(6, 6), textcoords="offset points", fontsize=10)
    ax2.set_xlabel("iteration (체크포인트)"); ax2.set_ylabel("관절 오차 RMS (rad)")
    ax2.set_ylim(0, 0.32); ax2.grid(alpha=0.3); ax2.spines[["top", "right"]].set_visible(False)
    rbox(ax, 10.6, 0.9, 5.1, 2.6, "같은 레퍼런스, 다른 체크포인트", ["64대 · 12초, 모두 넘어짐 0",
                                                                f"300 it (약 4분): {np.rad2deg(0.096):.1f}°",
                                                                f"1999 it (29분): {np.rad2deg(0.074):.1f}°"],
         fc=C["blue"], align="left", dy=0.5)
    save(fig, "l09_progress.png")


if __name__ == "__main__":
    for f in (why_mimic, mimic, motions, obs, rewards, rsi, flow10, walk_progress):
        f()
