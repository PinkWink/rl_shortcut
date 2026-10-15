"""Figures for pages ④ ⑤ (PinkLAB pendulum env, PPO training). Output: docs/lecture/l04_*.png, l05_*.png

  python3 docs/lecture/draw_lesson04_05_figures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import INK, RED, C, arrow, fig_ax, note, rbox, save  # noqa: E402


def loop():
    fig, ax = fig_ax(16, 4.6)
    rbox(ax, 0.6, 1.4, 4.4, 2.2, "정책 (신경망)", ["관측 4개 → 토크 1개", "처음에는 무작위에 가깝다"], fc=C["purple"])
    rbox(ax, 10.9, 1.4, 4.5, 2.2, "환경 (Isaac Sim)", ["팔에 토크를 주고 물리 진행", "키트 4096 개가 동시에"], fc=C["blue"])
    arrow(ax, (5.0, 3.1), (10.9, 3.1)); ax.text(7.95, 3.3, "행동: 팔 모터 토크", ha="center", fontsize=11, color=INK)
    arrow(ax, (10.9, 2.5), (5.0, 2.5)); ax.text(7.95, 2.65, "관측: 팔 · 진자 각도와 각속도", ha="center", fontsize=11, color=INK)
    arrow(ax, (10.9, 1.9), (5.0, 1.9), color=RED); ax.text(7.95, 1.45, "보상: 진자가 서 있으면 +, 쓰러지면 -", ha="center", fontsize=11, color=RED)
    note(ax, 0.6, 0.6, "모방학습: 정답 행동(시연)을 따라 배운다.   강화학습: 정답 없이, 보상이 커지는 쪽으로 스스로 바뀐다.", fs=11)
    save(fig, "l04_loop.png")


def managers():
    fig, ax = fig_ax(16, 4.4)
    rbox(ax, 0.3, 0.5, 3.2, 3.4, "Scene", ["바닥 · 로봇 · 조명", "num_envs 개 복제"], fc=C["grey"])
    items = [("Actions", "정책 출력 → 로봇 명령", C["purple"]), ("Observations", "정책 입력 (센서 값)", C["blue"]),
             ("Events", "리셋 · 무작위화", C["teal"]), ("Rewards", "무엇이 좋은가 (점수)", C["red"]),
             ("Terminations", "언제 에피소드를 끝내나", C["yellow"])]
    for i, (name, what, fc) in enumerate(items):
        y = 3.4 - i * 0.72
        rbox(ax, 4.2, y, 3.6, 0.58, name, fc=fc, tfs=11.5, lw=1.0)
        rbox(ax, 8.0, y, 7.7, 0.58, what, fc="white", tfs=10.5, lw=1.0)
    arrow(ax, (3.5, 2.2), (4.2, 2.2))
    save(fig, "l04_managers.png")


def rewards():
    fig, ax = fig_ax(16, 5.5)
    rows = [("alive", "+1.0", "살아 있는 매 스텝", "오래 버틸수록 점수"),
            ("terminating", "-2.0", "실패로 끝난 스텝", "진자가 쓰러지면 큰 감점"),
            ("pole_upright", "-1.0", "(진자 각도 - 목표)²", "기울수록 감점 (직접 쓴 함수)"),
            ("arm_center", "-0.1", "팔 각도²", "팔을 가운데 근처에"),
            ("arm_vel", "-0.01", "|팔 각속도|", "팔을 덜 휘두르게"),
            ("pole_vel", "-0.005", "|진자 각속도|", "진자를 덜 흔들게")]
    x, w = [0.3, 4.0, 5.9, 10.2], [3.5, 1.7, 4.1, 5.5]
    for j, t in enumerate(["항목", "가중치", "값", "뜻"]):
        rbox(ax, x[j], 4.65, w[j], 0.56, t, fc=C["grey"], tfs=11, lw=1.0)
    for i, r in enumerate(rows):
        y = 3.95 - i * 0.66
        for j, t in enumerate(r):
            fc = (C["green"] if r[1].startswith("+") else C["red"]) if j == 1 else "white"
            rbox(ax, x[j], y, w[j], 0.54, t, fc=fc, tfs=10.5, lw=1.0)
    note(ax, 0.3, 0.2, "한 스텝의 보상 = Σ 가중치 × 값 × dt.  로그의 Episode_Reward/<항목> 은 항목별 점수입니다.", fs=10.5)
    save(fig, "l04_rewards.png")


def ppo():
    fig, ax = fig_ax(16, 4.6)
    rbox(ax, 0.3, 1.2, 4.8, 2.8, "① 모으기 (rollout)", ["4096 환경 × 16 스텝", "= 65,536 개의 (관측, 행동, 보상)",
                                                    "지금 정책 + 탐색 잡음"], fc=C["blue"], align="left", dy=0.45)
    rbox(ax, 5.6, 1.2, 4.8, 2.8, "② 평가하기", ["크리틱: 이 상태의 앞으로의 점수 예측",
                                             "어드밴티지: 예측보다 잘했나?", "(GAE, γ 0.99, λ 0.95)"],
         fc=C["yellow"], align="left", dy=0.45)
    rbox(ax, 10.9, 1.2, 4.8, 2.8, "③ 고치기 (update)", ["잘한 행동은 더 자주, 못한 행동은 덜",
                                                      "한 번에 너무 바꾸지 않게 (clip 0.2)", "미니배치 4 × 5 epoch"],
         fc=C["purple"], align="left", dy=0.45)
    arrow(ax, (5.1, 2.6), (5.6, 2.6)); arrow(ax, (10.4, 2.6), (10.9, 2.6))
    ax.plot([13.3, 13.3, 2.7], [1.2, 0.75, 0.75], color=INK, lw=1.4)
    arrow(ax, (2.7, 0.75), (2.7, 1.2))
    note(ax, 5.6, 0.45, "이 한 바퀴가 iteration 하나. 역진자는 150 바퀴 (이 PC 에서 약 20초).", fs=10.5)
    save(fig, "l05_ppo.png")


def log():
    fig, ax = fig_ax(16, 5.4)
    rows = [("Computation", "665,389 steps/s", "1초에 모은 환경 스텝 수"),
            ("Mean action noise std", "0.00", "탐색 잡음 크기 (1.0 에서 시작해 줄어듦)"),
            ("Mean reward", "4.98", "에피소드 하나의 보상 합 (평균)"),
            ("Mean episode length", "500.00", "에피소드 길이, 최대 500 스텝 (5초)"),
            ("Episode_Reward/alive", "1.0000", "보상 항목별 점수 (초당)"),
            ("Episode_Termination/time_out", "1.0000", "끝난 에피소드 중 시간을 다 채운 비율"),
            ("Iteration time", "0.10s", "iteration 한 바퀴에 걸린 시간")]
    for i, (k, v, m) in enumerate(rows):
        y = 4.55 - i * 0.66
        rbox(ax, 0.3, y, 5.6, 0.54, k, fc=C["blue"], tfs=10.5, lw=1.0)
        rbox(ax, 6.1, y, 2.9, 0.54, v, fc=C["yellow"], tfs=10.5, lw=1.0)
        rbox(ax, 9.2, y, 6.5, 0.54, m, fc="white", tfs=10.5, lw=1.0)
    note(ax, 0.3, 0.25, "값은 역진자 학습의 마지막 iteration (149) 실측입니다.", fs=10.5)
    save(fig, "l05_log.png")




def task():
    import math
    from matplotlib.patches import Arc, Circle, FancyBboxPatch
    fig, ax = fig_ax(16, 4.8)
    # left: top view, the motor turns the arm
    cx, cy, r = 2.6, 2.4, 1.6
    ax.add_patch(FancyBboxPatch((cx - 1.0, cy - 1.0), 2.0, 2.0, boxstyle="round,pad=0.05", fc=C["grey"], ec=INK, lw=1.2))
    a = math.radians(25)
    ax.plot([cx, cx + r * math.sin(a)], [cy, cy + r * math.cos(a)], color="#555555", lw=9, solid_capstyle="round")
    ax.add_patch(Circle((cx + r * math.sin(a), cy + r * math.cos(a)), 0.16, fc="#f2a7c3", ec=INK, lw=1.2))
    ax.plot([cx, cx], [cy, cy + r + 0.3], color="#999999", lw=1, ls="--")
    ax.add_patch(Arc((cx, cy), 1.3, 1.3, theta1=90 - 25, theta2=90, color=RED, lw=1.5))
    ax.text(cx + 0.15, cy + 0.85, "φ", fontsize=14, color=RED)
    ax.add_patch(Arc((cx, cy), 0.8, 0.8, theta1=200, theta2=340, color=RED, lw=2.4))
    ax.annotate("", xy=(cx + 0.37, cy - 0.16), xytext=(cx + 0.33, cy - 0.24), arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
    ax.text(cx, cy - 0.75, "토크 τ (행동)", ha="center", fontsize=11.5, color=RED)
    ax.text(cx, 0.15, "위에서 본 모습: 모터가 팔을 돌린다", ha="center", fontsize=11, color=INK)
    # middle: looking along the arm, the pendulum tilts
    px, py = 7.4, 1.4
    ax.add_patch(FancyBboxPatch((px - 0.9, 0.55), 1.8, 0.85, boxstyle="round,pad=0.03", fc=C["grey"], ec=INK, lw=1.2))
    ax.add_patch(Circle((px, py + 0.25), 0.22, fc="#555555", ec=INK))
    b = math.radians(14)
    ex, ey = px + 2.3 * math.sin(b), py + 0.25 + 2.3 * math.cos(b)
    ax.plot([px, ex], [py + 0.25, ey], color="#f2a7c3", lw=7, solid_capstyle="round")
    ax.add_patch(Circle((ex, ey), 0.2, fc="#f2a7c3", ec=INK, lw=1.2))
    ax.plot([px, px], [py + 0.25, py + 2.8], color="#999999", lw=1, ls="--")
    ax.add_patch(Arc((px, py + 0.25), 2.2, 2.2, theta1=90 - 14, theta2=90, color=RED, lw=1.5))
    ax.text(px + 0.12, py + 1.55, "θ", fontsize=14, color=RED)
    ax.text(px, 0.15, "팔 방향에서 본 모습: 진자는 모터 없이 흔들린다", ha="center", fontsize=11, color=INK)
    rbox(ax, 11.0, 0.8, 4.7, 3.5, "PinkLAB 역진자 키트", ["행동: 팔 모터 토크 1개 (최대 0.14 Nm)",
                                                   "관측: φ · θ · 각속도 2개 = 4개", "목표: θ = 0 (진자 세우기)",
                                                   "실패: |θ| > 0.8 rad 또는 |φ| > 2 rad"],
         fc=C["yellow"], align="left", dy=0.6)
    save(fig, "l04_task.png")


def summary():
    fig, ax = fig_ax(16, 3.8)
    rows = [("역진자", "관측 4 · 행동 1", "진자 세우기", "약 20초"), ("Go2", "관측 48 · 행동 12", "속도 명령 따라 걷기", "2.7분"),
            ("G1 걷기", "관측 107 · 행동 23", "레퍼런스(걷기) 따라 하기", "약 30분"),
            ("G1 쿵푸", "관측 107 · 행동 23", "레퍼런스(쿵푸) 따라 하기", "약 40분")]
    x, w = [0.3, 3.4, 7.3, 12.6], [2.9, 3.7, 5.1, 3.1]
    for j, t in enumerate(["로봇", "정책 크기", "보상이 원하는 것", "학습 (RTX 5060 Ti)"]):
        rbox(ax, x[j], 3.0, w[j], 0.55, t, fc=C["grey"], tfs=11, lw=1.0)
    for i, r in enumerate(rows):
        y = 2.3 - i * 0.62
        for j, t in enumerate(r):
            rbox(ax, x[j], y, w[j], 0.52, t, fc=[C["yellow"], C["teal"], C["red"], C["red"]][i] if j == 0 else "white",
                 tfs=10.5, lw=1.0)
    save(fig, "l10_summary.png")


def experiments():
    fig, ax = fig_ax(16, 4.9)
    rows = [("그대로 (기준)", "500", "0", "0.006 rad", "진자를 세운다", C["green"]),
            ("MAX_TORQUE 0.14 → 0.0014 (1/100)", "438", "0", "0.006 rad", "그래도 세운다: 세워 두는 데는 힘이 거의 안 든다", C["yellow"]),
            ("joint_vel 관측 빼기", "57", "110", "0.195 rad", "각도만 보고는 못 세운다", C["red"]),
            ("pole_upright 가중치 -1 → 0", "333", "0", "0.004 rad", "세운다: '쓰러지면 끝'이 대신 가르친다", C["yellow"]),
            ("alive 가중치 1 → 0", "3", "2389", "0.312 rad", "일부러 바로 쓰러뜨려 빨리 끝낸다", C["red"])]
    heads = ["바꾼 것", "학습 끝 에피소드 길이", "play 5초 실패", "진자 각도 평균", "정책이 배운 것"]
    x, w = [0.3, 4.5, 7.2, 9.1, 11.0], [4.0, 2.5, 1.7, 1.7, 4.7]
    for j, t in enumerate(heads):
        rbox(ax, x[j], 4.05, w[j], 0.56, t, fc=C["grey"], tfs=10, lw=1.0)
    for i, r in enumerate(rows):
        y = 3.3 - i * 0.66
        for j, t in enumerate(r[:5]):
            rbox(ax, x[j], y, w[j], 0.54, t, fc=r[5] if j == 4 else "white", tfs=9.4, lw=1.0)
    note(ax, 0.3, 0.15, "각각 150 iteration 학습 후 play.py --seconds 5 (16개 환경).  진자 각도 = 똑바로 선 자세에서 기운 정도.", fs=10)
    save(fig, "l05_experiments.png")


if __name__ == "__main__":
    for f in (task, loop, managers, rewards, ppo, log, experiments, summary):
        f()
