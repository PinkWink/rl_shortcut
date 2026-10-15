"""Figures for pages ④ ⑤ (cartpole env, PPO training). Output: docs/lecture/l04_*.png, l05_*.png

  python3 docs/lecture/draw_lesson04_05_figures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import INK, RED, C, arrow, fig_ax, note, rbox, save  # noqa: E402


def loop():
    fig, ax = fig_ax(16, 4.6)
    rbox(ax, 0.6, 1.4, 4.4, 2.2, "정책 (신경망)", ["관측 4개 → 힘 1개", "처음에는 무작위에 가깝다"], fc=C["purple"])
    rbox(ax, 10.9, 1.4, 4.5, 2.2, "환경 (Isaac Sim)", ["카트에 힘을 주고 물리 진행", "역진자 4096 개가 동시에"], fc=C["blue"])
    arrow(ax, (5.0, 3.1), (10.9, 3.1)); ax.text(7.95, 3.3, "행동: 카트에 줄 힘", ha="center", fontsize=11, color=INK)
    arrow(ax, (10.9, 2.5), (5.0, 2.5)); ax.text(7.95, 2.65, "관측: 카트 위치 · 막대 각도 · 속도", ha="center", fontsize=11, color=INK)
    arrow(ax, (10.9, 1.9), (5.0, 1.9), color=RED); ax.text(7.95, 1.45, "보상: 막대가 서 있으면 +, 쓰러지면 -", ha="center", fontsize=11, color=RED)
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
    fig, ax = fig_ax(16, 4.9)
    rows = [("alive", "+1.0", "살아 있는 매 스텝", "오래 버틸수록 점수"),
            ("terminating", "-2.0", "실패로 끝난 스텝", "레일 밖으로 나가면 큰 감점"),
            ("pole_upright", "-1.0", "막대 각도²", "기울수록 감점 (직접 쓴 함수)"),
            ("cart_vel", "-0.01", "|카트 속도|", "카트를 덜 흔들게"),
            ("pole_vel", "-0.005", "|막대 각속도|", "막대를 덜 흔들게")]
    x, w = [0.3, 4.0, 5.9, 10.2], [3.5, 1.7, 4.1, 5.5]
    for j, t in enumerate(["항목", "가중치", "값", "뜻"]):
        rbox(ax, x[j], 4.05, w[j], 0.56, t, fc=C["grey"], tfs=11, lw=1.0)
    for i, r in enumerate(rows):
        y = 3.35 - i * 0.66
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
    note(ax, 5.6, 0.45, "이 한 바퀴가 iteration 하나. 역진자는 150 바퀴 (이 PC 에서 18초).", fs=10.5)
    save(fig, "l05_ppo.png")


def log():
    fig, ax = fig_ax(16, 5.4)
    rows = [("Computation", "603,609 steps/s", "1초에 모은 환경 스텝 수"),
            ("Mean action noise std", "0.09", "탐색 잡음 크기 (1.0 에서 시작해 줄어듦)"),
            ("Mean reward", "4.71", "에피소드 하나의 보상 합 (평균)"),
            ("Mean episode length", "287.52", "에피소드 길이, 최대 300 스텝 (5초)"),
            ("Episode_Reward/alive", "0.9367", "보상 항목별 점수 (초당)"),
            ("Episode_Termination/time_out", "0.9181", "끝난 에피소드 중 시간을 다 채운 비율"),
            ("Iteration time", "0.11s", "iteration 한 바퀴에 걸린 시간")]
    for i, (k, v, m) in enumerate(rows):
        y = 4.55 - i * 0.66
        rbox(ax, 0.3, y, 5.6, 0.54, k, fc=C["blue"], tfs=10.5, lw=1.0)
        rbox(ax, 6.1, y, 2.9, 0.54, v, fc=C["yellow"], tfs=10.5, lw=1.0)
        rbox(ax, 9.2, y, 6.5, 0.54, m, fc="white", tfs=10.5, lw=1.0)
    note(ax, 0.3, 0.25, "값은 역진자 학습의 마지막 iteration (149) 실측입니다.", fs=10.5)
    save(fig, "l05_log.png")




def task():
    import math
    from matplotlib.patches import Circle, FancyBboxPatch
    fig, ax = fig_ax(16, 4.6)
    ax.plot([1.0, 10.0], [1.0, 1.0], color=INK, lw=3)
    for x in (1.0, 10.0):
        ax.plot([x, x], [0.8, 1.2], color=INK, lw=3)
    ax.add_patch(FancyBboxPatch((4.6, 1.0), 1.8, 0.7, boxstyle="round,pad=0.02", fc=C["blue"], ec=INK, lw=1.4))
    a = math.radians(18)
    px, py = 5.5, 1.7
    ax.plot([px, px + 2.4 * math.sin(a)], [py, py + 2.4 * math.cos(a)], color="#7a4b00", lw=6, solid_capstyle="round")
    ax.add_patch(Circle((px, py), 0.08, fc=INK))
    ax.plot([px, px], [py, py + 2.5], color="#999999", lw=1, ls="--")
    ax.text(px + 0.08, py + 1.25, "θ", fontsize=15, color=RED)
    arrow(ax, (3.3, 1.35), (4.55, 1.35), color=RED, lw=2.4)
    ax.text(3.0, 1.75, "힘 F (행동)", fontsize=12, color=RED)
    ax.text(5.5, 0.55, "카트 위치 x", ha="center", fontsize=11, color=INK)
    ax.text(1.0, 0.4, "-3 m", ha="center", fontsize=10, color=INK); ax.text(10.0, 0.4, "+3 m", ha="center", fontsize=10, color=INK)
    rbox(ax, 11.0, 0.8, 4.7, 3.3, "역진자", ["행동: 카트에 줄 힘 1개", "관측: x · θ · 속도 2개 = 4개",
                                        "목표: θ = 0 (막대 세우기)", "실패: 카트가 ±3 m 를 넘으면"],
         fc=C["yellow"], align="left", dy=0.55)
    save(fig, "l04_task.png")


def summary():
    fig, ax = fig_ax(16, 3.8)
    rows = [("역진자", "관측 4 · 행동 1", "막대 세우기", "18초"), ("Go2", "관측 48 · 행동 12", "속도 명령 따라 걷기", "2.7분"),
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
    fig, ax = fig_ax(16, 4.6)
    rows = [("그대로 (기준)", "287", "1", "0.06 rad (4°)", "막대를 세운다", C["green"]),
            ("행동 scale 100 → 10", "45", "102", "1.30 rad (74°)", "힘이 모자라 못 세운다", C["red"]),
            ("pole_upright 가중치 -1 → 0", "300", "0", "2.99 rad (171°)", "막대는 늘어뜨리고 레일 안에만 있는다", C["yellow"]),
            ("alive 가중치 1 → 0", "8", "558", "0.51 rad", "일부러 레일 밖으로 나가 빨리 끝낸다", C["red"])]
    heads = ["바꾼 것", "학습 끝 에피소드 길이", "play 5초 실패", "막대 각도 평균", "정책이 배운 것"]
    x, w = [0.3, 4.5, 7.2, 9.3, 11.6], [4.0, 2.5, 1.9, 2.1, 4.1]
    for j, t in enumerate(heads):
        rbox(ax, x[j], 3.75, w[j], 0.56, t, fc=C["grey"], tfs=10, lw=1.0)
    for i, r in enumerate(rows):
        y = 3.0 - i * 0.66
        for j, t in enumerate(r[:5]):
            rbox(ax, x[j], y, w[j], 0.54, t, fc=r[5] if j == 4 else "white", tfs=9.8, lw=1.0)
    note(ax, 0.3, 0.15, "각각 150 iteration 학습 후 play.py --seconds 5 (16개 환경).  막대 각도 0 = 똑바로 섬, π (3.14) = 거꾸로 매달림.", fs=10)
    save(fig, "l05_experiments.png")


if __name__ == "__main__":
    for f in (task, loop, managers, rewards, ppo, log, experiments, summary):
        f()
