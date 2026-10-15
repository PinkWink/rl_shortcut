"""Figures for page ① (introduction and curriculum). Output: docs/lecture/l01_*.png

  python3 docs/lecture/draw_lesson01_figures.py      (after render_sim.py: uses raw/*.png)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import INK, OUT, C, arrow, chain, fig_ax, note, rbox, save  # noqa: E402

RAW = OUT / "raw"


def overview():
    fig, ax = fig_ax(16, 3.4)
    chain(ax, [("환경 설정", ["~/rl_shortcut", "Isaac Sim 5.1 · Isaac Lab 2.3"]),
               ("물리엔진", ["USD · PhysX", "로봇 4096 대 병렬"]),
               ("역진자", ["강화학습 첫 환경", "PPO 18초"]),
               ("Unitree Go2", ["USD 에서 시작", "속도 명령 따라 걷기"]),
               ("Unitree G1", ["모션 모방: 걷기", "→ 이소룡 쿵푸"])],
          0.8, 2.1, colors=[C["grey"], C["blue"], C["yellow"], C["teal"], C["red"]], tfs=12, fs=9.8)
    note(ax, 0.3, 0.35, "같은 틀(환경 다섯 부분 + PPO)을 점점 어려운 로봇에 적용합니다.", fs=11)
    save(fig, "l01_overview.png")


def il_vs_rl():
    fig, ax = fig_ax(16, 4.6)
    rbox(ax, 0.3, 0.9, 7.4, 3.3, "모방학습 (앞 과정)", ["정답: 사람의 시연 (관측, 행동)", "학습: 시연과 같은 행동을 내도록 (지도 학습)",
                                                    "데이터: 사람이 리더 팔로 직접 모은다", "약점: 시연에 없던 상황"],
         fc=C["yellow"], align="left", dy=0.55)
    rbox(ax, 8.3, 0.9, 7.4, 3.3, "강화학습 (오늘)", ["정답 없음, 보상(점수)만 있다", "학습: 보상의 합이 커지도록 (PPO)",
                                                 "데이터: 로봇이 시뮬레이터에서 스스로 모은다", "약점: 보상 설계 · 시뮬레이터와 실물의 차이"],
         fc=C["blue"], align="left", dy=0.55)
    note(ax, 0.3, 0.4, "G1 (⑨⑩): 사람의 동작을 본보기로 주고(모방), 그 동작을 해내는 모터 명령은 보상으로 찾는다(강화).", fs=11)
    save(fig, "l01_il_vs_rl.png")


def results():
    names = ["cartpole_0", "go2_play_0", "g1_play_walk_3", "g1_play_bruce_lee_2"]
    caps = ["역진자: 막대 세우기", "Go2: 속도 명령 따라 걷기", "G1: 사람처럼 걷기", "G1: 이소룡 쿵푸"]
    fig = plt.figure(figsize=(16, 3.0), dpi=150)
    for k, (n, cap) in enumerate(zip(names, caps)):
        ax = fig.add_subplot(1, 4, k + 1)
        im = plt.imread(RAW / f"{n}.png")
        if n.startswith("g1"):
            im = im[40:720, 300:980]
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(cap, fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "l01_results.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved l01_results.png")


def curriculum():
    fig, ax = fig_ax(16, 6.2)
    pages = [("①", "소개 · 커리큘럼", "", C["grey"]), ("②", "환경 설정", "venv · Isaac Sim · Isaac Lab", C["grey"]),
             ("③", "Isaac Sim 물리엔진", "USD · PhysX · 공 낙하 · 로봇 4096 대", C["blue"]),
             ("④", "역진자 (1) 환경", "행동 · 관측 · 이벤트 · 보상 · 종료", C["yellow"]),
             ("⑤", "역진자 (2) 학습 · 평가", "PPO · 로그 · 학습 곡선 · play", C["yellow"]),
             ("⑥", "Go2 (1) USD 에서 서기까지", "링크 · 관절 순서 · PD 모터", C["teal"]),
             ("⑦", "Go2 (2) 걷기 환경", "속도 명령 · 관측 48 · 보상 설계", C["teal"]),
             ("⑧", "Go2 (3) 학습 · 평가", "2.7분 학습 · 명령별 속도 · 영상", C["teal"]),
             ("⑨", "G1 (1) 앞으로 걷기", "모션 모방 · RSI · 30분 학습", C["red"]),
             ("⑩", "G1 (2) 이소룡 쿵푸", "걷기 모델로 쿵푸? → 쿵푸 학습 → 추론", C["red"])]
    for i, (num, title, what, fc) in enumerate(pages):
        y = 5.55 - i * 0.6
        rbox(ax, 0.3, y, 0.8, 0.5, num, fc=fc, tfs=12, lw=1.0)
        rbox(ax, 1.3, y, 5.4, 0.5, title, fc=fc, tfs=11, lw=1.0)
        if what:
            rbox(ax, 6.9, y, 8.8, 0.5, what, fc="white", tfs=10.5, lw=1.0)
    save(fig, "l01_curriculum.png")


def timetable():
    fig, ax = fig_ax(16, 4.4)
    slots = [("09:30", 20, "①", C["grey"]), ("09:50", 60, "② 설치", C["grey"]), ("10:50", 40, "③ 물리", C["blue"]),
             ("11:30", 30, "④ 역진자", C["yellow"]), ("12:00", 60, "점심", "white"), ("13:00", 30, "⑤ 학습", C["yellow"]),
             ("13:30", 40, "⑥ Go2", C["teal"]), ("14:10", 40, "⑦ 환경", C["teal"]), ("14:50", 30, "⑧ 학습", C["teal"]),
             ("15:20", 10, "휴식", "white"), ("15:30", 50, "⑨ G1 걷기", C["red"]), ("16:20", 60, "⑩ G1 쿵푸", C["red"]),
             ("17:20", 10, "정리", C["grey"])]
    total = sum(m for _, m, _, _ in slots)
    x, scale = 0.3, 15.4 / total
    for t, m, name, fc in slots:
        w = m * scale
        rbox(ax, x, 2.0, w - 0.03, 1.1, name if w > 0.6 else "", fc=fc, tfs=10, lw=1.0)
        ax.text(x, 3.3, t, fontsize=9, color=INK, rotation=40, ha="left", va="bottom")
        x += w
    ax.text(x, 3.3, "17:30", fontsize=9, color=INK, rotation=40, ha="left", va="bottom")
    note(ax, 0.3, 1.45, "학습을 걸어 두는 때:  ⑨ 시작할 때 G1 걷기 학습 (약 30분),  ⑩ 시작할 때 G1 쿵푸 학습 (약 40분)", fs=10.5)
    note(ax, 0.3, 0.95, "끝나기를 기다리지 않고, checkpoints/ 의 미리 학습한 정책으로 평가를 진행합니다.", fs=10.5)
    save(fig, "l01_timetable.png")


if __name__ == "__main__":
    for f in (overview, il_vs_rl, curriculum, timetable):
        f()
    if (RAW / "g1_play_bruce_lee_2.png").exists():
        results()
