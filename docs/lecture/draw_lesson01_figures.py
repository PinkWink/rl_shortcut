"""Figures for page ① (introduction and curriculum). Output: docs/lecture/l01_*.png

  python3 docs/lecture/draw_lesson01_figures.py      (after render_sim.py: uses raw/*.png)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import INK, OUT, C, arrow, chain, fig_ax, note, rbox, save, tiles  # noqa: E402

RAW = OUT / "raw"


def overview():
    fig, ax = fig_ax(16, 3.4)
    chain(ax, [("환경 설정", ["~/rl_shortcut", "Isaac Sim 5.1 · Isaac Lab 2.3"]),
               ("물리엔진", ["USD · PhysX", "로봇 4096 대 병렬"]),
               ("역진자", ["강화학습 첫 환경", "PPO 약 20초"]),
               ("Pinky Pro", ["YOLO 로 도로 인식", "핑크시티 순환 도로"]),
               ("Unitree Go2", ["USD 에서 시작", "속도 명령 따라 걷기"]),
               ("Unitree G1", ["모션 모방: 걷기", "→ 이소룡 쿵푸"])],
          0.8, 2.1, colors=[C["grey"], C["blue"], C["yellow"], C["purple"], C["teal"], C["red"]], tfs=12, fs=9.6)
    note(ax, 0.3, 0.35, "같은 틀(환경 다섯 부분 + PPO)을 점점 어려운 로봇에 적용", fs=11)
    save(fig, "l01_overview.png")


def il_vs_rl():
    fig, ax = fig_ax(16, 4.6)
    rbox(ax, 0.3, 0.9, 7.4, 3.3, "모방학습 (앞 과정)", ["정답: 사람의 시연 (관측, 행동)", "학습: 시연과 같은 행동을 내도록 (지도 학습)",
                                                    "데이터: 사람이 리더 팔로 직접 모은다", "약점: 시연에 없던 상황"],
         fc=C["yellow"], align="left", dy=0.55)
    rbox(ax, 8.3, 0.9, 7.4, 3.3, "강화학습 (이 과정)", ["정답 없음, 보상(점수)만 있다", "학습: 보상의 합이 커지도록 (PPO)",
                                                 "데이터: 로봇이 시뮬레이터에서 스스로 모은다", "약점: 보상 설계 · 시뮬레이터와 실물의 차이"],
         fc=C["blue"], align="left", dy=0.55)
    note(ax, 0.3, 0.4, "G1 (⑪⑫): 사람의 동작을 본보기로 주고(모방), 그 동작을 해내는 모터 명령은 보상으로 찾는다(강화).", fs=11)
    save(fig, "l01_il_vs_rl.png")


def results():
    names = ["pendulum_0", "pinky_city_overview", "go2_play_0", "g1_play_walk_3", "g1_play_bruce_lee_4"]
    caps = ["역진자: PinkLAB 키트 세우기", "Pinky Pro: 도로 따라 돌기", "Go2: 속도 명령 따라 걷기", "G1: 사람처럼 걷기",
            "G1: 이소룡 쿵푸"]
    fig = plt.figure(figsize=(16, 2.6), dpi=150)
    for k, (n, cap) in enumerate(zip(names, caps)):
        ax = fig.add_subplot(1, 5, k + 1)
        im = plt.imread(RAW / f"{n}.png")
        if n.startswith("g1"):
            im = im[40:720, 300:980]
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(cap, fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "l01_results.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved l01_results.png")


def curriculum():
    fig, ax = fig_ax(16, 7.4)
    pages = [("①", "소개 · 커리큘럼", "", C["grey"]), ("②", "환경 설정", "venv · Isaac Sim · Isaac Lab", C["grey"]),
             ("③", "Isaac Sim 물리엔진", "USD · PhysX · 공 낙하 · 로봇 4096 대", C["blue"]),
             ("④", "역진자 (1) 환경", "행동 · 관측 · 이벤트 · 보상 · 종료", C["yellow"]),
             ("⑤", "역진자 (2) 학습 · 평가", "PPO · 로그 · 학습 곡선 · play", C["yellow"]),
             ("⑥", "Pinky Pro (1) 도로 인식", "핑크시티 · 자동 라벨 · YOLO 세그먼테이션", C["purple"]),
             ("⑦", "Pinky Pro (2) 순환 도로 주행", "도로 시야 8 × 16 · 2.9분 학습 · 카메라 + YOLO 평가", C["purple"]),
             ("⑧", "Go2 (1) USD 에서 서기까지", "링크 · 관절 순서 · PD 모터", C["teal"]),
             ("⑨", "Go2 (2) 걷기 환경", "속도 명령 · 관측 48 · 보상 설계", C["teal"]),
             ("⑩", "Go2 (3) 학습 · 평가", "2.7분 학습 · 명령별 속도 · 영상", C["teal"]),
             ("⑪", "G1 (1) 앞으로 걷기", "모션 모방 · RSI · 30분 학습", C["red"]),
             ("⑫", "G1 (2) 이소룡 쿵푸", "걷기 모델로 쿵푸? → 쿵푸 학습 → 추론", C["red"])]
    for i, (num, title, what, fc) in enumerate(pages):
        y = 6.75 - i * 0.6
        rbox(ax, 0.3, y, 0.8, 0.5, num, fc=fc, tfs=12, lw=1.0)
        rbox(ax, 1.3, y, 5.4, 0.5, title, fc=fc, tfs=11, lw=1.0)
        if what:
            rbox(ax, 6.9, y, 8.8, 0.5, what, fc="white", tfs=10.5, lw=1.0)
    save(fig, "l01_curriculum.png")


def timetable():
    fig, ax = fig_ax(16, 6.6)
    days = [("1일차", [("09:30", 20, "①", C["grey"]), ("09:50", 60, "② 설치", C["grey"]), ("10:50", 40, "③ 물리", C["blue"]),
                      ("11:30", 30, "④ 역진자", C["yellow"]), ("12:00", 60, "점심", "white"), ("13:00", 40, "⑤ 학습", C["yellow"]),
                      ("13:40", 70, "⑥ Pinky 도로 인식", C["purple"]), ("14:50", 10, "휴식", "white"),
                      ("15:00", 70, "⑦ Pinky 주행", C["purple"]), ("16:10", 50, "⑧ Go2 USD", C["teal"]),
                      ("17:00", 30, "정리", C["grey"])], "17:30"),
            ("2일차", [("09:30", 50, "⑨ Go2 환경", C["teal"]), ("10:20", 40, "⑩ Go2 학습", C["teal"]),
                           ("11:00", 10, "휴식", "white"), ("11:10", 50, "⑪ G1 걷기", C["red"]),
                           ("12:00", 50, "⑫ G1 쿵푸", C["red"]), ("12:50", 10, "정리", C["grey"])], "13:00")]
    scale = 15.4 / 480
    for d, (name, slots, end) in enumerate(days):
        y = 4.2 - d * 2.3
        note(ax, 0.3, y + 1.95, name, bold=True, color=INK, fs=12)
        x = 0.3
        for t, m, label, fc in slots:
            w = m * scale
            rbox(ax, x, y, w - 0.03, 1.0, label if w > 0.6 else "", fc=fc, tfs=10, lw=1.0)
            ax.text(x, y + 1.1, t, fontsize=9, color=INK, rotation=40, ha="left", va="bottom")
            x += w
        ax.text(x, y + 1.1, end, fontsize=9, color=INK, rotation=40, ha="left", va="bottom")
    note(ax, 0.3, 1.25, "학습을 걸어 두는 때:  ⑪ 시작할 때 G1 걷기 학습 (약 30분),  ⑫ 시작할 때 G1 쿵푸 학습 (약 40분)", fs=10.5)
    note(ax, 0.3, 0.75, "끝나기를 기다리지 않고, checkpoints/ 의 미리 학습한 정책으로 평가 진행", fs=10.5)
    save(fig, "l01_timetable.png")


def versions():
    fig, ax = fig_ax(16, 4.75)
    tiles(ax, [("OS", ["Ubuntu 24.04.4 LTS"], C["grey"]),
               ("GPU · 드라이버", ["NVIDIA RTX 5060 Ti 16 GB", "드라이버 580.126"], C["grey"]),
               ("Python", ["3.11.15 (venv)"], C["blue"]),
               ("PyTorch", ["2.7.0 + CUDA 12.8"], C["blue"]),
               ("Isaac Sim", ["5.1.0 (pip)"], C["teal"]),
               ("Isaac Lab", ["2.3.2 (git tag v2.3.2)"], C["teal"]),
               ("RSL-RL", ["3.1.2 (PPO)"], C["yellow"]),
               ("설치 위치", ["~/rl_shortcut 폴더 하나", "다른 Isaac Sim 에 영향 없음"], C["yellow"])],
          cols=4, h=1.5, tfs=12.5, fs=10.5)
    note(ax, 0.3, 0.4, "이 조합으로 설치 · 학습 · 평가를 모두 확인 (② 페이지에서 그대로 설치).", fs=11)
    save(fig, "l01_versions.png")


if __name__ == "__main__":
    for f in (overview, il_vs_rl, curriculum, timetable, versions):
        f()
    if (RAW / "g1_play_bruce_lee_4.png").exists():
        results()
