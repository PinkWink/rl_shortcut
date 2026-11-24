"""Figures for the Pinky Pro pages (⑥ City of Pink + YOLO road segmentation, ⑦ driving the ring road).
Output: docs/lecture/l11_*.png, l12_*.png

  .venv/bin/python docs/lecture/draw_lesson11_12_figures.py    (after render_pinky.py, pinky_dataset.py, pinky_yolo.py)
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import BLUE, GREEN, INK, OUT, RED, C, arrow, chain, fig_ax, note, rbox, save, tiles  # noqa: E402

RAW = OUT / "raw"
ROOT = OUT.parents[1]


def shots(names, captions, out, width=16, crop=None, folder=RAW):
    imgs = [plt.imread(folder / n) for n in names]
    if crop:
        imgs = [im[crop[1]:crop[3], crop[0]:crop[2]] for im in imgs]
    h, w = imgs[0].shape[:2]
    cell = width / len(imgs)
    fig = plt.figure(figsize=(width, cell * h / w + 0.45), dpi=150)
    for k, (im, cap) in enumerate(zip(imgs, captions)):
        ax = fig.add_subplot(1, len(imgs), k + 1)
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#999999")
        ax.set_xlabel(cap, fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


# ---------------------------------------------------------------------------- page ⑥
def task():
    shots(["pinky_city_overview.png", "pinky_city_top.png"],
          ["핑크시티 (3 × 3 m) 와 Pinky Pro", "위에서 본 모습: 바깥 순환 도로를 반시계 방향으로"], "l11_task.png")


def pipeline():
    fig, ax = fig_ax(16, 3.4)
    chain(ax, [("카메라 영상", ["640 × 360", "Pinky Pro 앞 카메라"]),
               ("YOLO 세그먼테이션", ["yolo11n-seg", "도로 · 보도 찾기"]),
               ("도로 마스크", ["픽셀마다", "도로 = 1 · 아니면 0"]),
               ("도로 시야", ["아래쪽 55 % 를", "8 × 16 칸으로 축소"]),
               ("정책 (MLP)", ["관측 130개", "→ 행동 2개"]),
               ("바퀴 속도", ["왼쪽 · 오른쪽", "최대 0.25 m/s"])],
          0.8, 2.1, colors=[C["grey"], C["purple"], C["purple"], C["yellow"], C["blue"], C["teal"]], tfs=11.5, fs=9.6)
    note(ax, 0.3, 0.35, "⑥ 페이지: 카메라 영상 → 도로 시야 (YOLO)      ⑦ 페이지: 도로 시야 → 바퀴 속도 (강화학습)", fs=11)
    save(fig, "l11_pipeline.png")


def robot():
    shots(["pinky_robot.png", "pinky_street.png"], ["Pinky Pro: 구동 바퀴 2개 + 뒤 캐스터", "오른쪽 차로를 달리는 모습"],
          "l11_robot.png")


def camera():
    shots(["pinky_camera_a.png", "pinky_camera_b.png", "pinky_camera_c.png"],
          ["직선 구간", "모퉁이 앞", "모퉁이"], "l11_camera.png")


def autolabel():
    fig, ax = fig_ax(16, 3.6)
    chain(ax, [("카메라 영상 + 거리", ["Isaac Sim 은 픽셀마다", "거리(depth)도 함께 제공"]),
               ("픽셀이 보는 점", ["거리 + 카메라 위치", "→ 도시 안의 (x, y, z)"]),
               ("바닥인가?", ["z ≈ 0 이면 바닥", "아니면 배경 (벽 · 건물)"]),
               ("바닥 정답 지도", ["floor_labels.png 에서", "그 (x, y) 의 종류 읽기"]),
               ("YOLO 라벨", ["도로 · 보도 영역의", "윤곽선(다각형)"])],
          0.9, 2.2, colors=[C["grey"], C["blue"], C["blue"], C["yellow"], C["green"]], tfs=11.5, fs=9.6)
    note(ax, 0.3, 0.4, "사람이 영상에 색칠할 필요 없음 · 시뮬레이터가 정답을 알고 있기 때문", fs=11)
    save(fig, "l11_autolabel.png")


def dataset():
    folder = ROOT / "datasets/pinky_road/preview"
    names = ["00000.jpg", "00003.jpg", "00006.jpg"]
    fig = plt.figure(figsize=(16, 16 * 3 * 360 / 1920 + 0.5), dpi=150)
    for k, n in enumerate(names):
        ax = fig.add_subplot(3, 1, k + 1)
        ax.imshow(plt.imread(folder / n)); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("왼쪽: 카메라 영상      가운데: 자동 라벨 (회색 = 도로 · 분홍 = 보도 · 검정 = 배경)      오른쪽: 겹친 것",
                  fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "l11_dataset.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved l11_dataset.png")


def yolo_curve():
    run = sorted((ROOT / "logs/pinky_yolo").glob("*/results.csv"))[-1]
    rows = list(csv.DictReader(run.open()))
    rows = [{k.strip(): float(v) for k, v in r.items()} for r in rows]
    epoch = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.2), dpi=150)
    axes[0].plot(epoch, [r["train/seg_loss"] for r in rows], color=BLUE, lw=1.8, label="학습")
    axes[0].plot(epoch, [r["val/seg_loss"] for r in rows], color=RED, lw=1.8, label="검증")
    axes[0].set_title("세그먼테이션 손실", fontsize=12, color=INK); axes[0].legend(fontsize=10)
    axes[1].plot(epoch, [r["metrics/mAP50(M)"] for r in rows], color=GREEN, lw=1.8, label="mAP50")
    axes[1].plot(epoch, [r["metrics/mAP50-95(M)"] for r in rows], color="#6A1B9A", lw=1.8, label="mAP50-95")
    axes[1].set_title("마스크 정확도 (검증 40장)", fontsize=12, color=INK); axes[1].legend(fontsize=10)
    for ax in axes:
        ax.set_xlabel("epoch", fontsize=10); ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "l11_yolo_curve.png", facecolor="white")
    plt.close(fig)
    print("saved l11_yolo_curve.png")


def check():
    im = plt.imread(ROOT / "outputs/pinky_yolo_check.jpg")[:3 * 360]
    fig = plt.figure(figsize=(16, 16 * im.shape[0] / im.shape[1] + 0.5), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("왼쪽: 카메라 영상      가운데: YOLO 가 찾은 도로 (초록)      오른쪽: 도로 시야 8 × 16 (흰색 = 도로)",
                  fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "l11_check.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved l11_check.png")


def view():
    """Camera image with the view area marked -> the 8 x 16 road view."""
    data = np.load(RAW / "pinky_views.npz")
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.6), dpi=150)
    im = (plt.imread(RAW / "pinky_camera_b.png")[..., :3] * 255).astype(np.uint8)
    seen = im.copy()
    m = data["mask_b"]
    seen[m] = (0.45 * seen[m] + 0.55 * np.array([80, 200, 80])).astype(np.uint8)
    axes[0].imshow(im); axes[0].set_xlabel("① 카메라 영상 640 × 360", fontsize=12, color=INK)
    axes[1].imshow(seen)
    axes[1].add_patch(plt.Rectangle((1, 162), 637, 196, fill=False, ec="#FFD600", lw=2.5))
    for c in range(1, 16):
        axes[1].plot([c * 40, c * 40], [162, 359], color="#FFD600", lw=0.6)
    for r in range(1, 8):
        axes[1].plot([0, 639], [162 + r * 24.75] * 2, color="#FFD600", lw=0.6)
    axes[1].set_xlabel("② 도로 마스크 (초록) · 노란 칸 = 아래 55 %", fontsize=12, color=INK)
    axes[2].imshow(data["yolo_b"], cmap="gray", vmin=0, vmax=1, aspect=198 / 640 * 16 / 8)
    axes[2].set_xlabel("③ 도로 시야 8 × 16 = 칸마다 도로의 비율", fontsize=12, color=INK)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "l11_view.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved l11_view.png")


# ---------------------------------------------------------------------------- page ⑦
def train_vs_play():
    fig, ax = fig_ax(16, 5.0)
    rbox(ax, 0.3, 3.05, 1.9, 1.5, "학습", ["train.py", "로봇 1024 대"], fc=C["yellow"], tfs=12, fs=9.8)
    chain(ax, [("로봇의 위치 · 방향", ["시뮬레이터가 아는 값"]), ("바닥 정답 지도", ["칸마다 도로인지 계산"]),
               ("도로 시야 8 × 16", ["카메라 · 렌더링 없음"]), ("정책", ["→ 바퀴 속도"])],
          3.05, 1.5, x0=2.6, colors=[C["grey"], C["yellow"], C["green"], C["blue"]], tfs=11.5, fs=9.8)
    rbox(ax, 0.3, 0.95, 1.9, 1.5, "평가", ["play.py", "--perception yolo"], fc=C["purple"], tfs=12, fs=9.8)
    chain(ax, [("카메라 영상", ["로봇 카메라 렌더링"]), ("YOLO 세그먼테이션", ["⑥ 페이지에서 학습"]),
               ("도로 시야 8 × 16", ["같은 모양의 관측"]), ("정책", ["학습한 그대로"])],
          0.95, 1.5, x0=2.6, colors=[C["grey"], C["purple"], C["green"], C["blue"]], tfs=11.5, fs=9.8)
    note(ax, 0.3, 0.4, "두 길이 같은 관측(도로 시야)을 만듦 → 지도로 학습한 정책을 카메라 + YOLO 로 그대로 사용", fs=11)
    save(fig, "l12_train_vs_play.png")


def parts():
    fig, ax = fig_ax(16, 2.9)
    tiles(ax, [("행동 2", ["왼쪽 · 오른쪽 바퀴 속도", "1.0 = 0.25 m/s"], C["teal"]),
               ("관측 130", ["도로 시야 8 × 16 = 128", "직전 행동 2"], C["green"]),
               ("이벤트", ["리셋: 순환 도로의", "아무 곳 · 조금 틀어서"], C["grey"]),
               ("보상", ["도로를 따라 가는 속도", "오른쪽 차로 가운데"], C["yellow"]),
               ("종료", ["도로를 벗어남", "또는 30초"], C["red"])],
          cols=5, h=1.75, tfs=12, fs=10)
    note(ax, 0.3, 0.35, "역진자와 같은 다섯 부분 · 위치 · 방향은 관측에 없음 (보상 · 종료 계산에만 사용)", fs=11)
    save(fig, "l12_parts.png")


def lane():
    floor = plt.imread(ROOT / "assets/pink_city/textures/floor.png")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2), dpi=150)
    t = np.linspace(0, 2 * np.pi, 2000)

    def ring(offset):
        s, r = 1.05 - 0.3, 0.3 + offset
        pts = []
        for k, (cx, cy) in enumerate(((s, s), (-s, s), (-s, -s), (s, -s))):
            a = np.linspace(k * np.pi / 2, (k + 1) * np.pi / 2, 40)
            pts.append(np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1))
        p = np.concatenate(pts)
        return np.concatenate([p, p[:1]])

    for ax in axes:
        ax.imshow(floor, extent=(-1.5, 1.5, -1.5, 1.5))
        ax.set_xlabel("x [m]", fontsize=10); ax.set_ylabel("y [m]", fontsize=10)
    lane_c, lo, hi = ring(0.1), ring(-0.16), ring(0.16)
    axes[0].plot(lane_c[:, 0], lane_c[:, 1], color="#00C853", lw=2.5, label="차로 가운데 (보상이 가장 큰 곳)")
    axes[0].plot(lo[:, 0], lo[:, 1], color="#FF1744", lw=1.5, ls="--", label="종료 경계 (도로 가장자리 4 cm 앞)")
    axes[0].plot(hi[:, 0], hi[:, 1], color="#FF1744", lw=1.5, ls="--")
    axes[0].annotate("", xy=(-1.15, -0.45), xytext=(-1.15, 0.25), arrowprops=dict(arrowstyle="-|>", color="#00C853", lw=3))
    axes[0].legend(fontsize=10, loc="center")
    axes[0].set_title("보상과 종료의 기준: 바깥 순환 도로의 오른쪽 차로 (반시계 방향)", fontsize=12, color=INK)
    track = np.load(RAW / "pinky_track.npy")
    axes[1].plot(track[:, 0], track[:, 1], color="#2962FF", lw=2.2, label="YOLO 로 달린 36초의 궤적")
    axes[1].plot(track[0, 0], track[0, 1], "o", color="#2962FF", ms=9)
    axes[1].legend(fontsize=10, loc="center")
    axes[1].set_title("평가: 카메라 + YOLO 로 한 바퀴", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "l12_lane.png", facecolor="white")
    plt.close(fig)
    print("saved l12_lane.png")


def rewards():
    fig, ax = fig_ax(16, 4.2)
    rows = [("progress", "+8.0", "도로 방향 속도 [m/s]", "순환 도로를 따라 반시계 방향으로 빨리"),
            ("in_lane", "+1.0", "exp(-(차로 오차 / 5 cm)²)", "오른쪽 차로 가운데면 1 · 멀어지면 0"),
            ("action_rate", "-0.05", "(행동 변화)²", "바퀴 속도를 급하게 바꾸지 않게"),
            ("terminating", "-5.0", "도로를 벗어난 스텝", "벗어나면 큰 감점")]
    x, w = [0.3, 3.5, 5.4, 10.2], [3.0, 1.7, 4.6, 5.5]
    for j, t in enumerate(["항목", "가중치", "값", "뜻"]):
        rbox(ax, x[j], 3.35, w[j], 0.56, t, fc=C["grey"], tfs=11, lw=1.0)
    for i, r in enumerate(rows):
        y = 2.65 - i * 0.66
        for j, t in enumerate(r):
            fc = (C["green"] if r[1].startswith("+") else C["red"]) if j == 1 else "white"
            rbox(ax, x[j], y, w[j], 0.54, t, fc=fc, tfs=10.5, lw=1.0)
    note(ax, 0.3, 0.25, "보상에는 로봇의 위치가 쓰이지만 관측에는 없음 → 정책은 도로 시야만 보고 이 점수를 얻는 법을 찾음", fs=10.5)
    save(fig, "l12_rewards.png")


def train_scene():
    shots(["pinky_train.png"], ["학습 장면: 도시마다 로봇 한 대 · 바닥 그림만 있고 벽 · 건물은 없음 (16개만 표시)"],
          "l12_train_scene.png", width=11)


def yolo_drive():
    names = [f"pinky_yolo_{k}.png" for k in range(6)]
    caps = [f"{t} 초" for t in (2, 8, 14, 20, 26, 32)]
    fig = plt.figure(figsize=(16, 3 * (8 * 720 / 1920 + 0.42)), dpi=150)
    for k, (n, cap) in enumerate(zip(names, caps)):
        ax = fig.add_subplot(3, 2, k + 1)
        ax.imshow(plt.imread(RAW / n)); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(cap, fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "l12_yolo_drive.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved l12_yolo_drive.png")


def progress():
    """play.py --perception yolo, 16 robots x 30 s, per checkpoint (measured on the course PC)."""
    its = ["0", "50", "100", "150", "299"]
    speed, error, fails = [0.038, 0.181, 0.207, 0.215, 0.239], [9.1, 8.3, 10.0, 9.7, 2.6], [79, 3, 0, 1, 0]
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.9), dpi=150)
    for ax, values, title, color, fmt in ((axes[0], speed, "도로 방향 속도 [m/s]", BLUE, "{:.3f}"),
                                          (axes[1], error, "차로 오차 [cm]", "#EF6C00", "{:.1f}"),
                                          (axes[2], fails, "도로 이탈 횟수 (16대 × 30초)", RED, "{}")):
        bars = ax.bar(its, values, color=color, width=0.6)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom", fontsize=10)
        ax.set_title(title, fontsize=12, color=INK); ax.set_xlabel("iteration", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(values) * 1.18)
    fig.tight_layout()
    fig.savefig(OUT / "l12_progress.png", facecolor="white")
    plt.close(fig)
    print("saved l12_progress.png")


if __name__ == "__main__":
    for f in (task, pipeline, robot, camera, autolabel, dataset, yolo_curve, check, view,
              train_vs_play, parts, lane, rewards, train_scene, yolo_drive, progress):
        f()
