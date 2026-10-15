"""Lay out the rendered frames (docs/lecture/raw/*.png from render_sim.py) into page figures.

  python3 docs/lecture/draw_renders.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import INK, OUT, C, fig_ax, note, rbox, save  # noqa: E402

RAW = OUT / "raw"


def strip(names, captions, out, cols=None, width=16, crop=None, title=None):
    """Frames side by side with a caption under each. crop=(x0, y0, x1, y1) in pixels of the 1280x720 frames."""
    cols = cols or len(names)
    rows = (len(names) + cols - 1) // cols
    imgs = [plt.imread(RAW / f"{n}.png") for n in names]
    if crop:
        x0, y0, x1, y1 = crop
        imgs = [im[y0:y1, x0:x1] for im in imgs]
    h, w = imgs[0].shape[:2]
    cell_w = width / cols
    cell_h = cell_w * h / w
    fig = plt.figure(figsize=(width, rows * (cell_h + 0.45) + (0.4 if title else 0)), dpi=150)
    for k, (im, cap) in enumerate(zip(imgs, captions)):
        ax = fig.add_subplot(rows, cols, k + 1)
        ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#999999")
        ax.set_xlabel(cap, fontsize=12, color=INK)
    if title:
        fig.suptitle(title, fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def image_with_table(name, rows, out, title, crop=None):
    fig, ax = fig_ax(16, 5.2)
    im = plt.imread(RAW / f"{name}.png")
    if crop:
        x0, y0, x1, y1 = crop
        im = im[y0:y1, x0:x1]
    h, w = im.shape[:2]
    iw = min(8.6, 4.8 * w / h)              # fit 8.6 wide or 4.8 high
    ax.imshow(im, extent=(0.2, 0.2 + iw, 0.2, 0.2 + iw * h / w), zorder=1)
    tx = 0.2 + iw + 0.6                     # table right of the image
    note(ax, tx, 4.85, title, bold=True, color=INK, fs=12)
    for i, (k, v, fc) in enumerate(rows):
        y = 4.15 - i * 0.62
        rbox(ax, tx, y, 2.4, 0.52, k, fc=fc, tfs=10.5, lw=1.0)
        rbox(ax, tx + 2.6, y, 4.6, 0.52, v, fc="white", tfs=10, lw=1.0)
    ax.set_xlim(0, tx + 7.3); ax.set_ylim(0, 5.2)
    save(fig, out)


def main():
    strip(["drop_0", "drop_1"], ["시작: 1 m 위 (빨강 e=0, 노랑 e=0.5, 파랑 e=0.9)", "0.8초 뒤: 첫 바운드 이후"], "l03_drop.png")
    strip(["parallel"], ["Go2 16대 (env_0 ~ env_15), 1.5 m 간격"], "l03_parallel.png", width=11)
    strip(["pendulum_kit_0"], ["PinkLAB 회전형 역진자 키트 (assets/pinklab_pendulum)"], "l04_kit.png", width=11)
    strip(["pendulum_0"], ["학습한 정책: 16개 키트가 진자를 세우고 있다 (시작 2초 뒤)"], "l05_play.png", width=11)
    strip([f"pendulum_balance_{k}" for k in range(6)], [f"{t:.1f} s" for t in (0, 0.1, 0.2, 0.3, 0.5, 1.0)],
          "l05_balance.png", cols=6, crop=(400, 40, 880, 620))
    strip(["stand"], ["왼쪽부터 kp 25 · kp 5 · kp 0, 떨어뜨리고 2초 뒤"], "l06_stand.png", width=13)
    strip(["go2_play_0"], ["16대가 각자 받은 명령을 따라 걷는다 (초록 화살표 = 명령, 파랑 = 실제 속도)"], "l08_play.png", width=12)
    strip([f"go2_forward_{k}" for k in range(4)], ["1.0 s", "1.4 s", "1.8 s", "2.2 s"], "l08_video.png",
          crop=(160, 200, 1120, 720))
    strip([f"g1_ref_walk_{k}" for k in range(6)], [f"{t:.1f} s" for t in (0, 0.6, 1.2, 1.8, 2.4, 3.0)],
          "l09_ref_walk.png", cols=6, crop=(360, 60, 920, 720))
    strip([f"g1_ref_bruce_lee_{k}" for k in range(6)], [f"{t:.1f} s" for t in (0, 1.5, 3.5, 5.5, 7.5, 9.5)],
          "l10_ref_bruce_lee.png", cols=6, crop=(330, 40, 950, 720))
    strip([f"g1_play_walk_{k}" for k in range(6)], [f"{t} s" for t in (1, 2, 3, 4, 5, 6)], "l09_play.png",
          cols=6, crop=(380, 40, 900, 720))
    strip([f"g1_walk_on_bruce_lee_{k}" for k in range(6)], [f"{t:.1f} s" for t in (0.1, 0.3, 0.5, 0.7, 0.9, 1.1)],
          "l10_walk_on_bruce_lee.png", cols=6, crop=(330, 40, 950, 720))
    if (RAW / "g1_play_bruce_lee_4.png").exists():
        strip([f"g1_play_bruce_lee_{k}" for k in range(5)], [f"{t} s" for t in (1.5, 3.5, 5.5, 7.5, 9.5)],
              "l10_play_bruce_lee.png", cols=5, crop=(330, 40, 950, 720))
    legs = C["yellow"]
    image_with_table("go2", [("다리 4개", "FL · FR · RL · RR (앞왼 · 앞오 · 뒤왼 · 뒤오)", C["grey"]),
                             ("hip", "다리를 옆으로 벌리기 (±60°)", legs),
                             ("thigh", "허벅지 앞뒤 (앞 -90°..200°, 뒤 -30°..260°)", legs),
                             ("calf", "무릎 (-156°..-48°, 항상 접힘)", legs),
                             ("관절 이름", "FL_hip_joint, RR_calf_joint, ... 12개", C["blue"]),
                             ("무게", "약 15 kg (몸통 6.9 kg)", C["blue"])],
                     "l06_go2.png", "Go2 의 관절 12개 = 다리 4 × 3", crop=(80, 60, 1200, 700))
    image_with_table("g1", [("다리 12", "좌우 각 6: 고관절 3 · 무릎 · 발목 2", legs),
                            ("허리 1", "torso_joint (몸통 회전)", legs),
                            ("팔 10", "좌우 각 5: 어깨 3 · 팔꿈치 2", legs),
                            ("손가락 14", "이 과정에서는 고정 (기본 자세)", C["grey"]),
                            ("키 · 무게", "약 1.3 m · 약 35 kg", C["blue"]),
                            ("USD", "g1_minimal.usd (충돌 모양을 줄인 판)", C["blue"])],
                     "l09_g1.png", "정책이 움직이는 관절 23개", crop=(340, 0, 940, 720))


if __name__ == "__main__":
    main()
