"""Learning curves from a TensorBoard log (logs/<experiment>/<run>/events.out.tfevents.*).

Run with the course venv (tensorboard is installed there):
  .venv/bin/python docs/lecture/curves.py cartpole l05_curve.png Train/mean_reward Train/mean_episode_length
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import INK, OUT  # noqa: E402

ROOT = OUT.parents[1]
COLORS = ["#1565C0", "#C62828", "#2E7D32", "#6A1B9A", "#EF6C00"]
LABELS = {"Train/mean_reward": "보상 합 (에피소드 평균)", "Train/mean_episode_length": "에피소드 길이 (스텝)"}


def scalars(experiment, run=None):
    runs = sorted((ROOT / "logs" / experiment).glob("*/events.out.tfevents.*"))
    path = [r for r in runs if run is None or r.parent.name == run][-1]
    ea = EventAccumulator(str(path), size_guidance={"scalars": 0})
    ea.Reload()
    return {t: ([e.step for e in ea.Scalars(t)], [e.value for e in ea.Scalars(t)]) for t in ea.Tags()["scalars"]}


def plot(experiment, out, tags, labels=None, run=None, width=16, height=4.2):
    data = scalars(experiment, run)
    fig, axes = plt.subplots(1, len(tags), figsize=(width, height), dpi=150)
    axes = [axes] if len(tags) == 1 else axes
    for i, (ax, tag) in enumerate(zip(axes, tags)):
        steps, values = data[tag]
        ax.plot(steps, values, color=COLORS[i % len(COLORS)], lw=1.8)
        ax.set_title((labels or {}).get(tag, LABELS.get(tag, tag)), fontsize=12, color=INK)
        ax.set_xlabel("iteration", fontsize=10)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / out, facecolor="white")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    plot(sys.argv[1], sys.argv[2], sys.argv[3:])
