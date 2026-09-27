"""Plotting/logging helpers used to produce the figures required in the
ISMIR summary (fitness-over-generations is mandatory for the GA track)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_fitness_history(history, out_path, title: str = "Fitness over generations") -> None:
    gens = [h.generation for h in history]
    best = [h.best for h in history]
    avg = [h.average for h in history]
    worst = [h.worst for h in history]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(gens, best, label="best", linewidth=2)
    ax.plot(gens, avg, label="average", linewidth=1.5)
    ax.plot(gens, worst, label="worst", linewidth=1, alpha=0.6)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_multi_run_comparison(runs: dict, out_path, title: str = "Best fitness: hyperparameter comparison") -> None:
    """`runs` maps a label -> history (list of GenerationStats)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, history in runs.items():
        gens = [h.generation for h in history]
        best = [h.best for h in history]
        ax.plot(gens, best, label=label, linewidth=1.8)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
