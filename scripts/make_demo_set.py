#!/usr/bin/env python
"""Generates the demo set of songs required by the assignment: at least 3
pieces showing variation between them. This script produces 6, split in two
groups:

  * song1a / song1b: the exact same configuration, different random seeds
    -> demonstrates that the system produces multiple different pieces from
       the same configuration (stochastic population init + operators).
  * song2..song5: each changes exactly one hyperparameter with respect to
    song1a (mutation rate, key/scale/progression, rhythmic grid resolution,
    tempo+progression+mutation) -> demonstrates hyperparameter-driven
    variation, as required.

Outputs go to outputs/demo/ (MIDI, WAV, per-run fitness plot, JSON log) and a
combined best-fitness comparison plot + a summary table are written too.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ga_music.config import GAConfig  # noqa: E402
from ga_music.ga import run_ga  # noqa: E402
from ga_music.genome import build_spec  # noqa: E402
from ga_music.arrangement import build_song  # noqa: E402
from ga_music.synth import render_wav  # noqa: E402
from ga_music.report import plot_fitness_history, plot_multi_run_comparison  # noqa: E402

OUT_DIR = Path("outputs/demo")

PRESETS = {
    "song1a_baseline": dict(
        varied="baseline configuration (reference point)",
        overrides=dict(seed=1),
    ),
    "song1b_same_config_seed": dict(
        varied="seed only (10 -> 11), everything else identical to song1a",
        overrides=dict(seed=11),
    ),
    "song2_high_mutation": dict(
        varied="mutation_rate: 0.06 -> 0.20 (more exploration, less convergent riff)",
        overrides=dict(seed=2, mutation_rate=0.20),
    ),
    "song3_major_key": dict(
        varied="key_mode/progression/lead_scale: A natural-minor pentatonic -> E major pop-rock",
        overrides=dict(
            seed=3, key_root="E", key_mode="major", progression_name="I-V-vi-IV",
            lead_scale_mode="major_pentatonic", lead_low_midi=60, lead_high_midi=84,
        ),
    ),
    "song4_dense_16th_grid": dict(
        varied="steps_per_bar: 8 -> 16 (sixteenth-note grid, busier/thrashier riff)",
        overrides=dict(seed=4, steps_per_bar=16, key_root="E", progression_name="i-VII-VI-VII"),
    ),
    "song5_slow_blues_rock": dict(
        varied="tempo_bpm: 132 -> 96 and progression -> i-iv-v (slower blues-rock feel)",
        overrides=dict(seed=5, tempo_bpm=96.0, progression_name="i-iv-v", key_root="D", mutation_rate=0.03),
    ),
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    histories = {}

    for name, spec in PRESETS.items():
        config = GAConfig(**spec["overrides"])
        print(f"[{name}] varied hyperparameter: {spec['varied']}")
        print(f"[{name}] running GA...")
        result = run_ga(config)
        stop_msg = (
            f"early-stopped@{result.stopped_early_at}"
            if result.stopped_early_at is not None
            else "ran full budget"
        )
        print(f"[{name}] {stop_msg}, best fitness = {result.best_fitness:.3f}")

        genome_spec = build_spec(config)
        song = build_song(result.best_genome, genome_spec, config)

        midi_path = OUT_DIR / f"{name}.mid"
        wav_path = OUT_DIR / f"{name}.wav"
        plot_path = OUT_DIR / f"{name}_fitness.png"
        log_path = OUT_DIR / f"{name}.json"

        song.write(str(midi_path))
        render_wav(song, wav_path, seed=config.seed)
        plot_fitness_history(result.history, plot_path, title=f"Fitness over generations — {name}")

        log = {
            "varied_hyperparameter": spec["varied"],
            "config": json.loads(config.to_json()),
            "best_fitness": result.best_fitness,
            "best_components": result.best_components,
            "stopped_early_at": result.stopped_early_at,
            "generations_run": len(result.history),
            "history": [h.__dict__ for h in result.history],
        }
        log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

        histories[name] = result.history
        summary_rows.append({
            "name": name,
            "varied_hyperparameter": spec["varied"],
            "best_fitness": round(result.best_fitness, 4),
            "generations_run": len(result.history),
            "stopped_early": result.stopped_early_at is not None,
        })

    comparison_path = OUT_DIR / "fitness_comparison.png"
    plot_multi_run_comparison(histories, comparison_path)

    summary_path = OUT_DIR / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nWrote demo set to {OUT_DIR}/  (comparison plot: {comparison_path}, summary: {summary_path})")


if __name__ == "__main__":
    main()
