#!/usr/bin/env python
"""Run one GA experiment and save the resulting song (MIDI + WAV), its
fitness-over-generations plot, and the full config/log as JSON.

Example:
    python scripts/generate.py --name song1 --seed 1 --key-root A --key-mode natural_minor \
        --progression i-VI-III-VII --tempo 132 --mutation-rate 0.06
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ga_music.config import GAConfig  # noqa: E402
from ga_music.ga import run_ga  # noqa: E402
from ga_music.genome import build_spec  # noqa: E402
from ga_music.arrangement import build_song  # noqa: E402
from ga_music.synth import render_wav  # noqa: E402
from ga_music.report import plot_fitness_history  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="song")
    p.add_argument("--out-dir", default="outputs/demo")
    p.add_argument("--key-root", default="A")
    p.add_argument("--key-mode", default="natural_minor", choices=["natural_minor", "major"])
    p.add_argument("--progression", default="i-VI-III-VII")
    p.add_argument("--tempo", type=float, default=132.0)
    p.add_argument("--num-bars", type=int, default=16)
    p.add_argument("--steps-per-bar", type=int, default=8, choices=[8, 16])
    p.add_argument("--lead-scale", default="minor_pentatonic", choices=["minor_pentatonic", "major_pentatonic"])
    p.add_argument("--population-size", type=int, default=80)
    p.add_argument("--generations", type=int, default=200)
    p.add_argument("--crossover-rate", type=float, default=0.85)
    p.add_argument("--mutation-rate", type=float, default=0.06)
    p.add_argument("--tournament-size", type=int, default=3)
    p.add_argument("--elitism", type=int, default=2)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--soundfont", default=None, help="Optional path to a .sf2 for a nicer FluidSynth render")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = GAConfig(
        key_root=args.key_root,
        key_mode=args.key_mode,
        progression_name=args.progression,
        tempo_bpm=args.tempo,
        num_bars=args.num_bars,
        steps_per_bar=args.steps_per_bar,
        lead_scale_mode=args.lead_scale,
        population_size=args.population_size,
        num_generations=args.generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        tournament_size=args.tournament_size,
        elitism_count=args.elitism,
        early_stop_patience=args.patience,
        seed=args.seed,
    )

    def progress(gen, stats):
        if not args.quiet and gen % 20 == 0:
            print(f"  gen {gen:4d}  best={stats.best:.3f}  avg={stats.average:.3f}")

    print(f"[{args.name}] running GA ({config.num_generations} generations max)...")
    result = run_ga(config, progress_callback=progress)
    stop_msg = (
        f"early-stopped at gen {result.stopped_early_at}"
        if result.stopped_early_at is not None
        else f"ran all {config.num_generations} generations"
    )
    print(f"[{args.name}] done ({stop_msg}). best fitness = {result.best_fitness:.3f}")
    print(f"[{args.name}] fitness breakdown: {result.best_components}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = build_spec(config)
    song = build_song(result.best_genome, spec, config)

    midi_path = out_dir / f"{args.name}.mid"
    wav_path = out_dir / f"{args.name}.wav"
    plot_path = out_dir / f"{args.name}_fitness.png"
    log_path = out_dir / f"{args.name}.json"

    song.write(str(midi_path))
    render_wav(song, wav_path, seed=config.seed, soundfont_path=args.soundfont)
    plot_fitness_history(result.history, plot_path, title=f"Fitness over generations — {args.name}")

    log = {
        "config": json.loads(config.to_json()),
        "best_fitness": result.best_fitness,
        "best_components": result.best_components,
        "stopped_early_at": result.stopped_early_at,
        "generations_run": len(result.history),
        "history": [h.__dict__ for h in result.history],
    }
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[{args.name}] wrote {midi_path}, {wav_path}, {plot_path}, {log_path}")


if __name__ == "__main__":
    main()
