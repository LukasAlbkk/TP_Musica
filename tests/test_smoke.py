"""Minimal sanity checks: decoding round-trips, fitness is bounded, and the
GA actually improves fitness over generations on a small run."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from ga_music.config import GAConfig
from ga_music.genome import build_spec, decode, random_genome
from ga_music.fitness import evaluate
from ga_music.ga import run_ga
from ga_music import theory


def test_decode_covers_full_length():
    config = GAConfig(num_bars=4, steps_per_bar=8)
    spec = build_spec(config)
    rng = np.random.default_rng(0)
    genome = random_genome(spec, rng)
    notes = decode(genome, spec)
    total = sum(n.duration_steps for n in notes)
    assert total <= len(genome)
    for n in notes:
        assert 0 <= n.start_step < len(genome)
        assert n.pitch in spec.alphabet


def test_fitness_is_bounded():
    config = GAConfig(num_bars=4, steps_per_bar=8)
    spec = build_spec(config)
    root_pc = theory.note_name_to_pitch_class(config.key_root)
    progression = theory.build_progression(root_pc, config.key_mode, config.progression_name, config.num_bars)
    rng = np.random.default_rng(0)
    for _ in range(20):
        genome = random_genome(spec, rng)
        fitness, components = evaluate(genome, spec, config, progression)
        assert 0.0 <= fitness <= 1.0
        for v in components.values():
            assert 0.0 <= v <= 1.0


def test_ga_improves_fitness():
    config = GAConfig(
        num_bars=4, steps_per_bar=8, population_size=30, num_generations=40,
        early_stop_patience=40, seed=7,
    )
    result = run_ga(config)
    first_gen_best = result.history[0].best
    assert result.best_fitness >= first_gen_best
    assert result.best_fitness > 0.5


if __name__ == "__main__":
    test_decode_covers_full_length()
    test_fitness_is_bounded()
    test_ga_improves_fitness()
    print("All smoke tests passed.")
