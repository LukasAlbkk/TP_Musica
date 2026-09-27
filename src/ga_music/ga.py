"""The genetic algorithm loop: selection, crossover, mutation, elitism,
early stopping, and per-generation fitness bookkeeping for the report plot."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import fitness as fitness_mod
from . import theory
from .genome import GenomeSpec, build_spec, random_genome


@dataclass
class GenerationStats:
    generation: int
    best: float
    average: float
    worst: float


@dataclass
class GARun:
    best_genome: np.ndarray
    best_fitness: float
    best_components: dict
    history: list[GenerationStats] = field(default_factory=list)
    stopped_early_at: int | None = None


def _tournament_select(pop, fits, rng, k) -> np.ndarray:
    idx = rng.integers(0, len(pop), size=k)
    winner = idx[np.argmax(fits[idx])]
    return pop[winner]


def _crossover(parent_a, parent_b, rng, rate) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() > rate:
        return parent_a.copy(), parent_b.copy()
    length = len(parent_a)
    point = rng.integers(1, length)
    child_a = np.concatenate([parent_a[:point], parent_b[point:]])
    child_b = np.concatenate([parent_b[:point], parent_a[point:]])
    return child_a, child_b


def _mutate(genome, rng, rate, num_symbols) -> np.ndarray:
    mask = rng.random(len(genome)) < rate
    if mask.any():
        genome = genome.copy()
        genome[mask] = rng.integers(0, num_symbols, size=mask.sum())
    return genome


def run_ga(config, progress_callback=None) -> GARun:
    rng = np.random.default_rng(config.seed)
    spec = build_spec(config)
    root_pc = theory.note_name_to_pitch_class(config.key_root)
    progression = theory.build_progression(
        root_pc, config.key_mode, config.progression_name, config.num_bars
    )

    population = [random_genome(spec, rng) for _ in range(config.population_size)]

    def evaluate_all(pop):
        results = [fitness_mod.evaluate(g, spec, config, progression) for g in pop]
        fits = np.array([r[0] for r in results])
        comps = [r[1] for r in results]
        return fits, comps

    fits, comps = evaluate_all(population)
    history: list[GenerationStats] = []
    best_idx = int(np.argmax(fits))
    best_genome = population[best_idx].copy()
    best_fitness = float(fits[best_idx])
    best_components = comps[best_idx]
    stalled_for = 0
    stopped_early_at = None

    for gen in range(config.num_generations):
        history.append(GenerationStats(gen, float(fits.max()), float(fits.mean()), float(fits.min())))
        if progress_callback:
            progress_callback(gen, history[-1])

        order = np.argsort(-fits)
        elites = [population[i].copy() for i in order[: config.elitism_count]]

        children = []
        while len(children) < config.population_size - len(elites):
            parent_a = _tournament_select(population, fits, rng, config.tournament_size)
            parent_b = _tournament_select(population, fits, rng, config.tournament_size)
            child_a, child_b = _crossover(parent_a, parent_b, rng, config.crossover_rate)
            child_a = _mutate(child_a, rng, config.mutation_rate, spec.num_symbols)
            child_b = _mutate(child_b, rng, config.mutation_rate, spec.num_symbols)
            children.extend([child_a, child_b])
        children = children[: config.population_size - len(elites)]

        population = elites + children
        fits, comps = evaluate_all(population)

        gen_best_idx = int(np.argmax(fits))
        if fits[gen_best_idx] > best_fitness:
            best_fitness = float(fits[gen_best_idx])
            best_genome = population[gen_best_idx].copy()
            best_components = comps[gen_best_idx]
            stalled_for = 0
        else:
            stalled_for += 1

        if stalled_for >= config.early_stop_patience:
            stopped_early_at = gen + 1
            break

    history.append(GenerationStats(len(history), float(fits.max()), float(fits.mean()), float(fits.min())))

    return GARun(
        best_genome=best_genome,
        best_fitness=best_fitness,
        best_components=best_components,
        history=history,
        stopped_early_at=stopped_early_at,
    )
