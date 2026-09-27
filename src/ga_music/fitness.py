"""Heuristic fitness function for the evolved lead riff.

Every component is a music-theory rule of thumb for a driving 80s/90s rock
riff, normalized to [0, 1]; the total fitness is their weighted average. The
weights below are the defaults used in all reported runs; they can be
overridden per-run via `GAConfig.fitness_weights` (this is one of the
hyperparameters the assignment asks us to vary between generated pieces).
"""

from __future__ import annotations

from .genome import GenomeSpec, decode

DEFAULT_WEIGHTS = {
    "chord_tone": 0.25,
    "smoothness": 0.20,
    "density": 0.15,
    "repetition": 0.15,
    "cadence": 0.10,
    "syncopation": 0.15,
}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _triangular(x: float, lo: float, hi: float, slack: float = 0.25) -> float:
    """1.0 inside [lo, hi], decaying linearly to 0 over `slack` outside it."""
    if lo <= x <= hi:
        return 1.0
    if x < lo:
        return _clamp01(1.0 - (lo - x) / slack)
    return _clamp01(1.0 - (x - hi) / slack)


def _interval_score(interval: int) -> float:
    interval = abs(interval)
    if interval == 0:
        return 0.4
    if interval <= 2:
        return 1.0
    if interval <= 4:
        return 0.85
    if interval <= 7:
        return 0.5
    return 0.15


def score_chord_tone_alignment(notes, progression, steps_per_bar: int) -> float:
    strong = {0, steps_per_bar // 2}
    hits, total = 0, 0
    for n in notes:
        if (n.start_step % steps_per_bar) not in strong:
            continue
        total += 1
        bar = min(n.start_step // steps_per_bar, len(progression) - 1)
        if progression[bar].contains_pc(n.pitch % 12):
            hits += 1
    return hits / total if total else 0.0


def score_melodic_smoothness(notes) -> float:
    if len(notes) < 2:
        return 0.5
    intervals = [b.pitch - a.pitch for a, b in zip(notes, notes[1:])]
    return sum(_interval_score(iv) for iv in intervals) / len(intervals)


def score_density(notes, genome_length: int) -> float:
    sounding = sum(n.duration_steps for n in notes)
    density = sounding / genome_length
    return _triangular(density, 0.55, 0.85, slack=0.3)


def score_repetition(genome, steps_per_bar: int, num_bars: int) -> float:
    """Rewards the lead riff resembling its own first bar (motif repetition,
    typical of a rock riff) using a graded per-gene similarity rather than
    requiring an exact bar match -- exact matches are astronomically unlikely
    to survive mutation/crossover, which would make this term a dead signal."""
    if num_bars < 2:
        return 1.0
    first_bar = genome[0:steps_per_bar]
    similarities = []
    for b in range(1, num_bars):
        bar = genome[b * steps_per_bar:(b + 1) * steps_per_bar]
        similarities.append(sum(1 for a, c in zip(first_bar, bar) if a == c) / steps_per_bar)
    fraction = sum(similarities) / len(similarities)
    return _triangular(fraction, 0.35, 0.75, slack=0.3)


def score_cadence(notes, final_chord) -> float:
    if not notes:
        return 0.0
    pc = notes[-1].pitch % 12
    if pc == final_chord.root_pc:
        return 1.0
    if pc in (final_chord.third_pc, final_chord.fifth_pc):
        return 0.7
    return 0.2


def score_syncopation(notes, steps_per_bar: int) -> float:
    if not notes:
        return 0.0
    beat_len = steps_per_bar // 4 if steps_per_bar >= 4 else 1
    offbeat = sum(1 for n in notes if (n.start_step % beat_len) != 0)
    fraction = offbeat / len(notes)
    return _triangular(fraction, 0.20, 0.45, slack=0.25)


def evaluate(genome, spec: GenomeSpec, config, progression) -> tuple[float, dict]:
    notes = decode(genome, spec)
    components = {
        "chord_tone": score_chord_tone_alignment(notes, progression, config.steps_per_bar),
        "smoothness": score_melodic_smoothness(notes),
        "density": score_density(notes, len(genome)),
        "repetition": score_repetition(genome, config.steps_per_bar, config.num_bars),
        "cadence": score_cadence(notes, progression[-1]),
        "syncopation": score_syncopation(notes, config.steps_per_bar),
    }
    weights = {**DEFAULT_WEIGHTS, **config.fitness_weights}
    wsum = sum(weights.values())
    total = sum(components[k] * weights[k] for k in components) / wsum
    return total, components
