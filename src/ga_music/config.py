"""Central hyperparameter configuration for one GA run (one generated song)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class GAConfig:
    # --- style / stylistic constraint (80s/90s rock) ---
    key_root: str = "A"                 # tonic note name
    key_mode: str = "natural_minor"     # "natural_minor" or "major"
    progression_name: str = "i-VI-III-VII"
    tempo_bpm: float = 132.0
    num_bars: int = 16
    steps_per_bar: int = 8              # 8 = eighth-note grid, 16 = sixteenth-note grid

    # lead melody pitch alphabet (built from a pentatonic scale over ~1.5 octaves)
    lead_scale_mode: str = "minor_pentatonic"  # "minor_pentatonic" or "major_pentatonic"
    lead_low_midi: int = 57             # A3
    lead_high_midi: int = 81            # A5

    # --- genetic algorithm hyperparameters ---
    population_size: int = 80
    num_generations: int = 200
    crossover_rate: float = 0.85
    mutation_rate: float = 0.06         # per-gene probability
    tournament_size: int = 3
    elitism_count: int = 2
    early_stop_patience: int = 30       # stop if best fitness stalls this many gens
    seed: int = 42

    # optional overrides for fitness term weights (see fitness.py DEFAULT_WEIGHTS)
    fitness_weights: dict = field(default_factory=dict)

    def genome_length(self) -> int:
        return self.num_bars * self.steps_per_bar

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @staticmethod
    def from_json(text: str) -> "GAConfig":
        return GAConfig(**json.loads(text))
