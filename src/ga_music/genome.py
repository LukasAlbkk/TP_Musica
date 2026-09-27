"""Genotype representation for the evolved lead riff/melody.

A genome is a fixed-length integer array on an eighth- (or sixteenth-) note
grid, `steps_per_bar * num_bars` genes long. Each gene is one of:

    0            -> REST
    1            -> HOLD  (sustain the previously started note by one step)
    2 .. P+1     -> start a new note at pitch `alphabet[gene - 2]`

`alphabet` is the ordered list of MIDI pitches available to the lead voice
(built from a pentatonic scale sharing the song's tonic, see theory.py). This
grid + REST/HOLD/pitch-index scheme keeps the genome a plain fixed-length
integer vector, so classic single-point crossover and per-gene mutation apply
directly without any repair step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import theory

REST = 0
HOLD = 1
PITCH_OFFSET = 2  # gene value `PITCH_OFFSET + i` selects alphabet[i]


@dataclass(frozen=True)
class GenomeSpec:
    length: int
    alphabet: list[int]  # MIDI pitches available to the lead voice

    @property
    def num_symbols(self) -> int:
        return PITCH_OFFSET + len(self.alphabet)

    def pitch_for_gene(self, gene: int) -> int:
        return self.alphabet[gene - PITCH_OFFSET]


@dataclass(frozen=True)
class NoteEvent:
    start_step: int
    duration_steps: int
    pitch: int  # MIDI note number


def build_spec(config) -> GenomeSpec:
    root_pc = theory.note_name_to_pitch_class(config.key_root)
    alphabet = theory.scale_pitches_in_range(
        root_pc, config.lead_scale_mode, config.lead_low_midi, config.lead_high_midi
    )
    return GenomeSpec(length=config.genome_length(), alphabet=alphabet)


def random_genome(spec: GenomeSpec, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, spec.num_symbols, size=spec.length)


def decode(genome: np.ndarray, spec: GenomeSpec) -> list[NoteEvent]:
    """Turn the raw gene array into a list of (start, duration, pitch) notes.
    A HOLD with no preceding open note is treated as a REST (no-op)."""
    notes: list[NoteEvent] = []
    open_start = None
    open_pitch = None

    def close_note(end_step: int) -> None:
        nonlocal open_start, open_pitch
        if open_start is not None:
            notes.append(NoteEvent(open_start, end_step - open_start, open_pitch))
            open_start, open_pitch = None, None

    for step, gene in enumerate(genome):
        gene = int(gene)
        if gene == REST:
            close_note(step)
        elif gene == HOLD:
            if open_start is None:
                continue  # dangling HOLD -> silently treated as rest
            # keep the note open, nothing to do until it's closed
        else:
            close_note(step)
            open_start = step
            open_pitch = spec.pitch_for_gene(gene)
    close_note(len(genome))
    return notes
