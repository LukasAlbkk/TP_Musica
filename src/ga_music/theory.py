"""Music-theory building blocks used to pin the output to an 80s/90s rock style.

Only the lead riff/melody is evolved by the genetic algorithm (see genome.py /
ga.py). The harmonic backbone defined here (scale, chord progressions, GM
instrument choices, drum pattern) is fixed by rule so that every individual in
every generation is already "rock-shaped": diatonic chords built from power
chords, a straight backbeat, and a walking-root bass line.
"""

from __future__ import annotations

from dataclasses import dataclass

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

SCALE_INTERVALS = {
    # 7-note diatonic scales (semitone offsets from the root)
    "natural_minor": [0, 2, 3, 5, 7, 8, 10],
    "major": [0, 2, 4, 5, 7, 9, 11],
    # 5-note scale used only for building the lead-melody pitch alphabet
    "minor_pentatonic": [0, 3, 5, 7, 10],
    "major_pentatonic": [0, 2, 4, 7, 9],
}

# Roman-numeral rock progressions expressed as 0-indexed diatonic scale
# degrees (0 = i/I, 1 = ii, 2 = iii/III, ... 6 = vii/VII). One chord per bar,
# cycling if the progression is shorter than num_bars.
MINOR_PROGRESSIONS = {
    "i-VI-III-VII": [0, 5, 2, 6],   # e.g. Am-F-C-G, ubiquitous 80s/90s rock & pop-punk
    "i-iv-v": [0, 3, 4],            # Am-Dm-Em, minor blues-rock
    "i-VII-VI-VII": [0, 6, 5, 6],   # Am-G-F-G, anthemic hard rock
}

MAJOR_PROGRESSIONS = {
    "I-V-vi-IV": [0, 4, 5, 3],      # the "four chords" pop-rock progression
    "I-IV-V": [0, 3, 4],            # classic rock / blues-rock
    "vi-IV-I-V": [5, 3, 0, 4],      # common 90s alt-rock ordering
}

PROGRESSIONS_BY_MODE = {
    "natural_minor": MINOR_PROGRESSIONS,
    "major": MAJOR_PROGRESSIONS,
}

# General MIDI program numbers (0-indexed, as used by pretty_midi)
GM_LEAD_GUITAR = 29     # Overdriven Guitar
GM_RHYTHM_GUITAR = 30   # Distortion Guitar
GM_BASS = 34            # Electric Bass (pick)

# General MIDI percussion key map (channel 10)
DRUM_KICK = 36
DRUM_SNARE = 38
DRUM_HIHAT_CLOSED = 42
DRUM_HIHAT_OPEN = 46
DRUM_CRASH = 49


def note_name_to_pitch_class(name: str) -> int:
    name = name.strip().upper().replace("♭", "b")
    if name not in NOTE_NAMES:
        # accept a couple of common flat spellings
        flats = {"DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#"}
        name = flats.get(name, name)
    return NOTE_NAMES.index(name)


def scale_pitch_classes(root_pc: int, mode: str) -> list[int]:
    """The 7 (or 5) pitch classes of a scale, in ascending order from the root."""
    return [(root_pc + iv) % 12 for iv in SCALE_INTERVALS[mode]]


def scale_pitches_in_range(root_pc: int, mode: str, low_midi: int, high_midi: int) -> list[int]:
    """All absolute MIDI pitches of a scale within [low_midi, high_midi]."""
    pcs = set(scale_pitch_classes(root_pc, mode))
    return [p for p in range(low_midi, high_midi + 1) if p % 12 in pcs]


@dataclass(frozen=True)
class Chord:
    degree: int
    root_pc: int
    third_pc: int
    fifth_pc: int

    def contains_pc(self, pc: int) -> bool:
        return pc in (self.root_pc, self.third_pc, self.fifth_pc)


def diatonic_triad(root_pc: int, mode: str, degree: int) -> Chord:
    """Build the diatonic triad on `degree` (0-indexed) by stacking thirds
    within the parent scale (works uniformly for major and natural minor)."""
    degrees = scale_pitch_classes(root_pc, mode)
    n = len(degrees)
    root = degrees[degree % n]
    third = degrees[(degree + 2) % n]
    fifth = degrees[(degree + 4) % n]
    return Chord(degree=degree, root_pc=root, third_pc=third, fifth_pc=fifth)


def build_progression(root_pc: int, mode: str, progression_name: str, num_bars: int) -> list[Chord]:
    """One Chord per bar, cycling the named progression to fill num_bars."""
    degrees = PROGRESSIONS_BY_MODE[mode][progression_name]
    return [diatonic_triad(root_pc, mode, degrees[i % len(degrees)]) for i in range(num_bars)]
