"""Turns an evolved lead-riff genome into a full 4-track rock arrangement:
lead guitar (evolved), rhythm power chords, bass and drums (all three fixed
by rule from the chord progression, to keep the style constraint explicit
and independent of what the GA is optimizing)."""

from __future__ import annotations

import pretty_midi as pm

from . import theory
from .genome import GenomeSpec, decode


def _seconds_per_step(config) -> float:
    return 240.0 / (config.tempo_bpm * config.steps_per_bar)


def _pitch_near(pc: int, target_midi: int) -> int:
    best = None
    for octave in range(0, 11):
        candidate = pc + 12 * octave
        if best is None or abs(candidate - target_midi) < abs(best - target_midi):
            best = candidate
    return best


def _add_lead(pmid: pm.PrettyMIDI, genome, spec: GenomeSpec, config) -> None:
    sps = _seconds_per_step(config)
    inst = pm.Instrument(program=theory.GM_LEAD_GUITAR, name="Lead Riff (evolved)")
    for note in decode(genome, spec):
        inst.notes.append(pm.Note(
            velocity=100,
            pitch=note.pitch,
            start=note.start_step * sps,
            end=(note.start_step + note.duration_steps) * sps,
        ))
    pmid.instruments.append(inst)


def _add_rhythm_guitar(pmid: pm.PrettyMIDI, progression, config) -> None:
    sps = _seconds_per_step(config)
    steps_per_beat = max(1, config.steps_per_bar // 4)
    inst = pm.Instrument(program=theory.GM_RHYTHM_GUITAR, name="Rhythm Power Chords")
    for bar, chord in enumerate(progression):
        root = _pitch_near(chord.root_pc, 43)   # ~ G2, palm-mute register
        fifth = _pitch_near(chord.fifth_pc, root + 7)
        for beat in range(4):
            step = bar * config.steps_per_bar + beat * steps_per_beat
            start = step * sps
            end = start + steps_per_beat * sps * 0.9
            velocity = 108 if beat in (0, 2) else 92
            for pitch in (root, fifth, root + 12):
                inst.notes.append(pm.Note(velocity=velocity, pitch=pitch, start=start, end=end))
    pmid.instruments.append(inst)


def _add_bass(pmid: pm.PrettyMIDI, progression, config) -> None:
    sps = _seconds_per_step(config)
    steps_per_beat = max(1, config.steps_per_bar // 4)
    inst = pm.Instrument(program=theory.GM_BASS, name="Bass")
    for bar, chord in enumerate(progression):
        root = _pitch_near(chord.root_pc, 33)  # ~ A1/G1 register
        for beat in range(4):
            step = bar * config.steps_per_bar + beat * steps_per_beat
            start = step * sps
            end = start + steps_per_beat * sps * 0.95
            velocity = 110 if beat in (0, 2) else 95
            inst.notes.append(pm.Note(velocity=velocity, pitch=root, start=start, end=end))
    pmid.instruments.append(inst)


def _add_drums(pmid: pm.PrettyMIDI, config) -> None:
    sps = _seconds_per_step(config)
    steps_per_beat = max(1, config.steps_per_bar // 4)
    eighth_step = max(1, steps_per_beat // 2)
    inst = pm.Instrument(program=0, is_drum=True, name="Drums")
    total_steps = config.num_bars * config.steps_per_bar

    def hit(step, pitch, velocity, dur_steps=1):
        start = step * sps
        end = start + max(dur_steps, 1) * sps
        inst.notes.append(pm.Note(velocity=velocity, pitch=pitch, start=start, end=end))

    step = 0
    while step < total_steps:
        bar = step // config.steps_per_bar
        pos_in_bar = step % config.steps_per_bar
        beat = pos_in_bar // steps_per_beat if steps_per_beat else 0

        if pos_in_bar == 0 and bar % 4 == 0:
            hit(step, theory.DRUM_CRASH, 100, dur_steps=steps_per_beat)
        if pos_in_bar % steps_per_beat == 0 and beat in (0, 2):
            hit(step, theory.DRUM_KICK, 105)
        if pos_in_bar % steps_per_beat == 0 and beat in (1, 3):
            hit(step, theory.DRUM_SNARE, 100)
        if pos_in_bar % eighth_step == 0:
            hit(step, theory.DRUM_HIHAT_CLOSED, 75)
        step += eighth_step

    pmid.instruments.append(inst)


def build_song(genome, spec: GenomeSpec, config) -> pm.PrettyMIDI:
    pmid = pm.PrettyMIDI(initial_tempo=config.tempo_bpm)
    root_pc = theory.note_name_to_pitch_class(config.key_root)
    progression = theory.build_progression(root_pc, config.key_mode, config.progression_name, config.num_bars)
    _add_lead(pmid, genome, spec, config)
    _add_rhythm_guitar(pmid, progression, config)
    _add_bass(pmid, progression, config)
    _add_drums(pmid, config)
    return pmid
