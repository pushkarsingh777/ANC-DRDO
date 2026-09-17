"""
Scans clean-speech and noise directories and produces reproducible manifest
files (FR-1.6), plus seeded train/val/test splits (FR-1.5).

Manifests are plain CSVs so they're easy to inspect, diff, and version with DVC.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, asdict

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import soundfile as sf

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3"}

NOISE_CATEGORIES = ["impulsive", "rotor", "vehicle", "ambient", "sirens"]


@dataclass
class SpeechEntry:
    path: str
    duration_s: float
    speaker_id: str
    split: str = ""


@dataclass
class NoiseEntry:
    path: str
    duration_s: float
    category: str
    split: str = ""


def _duration_s(path: str) -> float:
    info = sf.info(path)
    return info.frames / info.samplerate


def _infer_speaker_id(rel_path: str, filename: str) -> str:
    """
    Extracts speaker id from standard speech corpora layouts:
    - LibriSpeech filename / folder: '1272-128104-0000.flac' -> '1272' or dev-clean/1272/...
    - VCTK filename / folder: 'p225_001.wav' -> 'p225'
    - Nested folder: 'speaker_id/...'
    """
    stem = os.path.splitext(filename)[0]
    parts_hyphen = stem.split("-")
    if len(parts_hyphen) >= 2 and parts_hyphen[0].isdigit():
        return parts_hyphen[0]

    parts_under = stem.split("_")
    if len(parts_under) >= 2 and parts_under[0].startswith("p") and parts_under[0][1:].isdigit():
        return parts_under[0]

    parts = rel_path.split(os.sep)
    for i, p in enumerate(parts[:-1]):
        if p in {"LibriSpeech", "dev-clean", "train-clean-100", "train-clean-360", "test-clean", "wav48", "wav48_silence_trimmed"}:
            if i + 1 < len(parts) - 1:
                return parts[i + 1]

    if len(parts) > 1:
        return parts[0]
    return "unknown"


def scan_speech_dir(speech_dir: str) -> list[SpeechEntry]:
    """
    Expects either a flat folder of wavs, or LibriSpeech/VCTK-style
    `speaker_id/**/*.wav` layout. Speaker id is inferred from the filename
    or directory hierarchy.
    """
    entries: list[SpeechEntry] = []
    for root, _, files in os.walk(speech_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() not in AUDIO_EXTS:
                continue
            full = os.path.join(root, f)
            rel = os.path.relpath(full, speech_dir)
            speaker_id = _infer_speaker_id(rel, f)
            try:
                dur = _duration_s(full)
            except Exception:
                continue
            entries.append(SpeechEntry(path=full, duration_s=dur, speaker_id=speaker_id))
    return entries


def scan_noise_dir(noise_root: str) -> list[NoiseEntry]:
    """Expects noise_root/<category>/*.wav where category is one of NOISE_CATEGORIES."""
    entries: list[NoiseEntry] = []
    for category in NOISE_CATEGORIES:
        cat_dir = os.path.join(noise_root, category)
        if not os.path.isdir(cat_dir):
            continue
        for root, _, files in os.walk(cat_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() not in AUDIO_EXTS:
                    continue
                full = os.path.join(root, f)
                try:
                    dur = _duration_s(full)
                except Exception:
                    continue
                entries.append(NoiseEntry(path=full, duration_s=dur, category=category))
    return entries


def assign_splits(n: int, ratios: dict, rng) -> list[str]:
    """
    Return a list of split labels of length n, shuffled, matching `ratios`
    proportions as closely as integers allow (largest-remainder method, so
    small-N sets like a handful of speakers don't lose an entire split to
    naive rounding).
    """
    names = list(ratios.keys())
    fracs = np.array([ratios[k] for k in names], dtype=float)
    fracs = fracs / fracs.sum()
    raw = fracs * n
    base = np.floor(raw).astype(int)
    remainder = n - base.sum()
    # give the leftover slots to the entries with the largest fractional remainder
    order = np.argsort(-(raw - base))
    for i in range(remainder):
        base[order[i % len(order)]] += 1
    labels = []
    for name, count in zip(names, base):
        labels += [name] * int(count)
    rng.shuffle(labels)
    return labels


def split_speech_by_speaker(entries: list[SpeechEntry], ratios: dict, rng) -> list[SpeechEntry]:
    """
    Split at the SPEAKER level (not utterance level) so the same voice never
    leaks across train/val/test — important for generalization claims.
    """
    speakers = sorted({e.speaker_id for e in entries})
    rng.shuffle(speakers)
    speaker_split = dict(zip(speakers, assign_splits(len(speakers), ratios, rng)))
    for e in entries:
        e.split = speaker_split[e.speaker_id]
    return entries


def split_noise(entries: list[NoiseEntry], ratios: dict, rng) -> list[NoiseEntry]:
    splits = assign_splits(len(entries), ratios, rng)
    for e, s in zip(entries, splits):
        e.split = s
    return entries


def write_csv(entries: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not entries:
        with open(path, "w", newline="") as f:
            f.write("")
        return
    fieldnames = list(asdict(entries[0]).keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in entries:
            writer.writerow(asdict(e))


def read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))
