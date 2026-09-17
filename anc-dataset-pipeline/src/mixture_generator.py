"""
Core dataset-generation pipeline (FR-1.3, FR-1.4, FR-1.5).

For every clean speech utterance in a split:
  1. sample a noise category bundle according to configured weights
  2. sample a noise clip from that bundle, loop/crop to speech length
  3. optionally overlay extra impulsive bursts (gunshots/explosions) at random times
  4. sample a target SNR in [snr_db_min, snr_db_max] and mix
  5. optionally apply reverb / clipping / speed perturbation
  6. write mixture.wav, clean.wav, (optional) noise.wav + a manifest row

Everything is driven by a single seeded numpy Generator per split so runs are
reproducible end-to-end (FR-1.5) given the same config.
"""
from __future__ import annotations

import os
import csv
from dataclasses import dataclass, asdict

# pyrefly: ignore [missing-import]
import numpy as np

from .audio_utils import load_audio, save_audio, fit_or_loop_to_length, mix_at_snr
from .augmentation import apply_reverb, apply_clipping, apply_speed_perturbation, overlay_impulsive_bursts


@dataclass
class MixtureRecord:
    mixture_id: str
    speech_path: str
    noise_path: str
    split: str
    snr_db: float
    noise_category: str
    reverb_applied: bool
    clipping_applied: bool
    speed_rate: float
    impulsive_overlay_count: int
    mixture_wav: str
    clean_wav: str
    noise_wav: str


def _pick_category(cfg_weights: dict, rng: np.random.Generator) -> tuple[str, list]:
    """Pick one leaf noise category (e.g. 'ambient') given the bundle weights in config."""
    bundle_names = list(cfg_weights.keys())
    weights = np.array([cfg_weights[b]["weight"] for b in bundle_names], dtype=float)
    weights = weights / weights.sum()
    bundle = bundle_names[rng.choice(len(bundle_names), p=weights)]
    categories = cfg_weights[bundle]["categories"]
    category = categories[rng.integers(0, len(categories))]
    return category, categories


def generate_split(
    speech_entries: list,
    noise_entries: list,
    split: str,
    sr: int,
    cfg: dict,
    out_dir: str,
    rng: np.random.Generator,
) -> list[MixtureRecord]:
    noise_by_category: dict[str, list] = {}
    for n in noise_entries:
        if n["split"] != split and split != "all":
            continue
        noise_by_category.setdefault(n["category"], []).append(n)

    impulsive_pool_paths = [n["path"] for n in noise_by_category.get("impulsive", [])]

    mix_dir = os.path.join(out_dir, split, "mixture")
    clean_dir = os.path.join(out_dir, split, "clean")
    noise_dir = os.path.join(out_dir, split, "noise")
    for d in (mix_dir, clean_dir, noise_dir):
        os.makedirs(d, exist_ok=True)

    mixing_cfg = cfg["mixing"]
    aug_cfg = cfg["augmentation"]
    write_noise_ref = cfg["output"].get("write_noise_only_reference", True)
    subtype = cfg["output"].get("subtype", "PCM_16")

    records: list[MixtureRecord] = []
    split_speech = [s for s in speech_entries if s["split"] == split or split == "all"]
    total_split = len(split_speech)

    for idx, s in enumerate(split_speech, start=1):
        speech = load_audio(s["path"], sr)
        if len(speech) < int(0.5 * sr):
            continue  # skip near-empty clips

        category, _ = _pick_category(mixing_cfg["noise_category_weights"], rng)
        candidates = noise_by_category.get(category, [])
        if not candidates:
            # fall back to any available category for this split
            all_candidates = [n for cat in noise_by_category.values() for n in cat]
            if not all_candidates:
                continue
            candidates = all_candidates

        noise_meta = candidates[rng.integers(0, len(candidates))]
        noise = load_audio(noise_meta["path"], sr)
        noise = fit_or_loop_to_length(noise, len(speech), rng)

        impulsive_count = 0
        if rng.uniform() < mixing_cfg.get("impulsive_overlay_prob", 0.0) and impulsive_pool_paths:
            burst_pool = []
            for _ in range(3):  # load a small candidate pool once per utterance
                p = impulsive_pool_paths[rng.integers(0, len(impulsive_pool_paths))]
                burst_pool.append(load_audio(p, sr))
            lo, hi = mixing_cfg.get("impulsive_overlay_count_range", [1, 1])
            before = noise.copy()
            noise = overlay_impulsive_bursts(noise, burst_pool, sr, rng, count_range=(lo, hi))
            impulsive_count = int(rng.integers(lo, hi + 1))

        snr_db = float(rng.uniform(mixing_cfg["snr_db_min"], mixing_cfg["snr_db_max"]))
        speech, scaled_noise, mixture = mix_at_snr(speech, noise, snr_db)

        speed_rate = 1.0
        if rng.uniform() < aug_cfg["speed_perturbation"]["prob"]:
            lo, hi = aug_cfg["speed_perturbation"]["rate_range"]
            speed_rate = float(rng.uniform(lo, hi))
            speech = apply_speed_perturbation(speech, sr, speed_rate)
            mixture = apply_speed_perturbation(mixture, sr, speed_rate)
            scaled_noise = apply_speed_perturbation(scaled_noise, sr, speed_rate)

        reverb_applied = False
        if rng.uniform() < aug_cfg["reverb"]["prob"]:
            lo, hi = aug_cfg["reverb"]["rt60_range_s"]
            rt60 = float(rng.uniform(lo, hi))
            mixture = apply_reverb(mixture, rt60, sr, rng)
            reverb_applied = True

        clipping_applied = False
        if rng.uniform() < aug_cfg["clipping"]["prob"]:
            lo, hi = aug_cfg["clipping"]["clip_threshold_range"]
            thresh = float(rng.uniform(lo, hi))
            mixture = apply_clipping(mixture, thresh)
            clipping_applied = True

        mixture_id = f"{split}_{idx:06d}"
        mix_path = os.path.join(mix_dir, f"{mixture_id}.wav")
        clean_path = os.path.join(clean_dir, f"{mixture_id}.wav")
        noise_path = os.path.join(noise_dir, f"{mixture_id}.wav")

        save_audio(mix_path, mixture, sr, subtype)
        save_audio(clean_path, speech, sr, subtype)
        if write_noise_ref:
            save_audio(noise_path, scaled_noise, sr, subtype)
        else:
            noise_path = ""

        records.append(MixtureRecord(
            mixture_id=mixture_id,
            speech_path=s["path"],
            noise_path=noise_meta["path"],
            split=split,
            snr_db=round(snr_db, 3),
            noise_category=category,
            reverb_applied=reverb_applied,
            clipping_applied=clipping_applied,
            speed_rate=round(speed_rate, 4),
            impulsive_overlay_count=impulsive_count,
            mixture_wav=mix_path,
            clean_wav=clean_path,
            noise_wav=noise_path,
        ))

        if idx % 250 == 0 or idx == total_split:
            pct = (idx / total_split) * 100
            print(f"  [{split}] {idx:4d}/{total_split} ({pct:5.1f}%) mixtures created...")

    return records


def write_mixture_manifest(records: list[MixtureRecord], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not records:
        open(path, "w").close()
        return
    fieldnames = list(asdict(records[0]).keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))
