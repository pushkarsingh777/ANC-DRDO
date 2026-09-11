# ANC Dataset Pipeline (FR-1)

Implements the **Dataset Pipeline** phase of the Adaptive Noise Cancellation
PRD (Problem Statement 26052): turns clean speech + defence-relevant noise
into noisy/clean training pairs at controlled SNRs, with augmentation and
reproducible speaker-disjoint splits.

## Layout

```
configs/dataset_config.yaml   # all knobs: SNR range, split ratios, aug probs, paths
src/
  audio_utils.py              # load/save/resample/RMS/SNR-mixing
  augmentation.py             # reverb, clipping, speed perturbation, impulsive bursts
  manifest.py                 # directory scanning + seeded speaker-disjoint splits
  mixture_generator.py        # the actual per-utterance mixing pipeline
scripts/
  download_data.py            # fetch/arrange LibriSpeech, VCTK, ESC-50, UrbanSound8K, DEMAND
  download_gunshot_data.py    # automated gunshot/military noise from ESC-50, Zenodo, Kaggle
  generate_synthetic_smoke_data.py  # synthetic placeholder audio for wiring tests
  build_manifests.py          # scan data/ -> manifests with train/val/test splits
  generate_dataset.py         # manifests -> actual mixture wavs + mixture manifest
tests/
  test_pipeline_sanity.py     # verifies realized SNR matches target, no NaNs/corruption
```

## Quickstart (with synthetic placeholder data)

Real corpora aren't bundled here (see `scripts/download_data.py` for sources
and licenses). To validate the pipeline right away:

```bash
pip install -r requirements.txt

python scripts/generate_synthetic_smoke_data.py --n_speakers 12 --utts_per_speaker 6
python scripts/build_manifests.py
python scripts/generate_dataset.py
python tests/test_pipeline_sanity.py --manifest data/manifests/mixtures_train.csv
```

This produces `data/mixtures/{train,val,test}/{mixture,clean,noise}/*.wav`
plus CSV manifests in `data/manifests/` — the exact input format the
FR-2 (Model Training) stage will consume.

## Switching to real data

1. Run `python scripts/download_data.py --list` to see sources for LibriSpeech,
   VCTK, ESC-50, UrbanSound8K, and DEMAND, with per-dataset notes.
2. Arrange audio into:
   - `data/clean_speech/<speaker_id>/*.wav`
   - `data/noise/{impulsive,rotor,vehicle,ambient,sirens}/*.wav`
3. Re-run `build_manifests.py` then `generate_dataset.py` — no code changes needed.
4. **Real gunfire/artillery/rotor recordings under operational conditions are
   not available as open datasets** (PRD section 8.2 lists these as
   "custom"). That category has to come from DRDO's own field recordings or
   a cleared library; drop those files straight into `data/noise/impulsive/`
   and `data/noise/rotor/` and the pipeline picks them up automatically.

## Downloading gunshot / military noise (automated)

Instead of manually finding and downloading impulsive noise clips, run:

```bash
# Download from ALL three sources (ESC-50 + Zenodo + Kaggle):
python scripts/download_gunshot_data.py

# Or pick a single source:
python scripts/download_gunshot_data.py --source esc50
python scripts/download_gunshot_data.py --source zenodo
python scripts/download_gunshot_data.py --source kaggle

# List available sources:
python scripts/download_gunshot_data.py --list
```

This automatically downloads, extracts, and converts all clips to 16 kHz
mono PCM-16 WAV directly into `data/noise/impulsive/`. Sources:

| Source | Clips | Content | License |
|---|---|---|---|
| **ESC-50** | ~160 | Gunshot, fireworks, explosion, glass-breaking | CC-BY-NC 3.0 |
| **Zenodo IoBT** | Varies | Multi-firearm, multi-orientation firing range | CC-BY 4.0 |
| **Kaggle** | 851 | 8 gun models (AK-47, M16, M249, MP5, …) | Academic |

## Design notes / how each PRD requirement is covered

| PRD ref | How it's handled |
|---|---|
| FR-1.1/1.2 | `scan_speech_dir` / `scan_noise_dir` walk arbitrary folder layouts (LibriSpeech/VCTK-style `speaker/**/*.wav` or flat) |
| FR-1.3 | `mix_at_snr` mixes at a uniformly sampled SNR in `[snr_db_min, snr_db_max]` (default −10…+15 dB) |
| FR-1.4 | `augmentation.py`: reverb (synthetic RT60-parameterized IR), clipping, impulsive burst overlay, speed perturbation |
| FR-1.5 | Every stage takes a single `seed` from the config; splits and mixing use `np.random.default_rng`, and the split-name offset uses a SHA-256-based deterministic hash (not Python's salted `hash()`, which is **not** reproducible across runs) |
| FR-1.6 | `split.train/val/test` ratios in the config, with speaker-disjoint splitting (`split_speech_by_speaker`) so no voice leaks across splits |

## Known limitations of this first pass

- Speed perturbation is implemented via a cheap resample trick (changes
  pitch along with tempo) — fine as an augmentation, but if you want
  tempo-only perturbation later, swap in `librosa.effects.time_stretch`.
- Reverb IRs are synthetic (exponentially-decayed noise). Swap in real
  recorded IRs (e.g. OpenAIR) via the `ir=` argument of `apply_reverb`
  without touching the pipeline.
- `impulsive_overlay` currently reloads burst candidates from disk per
  utterance; fine for now, but worth caching in memory once you're running
  this at full corpus scale (hours, not thousands of files).
- This stage only produces the *dataset*. Model training (FR-2), the
  real-time inference engine (FR-3), and edge export (FR-4) are separate
  phases — happy to build those next.
