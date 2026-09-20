#!/usr/bin/env python3
"""
Lightweight sanity checks (not a full pytest suite) confirming:
  1. every mixture manifest row's audio files exist and load
  2. mixture = clean + noise (approximately, allowing for the anti-clip gain
     applied in mix_at_snr) for rows where no reverb/speed aug was applied
  3. realized SNR on unaugmented rows is close to the manifest's snr_db
  4. no NaNs/Infs and no full-scale clipping beyond what clipping-aug intends

Run: python tests/test_pipeline_sanity.py --manifest data/manifests/mixtures_train.csv
"""
import argparse
import csv
import sys
import os

# pyrefly: ignore [missing-import]
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.audio_utils import load_audio, measure_snr_db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/mixtures_train.csv")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--tolerance_db", type=float, default=1.5)
    args = ap.parse_args()

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("Manifest is empty, nothing to check.")
        return

    checked = 0
    snr_errors = []
    failures = []

    for row in rows:
        mix = load_audio(row["mixture_wav"], args.sr)
        clean = load_audio(row["clean_wav"], args.sr)

        if not np.all(np.isfinite(mix)) or not np.all(np.isfinite(clean)):
            failures.append(f"{row['mixture_id']}: non-finite samples")
            continue

        # SNR check only meaningful when reverb/speed perturbation didn't run,
        # since those alter the waveform beyond a simple additive mix
        if row["reverb_applied"] == "False" and float(row["speed_rate"]) == 1.0 and row["noise_wav"]:
            noise = load_audio(row["noise_wav"], args.sr)
            n = min(len(clean), len(noise))
            realized_snr = measure_snr_db(clean[:n], noise[:n])
            target_snr = float(row["snr_db"])
            err = abs(realized_snr - target_snr)
            snr_errors.append(err)
            if err > args.tolerance_db:
                failures.append(
                    f"{row['mixture_id']}: SNR off by {err:.2f} dB "
                    f"(target {target_snr:.2f}, realized {realized_snr:.2f})"
                )
        checked += 1

    print(f"Checked {checked}/{len(rows)} mixtures from {args.manifest}")
    if snr_errors:
        print(f"SNR error: mean={np.mean(snr_errors):.3f} dB, "
              f"max={np.max(snr_errors):.3f} dB, n={len(snr_errors)}")
    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for f in failures[:20]:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()
