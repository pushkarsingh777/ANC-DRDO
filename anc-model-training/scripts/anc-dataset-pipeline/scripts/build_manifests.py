#!/usr/bin/env python3
"""
Scan clean_speech_dir and noise_root_dir (see configs/dataset_config.yaml),
assign seeded train/val/test splits, and write manifests to manifests_dir.

Usage:
    python scripts/build_manifests.py --config configs/dataset_config.yaml
"""
import argparse
import os
import sys

# pyrefly: ignore [missing-import]
import numpy as np
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.manifest import (
    scan_speech_dir, scan_noise_dir,
    split_speech_by_speaker, split_noise,
    write_csv,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset_config.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rng = np.random.default_rng(cfg["seed"])
    ratios = cfg["split"]

    print(f"Scanning speech dir: {cfg['paths']['clean_speech_dir']}")
    speech_entries = scan_speech_dir(cfg["paths"]["clean_speech_dir"])
    print(f"  found {len(speech_entries)} speech files, "
          f"{len({e.speaker_id for e in speech_entries})} speakers")
    speech_entries = split_speech_by_speaker(speech_entries, ratios, rng)

    print(f"Scanning noise dir: {cfg['paths']['noise_root_dir']}")
    noise_entries = scan_noise_dir(cfg["paths"]["noise_root_dir"])
    print(f"  found {len(noise_entries)} noise files")
    noise_entries = split_noise(noise_entries, ratios, rng)

    os.makedirs(cfg["paths"]["manifests_dir"], exist_ok=True)
    speech_manifest_path = os.path.join(cfg["paths"]["manifests_dir"], "speech_manifest.csv")
    noise_manifest_path = os.path.join(cfg["paths"]["manifests_dir"], "noise_manifest.csv")

    write_csv(speech_entries, speech_manifest_path)
    write_csv(noise_entries, noise_manifest_path)

    print(f"Wrote {speech_manifest_path} ({len(speech_entries)} rows)")
    print(f"Wrote {noise_manifest_path} ({len(noise_entries)} rows)")

    if len(speech_entries) == 0 or len(noise_entries) == 0:
        print("\nWARNING: one or more manifests are empty. "
              "Populate data/clean_speech and data/noise/<category>/ "
              "(see scripts/download_data.py) before running generate_dataset.py.")


if __name__ == "__main__":
    main()
