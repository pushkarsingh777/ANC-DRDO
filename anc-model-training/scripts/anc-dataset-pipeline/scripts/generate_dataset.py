#!/usr/bin/env python3
"""
Generate noisy/clean training mixtures for every split, using the manifests
built by build_manifests.py (FR-1.3, FR-1.4).

Usage:
    python scripts/generate_dataset.py --config configs/dataset_config.yaml
"""
import argparse
import hashlib
import os
import sys

# pyrefly: ignore [missing-import]
import numpy as np
import yaml


def deterministic_offset(name: str, modulus: int = 10_000) -> int:
    """
    Stable string -> int offset across processes/runs.
    (Python's built-in hash() is salted per-process for security reasons and
    must NEVER be used where run-to-run reproducibility is required.)
    """
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return int(digest, 16) % modulus

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.manifest import read_csv
from src.mixture_generator import generate_split, write_mixture_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset_config.yaml")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    speech_manifest_path = os.path.join(cfg["paths"]["manifests_dir"], "speech_manifest.csv")
    noise_manifest_path = os.path.join(cfg["paths"]["manifests_dir"], "noise_manifest.csv")

    if not os.path.exists(speech_manifest_path) or not os.path.exists(noise_manifest_path):
        print("Manifests not found. Run scripts/build_manifests.py first.")
        sys.exit(1)

    speech_entries = read_csv(speech_manifest_path)
    noise_entries = read_csv(noise_manifest_path)

    if not speech_entries or not noise_entries:
        print("One or more manifests are empty — nothing to generate.")
        sys.exit(1)

    sr = cfg["sample_rate"]
    out_dir = cfg["paths"]["mixtures_dir"]
    base_seed = cfg["seed"]

    all_records = []
    for split in args.splits:
        # separate RNG stream per split so train/val/test generation is
        # independently reproducible even if you regenerate just one split
        split_seed = base_seed + deterministic_offset(split)
        rng = np.random.default_rng(split_seed)

        print(f"\nGenerating split '{split}'...")
        records = generate_split(speech_entries, noise_entries, split, sr, cfg, out_dir, rng)
        print(f"  wrote {len(records)} mixtures to {out_dir}/{split}/")

        manifest_path = os.path.join(cfg["paths"]["manifests_dir"], f"mixtures_{split}.csv")
        write_mixture_manifest(records, manifest_path)
        print(f"  manifest -> {manifest_path}")
        all_records.extend(records)

    print(f"\nDone. {len(all_records)} total mixtures generated.")


if __name__ == "__main__":
    main()
