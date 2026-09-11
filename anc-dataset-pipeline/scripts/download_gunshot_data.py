#!/usr/bin/env python3
"""
Automated gunshot / military noise downloader.

Downloads, extracts, sorts, and converts impulsive noise clips from three
open-access sources — no manual steps required:

  1. ESC-50  — gunshot + fireworks + explosion clips extracted from the full
               dataset via its CSV metadata.
  2. Zenodo  — Edge-Collected Gunshot/Gunfire Audio Dataset (IoBT, CC-BY 4.0)
               Multi-firearm, multi-orientation firing range recordings.
  3. Kaggle  — Gunshot Audio Dataset by Emrah Aydemir
               851 clips across 8 gun models (AK-47, M16, M249, MP5, etc.)

All output is written to  data/noise/impulsive/  as 16 kHz mono PCM-16 WAVs,
ready for the downstream pipeline (build_manifests.py → generate_dataset.py).

Usage
-----
    # Download from ALL sources:
    python scripts/download_gunshot_data.py

    # Download from a single source:
    python scripts/download_gunshot_data.py --source esc50
    python scripts/download_gunshot_data.py --source zenodo
    python scripts/download_gunshot_data.py --source kaggle

    # List available sources:
    python scripts/download_gunshot_data.py --list

Prerequisites
-------------
- Internet access (first run only — cached archives are reused).
- For Kaggle: either
    (a) the ``kaggle`` pip package + ``~/.kaggle/kaggle.json`` credential file, OR
    (b) the ``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` env-vars set.
  The script prints clear instructions if credentials are missing.
- No other external dependencies beyond numpy + scipy + soundfile (already in
  requirements.txt).

Notes
-----
- All downloads are cached under  data/_downloads/  so re-runs are instant.
- Original sample rates (44.1 kHz, 48 kHz, etc.) are automatically resampled
  to the project's target 16 kHz.
- File naming:  <source>_<category>_<index>.wav  to avoid collisions.
- Licenses: ESC-50 (CC-BY-NC 3.0), Zenodo IoBT (CC-BY 4.0),
  Kaggle Gunshot (academic/research).  Check each before redistribution.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from math import gcd

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import soundfile as sf
# pyrefly: ignore [missing-import]
from scipy.signal import resample_poly


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
TARGET_SR = 16_000
OUTPUT_SUBTYPE = "PCM_16"

# Where downloads are cached and final output lives (relative to CWD,
# which should be the repo root  anc-dataset-pipeline/).
DOWNLOAD_DIR = os.path.join("data", "_downloads")
OUTPUT_DIR = os.path.join("data", "noise", "impulsive")

# ESC-50 categories that map to "impulsive" noise for ANC training
ESC50_IMPULSIVE_CATEGORIES = {
    "gunshot",
    "fireworks",
    "explosion",           # not a default ESC-50 tag but included if present
    "glass_breaking",      # sharp transient, useful augmentation
}

ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"

# Zenodo IoBT Edge-Collected Gunshot Audio Dataset
ZENODO_URL = (
    "https://zenodo.org/records/7004819/files/"
    "edge-collected-gunshot-audio.zip?download=1"
)
ZENODO_FILENAME = "edge-collected-gunshot-audio.zip"

# Kaggle
KAGGLE_DATASET_SLUG = "emrahaydemir/gunshot-audio-dataset"
KAGGLE_FILENAME = "gunshot-audio-dataset.zip"


# ──────────────────────────────────────────────────────────────────────────────
# Audio helpers  (intentionally self-contained, no import from src/)
# ──────────────────────────────────────────────────────────────────────────────
def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    g = gcd(orig_sr, target_sr)
    up, down = target_sr // g, orig_sr // g
    return resample_poly(audio, up, down).astype(np.float32)


def convert_to_target_wav(src_path: str, dst_path: str) -> bool:
    """Load any audio file, convert to 16 kHz mono PCM-16 WAV.
    Returns True on success, False on skip/error."""
    try:
        audio, sr = sf.read(src_path, always_2d=False, dtype="float32")
    except Exception as exc:
        print(f"  [WARN] Cannot read {src_path}: {exc}")
        return False

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != TARGET_SR:
        audio = _resample(audio, sr, TARGET_SR)

    # Normalize peak to -1 dBFS to avoid clipping
    peak = np.max(np.abs(audio)) + 1e-10
    if peak > 0.99:
        audio = audio * (0.99 / peak)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    sf.write(dst_path, audio, TARGET_SR, subtype=OUTPUT_SUBTYPE)
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Download helpers
# ──────────────────────────────────────────────────────────────────────────────
def _download_file(url: str, dest: str, desc: str = "") -> str:
    """Download url → dest (skip if already exists). Returns dest path."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.isfile(dest):
        print(f"  [CACHED] {desc or os.path.basename(dest)}")
        return dest
    print(f"  Downloading {desc or url}  →  {dest}")
    try:
        urllib.request.urlretrieve(url, dest, _progress_hook)
        print()  # newline after progress
    except Exception as exc:
        print(f"\n  [ERROR] Download failed: {exc}")
        if os.path.isfile(dest):
            os.remove(dest)
        raise
    return dest


def _progress_hook(block_num: int, block_size: int, total_size: int):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        mb_down = downloaded / 1e6
        mb_total = total_size / 1e6
        print(f"\r  [{bar}] {pct:5.1f}%  ({mb_down:.1f}/{mb_total:.1f} MB)", end="", flush=True)
    else:
        mb_down = downloaded / 1e6
        print(f"\r  Downloaded {mb_down:.1f} MB", end="", flush=True)


def _extract_zip(archive: str, dest_dir: str):
    print(f"  Extracting {os.path.basename(archive)} → {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest_dir)


def _find_audio_files(root: str, extensions: tuple = (".wav", ".flac", ".ogg", ".mp3")) -> list[str]:
    """Recursively find all audio files under root."""
    found = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(extensions):
                found.append(os.path.join(dirpath, fn))
    return sorted(found)


# ──────────────────────────────────────────────────────────────────────────────
# Source 1: ESC-50
# ──────────────────────────────────────────────────────────────────────────────
def download_esc50():
    """Download ESC-50, parse its CSV metadata, extract only impulsive
    categories (gunshot, fireworks, explosion, glass_breaking), convert to
    16 kHz mono WAV."""
    print("\n━━━ ESC-50 (impulsive categories) ━━━")

    archive = os.path.join(DOWNLOAD_DIR, "ESC-50-master.zip")
    extract_dir = os.path.join(DOWNLOAD_DIR, "_esc50_raw")

    _download_file(ESC50_URL, archive, desc="ESC-50 dataset")

    # Only extract if not already done
    marker = os.path.join(extract_dir, ".extracted")
    if not os.path.isfile(marker):
        _extract_zip(archive, extract_dir)
        Path(marker).touch()

    # Locate the CSV and audio directory inside the extracted archive
    # ESC-50 extracts to ESC-50-master/
    esc_root = None
    for candidate in [
        os.path.join(extract_dir, "ESC-50-master"),
        extract_dir,
    ]:
        csv_check = os.path.join(candidate, "meta", "esc50.csv")
        if os.path.isfile(csv_check):
            esc_root = candidate
            break

    if esc_root is None:
        print("  [ERROR] Could not find ESC-50 meta/esc50.csv in extracted archive.")
        return 0

    csv_path = os.path.join(esc_root, "meta", "esc50.csv")
    audio_dir = os.path.join(esc_root, "audio")

    # Parse CSV to find filenames for impulsive categories
    count = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row.get("category", "").strip().lower()
            filename = row.get("filename", "").strip()
            if category in ESC50_IMPULSIVE_CATEGORIES and filename:
                src = os.path.join(audio_dir, filename)
                if not os.path.isfile(src):
                    continue
                safe_name = filename.replace(" ", "_")
                dst = os.path.join(
                    OUTPUT_DIR,
                    f"esc50_{category}_{safe_name}",
                )
                if not dst.lower().endswith(".wav"):
                    dst = os.path.splitext(dst)[0] + ".wav"

                if os.path.isfile(dst):
                    count += 1
                    continue  # already converted

                if convert_to_target_wav(src, dst):
                    count += 1

    print(f"  ✔ ESC-50: {count} impulsive clips → {OUTPUT_DIR}/")
    return count


# ──────────────────────────────────────────────────────────────────────────────
# Source 2: Zenodo — Edge-Collected Gunshot Audio (IoBT)
# ──────────────────────────────────────────────────────────────────────────────
def download_zenodo():
    """Download the IoBT gunshot/gunfire dataset from Zenodo, extract, and
    convert all audio to 16 kHz mono WAV."""
    print("\n━━━ Zenodo — IoBT Edge-Collected Gunshot Audio (CC-BY 4.0) ━━━")

    archive = os.path.join(DOWNLOAD_DIR, ZENODO_FILENAME)
    extract_dir = os.path.join(DOWNLOAD_DIR, "_zenodo_gunshot_raw")

    _download_file(ZENODO_URL, archive, desc="Zenodo IoBT gunshot dataset")

    marker = os.path.join(extract_dir, ".extracted")
    if not os.path.isfile(marker):
        _extract_zip(archive, extract_dir)
        Path(marker).touch()

    audio_files = _find_audio_files(extract_dir)
    if not audio_files:
        print("  [WARN] No audio files found in Zenodo archive.")
        return 0

    count = 0
    for i, src in enumerate(audio_files):
        # Preserve some of the original path info in the filename
        rel = os.path.relpath(src, extract_dir)
        # Flatten path separators into underscores for a clean filename
        flat = rel.replace(os.sep, "_").replace("/", "_").replace(" ", "_")
        stem = os.path.splitext(flat)[0]
        dst = os.path.join(OUTPUT_DIR, f"zenodo_{stem}.wav")

        if os.path.isfile(dst):
            count += 1
            continue

        if convert_to_target_wav(src, dst):
            count += 1

    print(f"  ✔ Zenodo IoBT: {count} clips → {OUTPUT_DIR}/")
    return count


# ──────────────────────────────────────────────────────────────────────────────
# Source 3: Kaggle — Gunshot Audio Dataset (Emrah Aydemir)
# ──────────────────────────────────────────────────────────────────────────────
def _check_kaggle_credentials() -> bool:
    """Return True if Kaggle credentials are available."""
    # Method 1: env vars
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    # Method 2: credentials file
    kaggle_json = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    if os.path.isfile(kaggle_json):
        return True
    return False


def download_kaggle():
    """Download the Kaggle Gunshot Audio Dataset using the kaggle CLI.
    Requires kaggle pip package + credentials."""
    print("\n━━━ Kaggle — Gunshot Audio Dataset (851 clips, 8 gun models) ━━━")

    extract_dir = os.path.join(DOWNLOAD_DIR, "_kaggle_gunshot_raw")
    archive = os.path.join(DOWNLOAD_DIR, KAGGLE_FILENAME)

    # Check if already extracted
    marker = os.path.join(extract_dir, ".extracted")
    if os.path.isfile(marker):
        print("  [CACHED] Already downloaded and extracted.")
    else:
        # Check kaggle is installed
        try:
            # pyrefly: ignore [missing-import]
            import kaggle  # noqa: F401
        except ImportError:
            print("  [INFO] Installing kaggle package...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "kaggle", "--quiet"],
                stdout=subprocess.DEVNULL,
            )

        if not _check_kaggle_credentials():
            print(
                "  [ERROR] Kaggle credentials not found.\n"
                "  To set up Kaggle API access:\n"
                "    1. Go to https://www.kaggle.com/settings → API → 'Create New Token'\n"
                "    2. Save the downloaded kaggle.json to:\n"
                f"       {os.path.join(os.path.expanduser('~'), '.kaggle', 'kaggle.json')}\n"
                "  OR set env vars: KAGGLE_USERNAME and KAGGLE_KEY\n"
            )
            return 0

        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        print(f"  Downloading {KAGGLE_DATASET_SLUG} via Kaggle API...")
        try:
            subprocess.check_call(
                [
                    sys.executable, "-m", "kaggle", "datasets", "download",
                    "-d", KAGGLE_DATASET_SLUG,
                    "-p", DOWNLOAD_DIR,
                    "--unzip",
                ],
                # Forward output so user sees progress
            )
        except subprocess.CalledProcessError as exc:
            print(f"  [ERROR] Kaggle download failed: {exc}")
            return 0

        # kaggle --unzip extracts directly to DOWNLOAD_DIR; move contents
        # to our dedicated subfolder for cleanliness
        os.makedirs(extract_dir, exist_ok=True)

        # The dataset may extract with a top-level folder or flat files
        # Move everything that looks like audio into extract_dir
        for item in os.listdir(DOWNLOAD_DIR):
            item_path = os.path.join(DOWNLOAD_DIR, item)
            if item.startswith("_") or item.startswith("."):
                continue
            if os.path.isdir(item_path) and "kaggle" not in item.lower() and "esc50" not in item.lower() and "zenodo" not in item.lower():
                # It's a directory that might be the extracted dataset
                dest = os.path.join(extract_dir, item)
                if not os.path.exists(dest):
                    shutil.move(item_path, dest)
            elif os.path.isfile(item_path) and item.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                dest = os.path.join(extract_dir, item)
                if not os.path.exists(dest):
                    shutil.move(item_path, dest)

        Path(marker).touch()

    audio_files = _find_audio_files(extract_dir)
    if not audio_files:
        print("  [WARN] No audio files found in Kaggle download.")
        return 0

    count = 0
    for src in audio_files:
        rel = os.path.relpath(src, extract_dir)
        flat = rel.replace(os.sep, "_").replace("/", "_").replace(" ", "_")
        stem = os.path.splitext(flat)[0]
        dst = os.path.join(OUTPUT_DIR, f"kaggle_{stem}.wav")

        if os.path.isfile(dst):
            count += 1
            continue

        if convert_to_target_wav(src, dst):
            count += 1

    print(f"  ✔ Kaggle: {count} clips → {OUTPUT_DIR}/")
    return count


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
SOURCES = {
    "esc50": {
        "fn": download_esc50,
        "desc": "ESC-50 — gunshot, fireworks, explosion, glass_breaking clips (CC-BY-NC 3.0)",
    },
    "zenodo": {
        "fn": download_zenodo,
        "desc": "Zenodo IoBT — edge-collected multi-firearm gunshot audio (CC-BY 4.0)",
    },
    "kaggle": {
        "fn": download_kaggle,
        "desc": "Kaggle — 851 gunshot clips across 8 gun models (AK-47, M16, M249, MP5, …)",
    },
}


def _apply_cli_paths(download_dir: str, output_dir: str):
    """Update module-level path constants from CLI arguments."""
    global DOWNLOAD_DIR, OUTPUT_DIR
    DOWNLOAD_DIR = download_dir
    OUTPUT_DIR = output_dir


def main():
    # Capture defaults before argparse references them
    default_download_dir = DOWNLOAD_DIR
    default_output_dir = OUTPUT_DIR

    ap = argparse.ArgumentParser(
        description="Download gunshot / military noise datasets automatically.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--source",
        choices=list(SOURCES.keys()),
        default=None,
        help="Download from a single source. Omit to download from ALL sources.",
    )
    ap.add_argument("--list", action="store_true", help="List available sources and exit.")
    ap.add_argument(
        "--download-dir",
        default=default_download_dir,
        help=f"Cache directory for raw archives (default: {default_download_dir})",
    )
    ap.add_argument(
        "--output-dir",
        default=default_output_dir,
        help=f"Destination for converted WAVs (default: {default_output_dir})",
    )
    args = ap.parse_args()

    # Allow overriding paths via CLI
    _apply_cli_paths(args.download_dir, args.output_dir)

    if args.list:
        print("Available gunshot/military noise sources:\n")
        for name, meta in SOURCES.items():
            print(f"  {name}")
            print(f"    {meta['desc']}\n")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    if args.source:
        sources_to_run = [args.source]
    else:
        sources_to_run = list(SOURCES.keys())

    total = 0
    for name in sources_to_run:
        try:
            n = SOURCES[name]["fn"]()
            total += n
        except Exception as exc:
            print(f"\n  [ERROR] {name} failed: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\n{'━' * 60}")
    print(f"  Total impulsive noise clips: {total}")
    print(f"  Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'━' * 60}")
    print("\nNext steps:")
    print("  python scripts/build_manifests.py")
    print("  python scripts/generate_dataset.py")


if __name__ == "__main__":
    main()
