#!/usr/bin/env python3
"""
Download / arrange the real corpora referenced in the PRD (section 8).

This script does NOT bundle copyrighted or license-restricted audio. It
downloads from each dataset's official host and arranges files into the
layout scan_speech_dir()/scan_noise_dir() expect:

    data/clean_speech/<speaker_id>/*.wav
    data/noise/impulsive/*.wav
    data/noise/rotor/*.wav
    data/noise/vehicle/*.wav
    data/noise/ambient/*.wav
    data/noise/sirens/*.wav

Run selectively, e.g.:
    python scripts/download_data.py --dataset librispeech-dev-clean
    python scripts/download_data.py --dataset esc50
    python scripts/download_data.py --list

Notes
-----
- LibriSpeech / VCTK / ESC-50 / UrbanSound8K / DEMAND are public research
  corpora with their own licenses (mostly CC-BY / CC0) — check each before
  redistribution, per NFR-10 / section 8.3.
- Defence-specific recordings (real gunfire, artillery, rotor wash under
  operational conditions) are NOT available as open datasets for obvious
  reasons. Per the PRD (section 8.2, "custom"), these must be sourced
  through DRDO's own field recording process or a cleared library provider.
  This script does not attempt to fetch weapons-fire audio from the open
  web — treat that category as "bring your own data" and drop files into
  data/noise/impulsive/.
- **For automated bulk gunshot/military noise downloads**, use the dedicated
  script  scripts/download_gunshot_data.py  which pulls from ESC-50 (sorted),
  Zenodo IoBT, and Kaggle, converts everything to 16 kHz mono WAV, and
  places it directly into data/noise/impulsive/.
- If your environment has no outbound internet access, use
  scripts/generate_synthetic_smoke_data.py instead to validate the pipeline
  end-to-end with synthetic placeholder audio, then swap in real data later
  without changing any downstream code.
"""
import argparse
import os
import tarfile
import zipfile
import urllib.request

DATASETS = {
    "librispeech-dev-clean": {
        "url": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
        "dest_subdir": "data/clean_speech",
        "kind": "tar.gz",
        "notes": "~5.4 hours, good for smoke-testing before pulling the full 1000h set.",
    },
    "librispeech-train-clean-100": {
        "url": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
        "dest_subdir": "data/clean_speech",
        "kind": "tar.gz",
        "notes": "~100 hours.",
    },
    "vctk": {
        "url": "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip",
        "dest_subdir": "data/clean_speech",
        "kind": "zip",
        "notes": "~44 hours, 110 speakers.",
    },
    "esc50": {
        "url": "https://github.com/karoldvl/ESC-50/archive/master.zip",
        "dest_subdir": "data/noise/_esc50_raw",
        "kind": "zip",
        "notes": "Contains a broad mix of environmental sounds; you'll need to "
                 "sort clips into impulsive/rotor/vehicle/ambient/sirens using "
                 "its meta/esc50.csv category labels — see sort step below.",
    },
    "urbansound8k": {
        "url": "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz",
        "dest_subdir": "data/noise/_urbansound8k_raw",
        "kind": "tar.gz",
        "notes": "Sirens + engine idling classes map to noise/sirens and noise/vehicle.",
    },
    "demand": {
        "url": "https://zenodo.org/record/1227121/files/",
        "dest_subdir": "data/noise/ambient",
        "kind": "manual",
        "notes": "DEMAND is split into many per-environment zip files on Zenodo; "
                 "download the ones you need (e.g. PARK, TRAFFIC, CAFE) individually.",
    },
    "zenodo-gunshot-iobt": {
        "url": "https://zenodo.org/records/7004819/files/edge-collected-gunshot-audio.zip?download=1",
        "dest_subdir": "data/noise/impulsive",
        "kind": "zip",
        "notes": "IoBT edge-collected multi-firearm gunshot audio (CC-BY 4.0). "
                 "For bulk download with auto-conversion, prefer: "
                 "python scripts/download_gunshot_data.py",
    },
}


def download(url: str, dest_path: str):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Downloading {url} -> {dest_path}")
    urllib.request.urlretrieve(url, dest_path)


def extract(archive_path: str, dest_dir: str, kind: str):
    os.makedirs(dest_dir, exist_ok=True)
    if kind == "tar.gz":
        with tarfile.open(archive_path) as t:
            t.extractall(dest_dir)
    elif kind == "zip":
        with zipfile.ZipFile(archive_path) as z:
            z.extractall(dest_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS.keys()))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--download-dir", default="data/_downloads")
    args = ap.parse_args()

    if args.list or not args.dataset:
        print("Available datasets:\n")
        for name, meta in DATASETS.items():
            print(f"  {name}")
            print(f"    -> {meta['dest_subdir']}")
            print(f"    {meta['notes']}\n")
        return

    meta = DATASETS[args.dataset]
    if meta["kind"] == "manual":
        print(f"'{args.dataset}' requires manual download: {meta['url']}")
        print(f"Notes: {meta['notes']}")
        return

    os.makedirs(args.download_dir, exist_ok=True)
    archive_path = os.path.join(args.download_dir, os.path.basename(meta["url"]))
    download(meta["url"], archive_path)
    extract(archive_path, meta["dest_subdir"], meta["kind"])
    print(f"Extracted to {meta['dest_subdir']}. Re-sort into category subfolders if needed.")


if __name__ == "__main__":
    main()
