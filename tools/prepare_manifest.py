#!/usr/bin/env python3
"""Convert standard English TBPR annotations to GRAM JSONL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DATASETS = {
    "CUHK-PEDES": ("CUHK-PEDES", ("reid_raw.json",), "file_path"),
    "ICFG-PEDES": ("ICFG-PEDES", ("ICFG-PEDES.json", "ICFG_PEDES.json"), "file_path"),
    "RSTPReid": ("RSTPReid", ("data_captions.json",), "img_path"),
}


def find_annotation(directory: Path, candidates: tuple[str, ...]) -> Path:
    for name in candidates:
        path = directory / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"none of {candidates} exists under {directory}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--root", required=True, type=Path, help="Directory containing datasets")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    dataset_dir_name, annotation_names, image_key = DATASETS[args.dataset]
    dataset_dir = args.root.expanduser().resolve() / dataset_dir_name
    annotation = find_annotation(dataset_dir, annotation_names)
    with annotation.open("r", encoding="utf-8") as handle:
        raw_records = json.load(handle)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for raw in raw_records:
            image = dataset_dir / "imgs" / raw[image_key]
            relative_image = os.path.relpath(image, args.output.parent.resolve())
            record = {
                "image": relative_image,
                "pid": int(raw["id"]),
                "split": str(raw["split"]),
                "captions": [{"en": caption} for caption in raw["captions"]],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(raw_records)} image records to {args.output}")


if __name__ == "__main__":
    main()
