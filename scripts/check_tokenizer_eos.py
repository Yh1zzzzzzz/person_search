import argparse
import random
from collections import Counter
import json
import os

import torch
from transformers import AutoProcessor


def iter_cuhk_captions(root: str, split: str = "train"):
    # Avoid importing local "datasets" package because it can be shadowed by HuggingFace `datasets`.
    # CUHK-PEDES raw annotation format is a list of dicts with keys: split/captions/file_path/id.
    anno_path = os.path.join(root, "CUHK-PEDES", "reid_raw.json")
    if not os.path.exists(anno_path):
        raise FileNotFoundError(f"Cannot find annotation file: {anno_path}")

    with open(anno_path, "r", encoding="utf-8") as f:
        annos = json.load(f)

    split = split.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split={split}")

    for anno in annos:
        if anno.get("split") != split:
            continue
        for caption in anno.get("captions", []):
            yield caption


def analyze(tokenizer, captions, max_length: int, pad_id: int, eos_id: int, limit: int):
    stats = Counter()
    examples = {
        "no_eos": [],
        "eos_not_last": [],
    }

    for i, cap in enumerate(captions):
        if limit and i >= limit:
            break
        tok = tokenizer(
            cap,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = tok.input_ids[0]

        # last non-pad token index
        non_pad = (input_ids != pad_id).nonzero(as_tuple=False).view(-1)
        if non_pad.numel() == 0:
            stats["all_pad"] += 1
            continue
        last_non_pad_idx = int(non_pad[-1].item())
        last_non_pad_id = int(input_ids[last_non_pad_idx].item())

        has_eos = bool((input_ids == eos_id).any().item())
        stats["n"] += 1
        stats["has_eos"] += int(has_eos)
        stats["eos_is_last_non_pad"] += int(last_non_pad_id == eos_id)

        if not has_eos and len(examples["no_eos"]) < 5:
            examples["no_eos"].append((cap, input_ids.tolist()))
        if has_eos and last_non_pad_id != eos_id and len(examples["eos_not_last"]) < 5:
            # show last few ids for debugging
            tail = input_ids[max(0, last_non_pad_idx - 10) : last_non_pad_idx + 1].tolist()
            examples["eos_not_last"].append((cap, tail, last_non_pad_id, last_non_pad_idx))

    return stats, examples


def ensure_eos_like_dataset(input_ids: torch.LongTensor, pad_id: int, eos_id: int) -> torch.LongTensor:
    """Mirror datasets/hf_t5gemma2.py::_ensure_eos_token behavior."""
    if bool((input_ids == eos_id).any().item()):
        return input_ids
    pad_pos = (input_ids == pad_id).nonzero(as_tuple=False).view(-1)
    if pad_pos.numel() > 0:
        input_ids = input_ids.clone()
        input_ids[int(pad_pos[0].item())] = eos_id
        return input_ids
    input_ids = input_ids.clone()
    input_ids[-1] = eos_id
    return input_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_model", default="T5_270M_Base")
    ap.add_argument("--dataset_root", default="training_data")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--text_length", type=int, default=77)
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--apply_eos_fix",
        action="store_true",
        help="Simulate dataset-side EOS insertion (recommended to validate your training pipeline).",
    )
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.hf_model, local_files_only=True)
    tokenizer = processor.tokenizer

    pad_id = int(tokenizer.pad_token_id)
    eos_id = int(tokenizer.eos_token_id)

    # sample captions deterministically
    caps = list(iter_cuhk_captions(args.dataset_root, split=args.split))
    random.Random(args.seed).shuffle(caps)

    def caption_iter():
        for cap in caps[: args.limit if args.limit else None]:
            yield cap

    # Inline analyze so we can optionally apply the dataset EOS fix
    stats = Counter()
    examples = {"no_eos": [], "eos_not_last": []}
    for i, cap in enumerate(caption_iter()):
        tok = tokenizer(
            cap,
            padding="max_length",
            truncation=True,
            max_length=args.text_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = tok.input_ids[0]
        if args.apply_eos_fix:
            input_ids = ensure_eos_like_dataset(input_ids, pad_id=pad_id, eos_id=eos_id)

        non_pad = (input_ids != pad_id).nonzero(as_tuple=False).view(-1)
        if non_pad.numel() == 0:
            stats["all_pad"] += 1
            continue
        last_non_pad_idx = int(non_pad[-1].item())
        last_non_pad_id = int(input_ids[last_non_pad_idx].item())
        has_eos = bool((input_ids == eos_id).any().item())

        stats["n"] += 1
        stats["has_eos"] += int(has_eos)
        stats["eos_is_last_non_pad"] += int(last_non_pad_id == eos_id)

        if not has_eos and len(examples["no_eos"]) < 5:
            examples["no_eos"].append((cap, input_ids.tolist()))
        if has_eos and last_non_pad_id != eos_id and len(examples["eos_not_last"]) < 5:
            tail = input_ids[max(0, last_non_pad_idx - 10) : last_non_pad_idx + 1].tolist()
            examples["eos_not_last"].append((cap, tail, last_non_pad_id, last_non_pad_idx))

    n = stats["n"]
    print(f"Tokenizer: pad_id={pad_id} eos_id={eos_id} apply_eos_fix={bool(args.apply_eos_fix)}")
    print(f"Dataset root={args.dataset_root} split={args.split}")
    print(f"Checked {n} samples (limit={args.limit}, max_length={args.text_length})")
    if n:
        print(f"has_eos: {stats['has_eos']}/{n} = {stats['has_eos']/n:.3f}")
        print(f"eos_is_last_non_pad: {stats['eos_is_last_non_pad']}/{n} = {stats['eos_is_last_non_pad']/n:.3f}")

    if examples["no_eos"]:
        print("\nExamples: no EOS found")
        for cap, ids in examples["no_eos"]:
            print("-", cap)
            print("  ids(head):", ids[:20], "... tail:", ids[-10:])

    if examples["eos_not_last"]:
        print("\nExamples: EOS exists but not last non-pad")
        for cap, tail, last_id, last_idx in examples["eos_not_last"]:
            print("-", cap)
            print(f"  last_non_pad_idx={last_idx} last_non_pad_id={last_id} tail={tail}")


if __name__ == "__main__":
    main()
