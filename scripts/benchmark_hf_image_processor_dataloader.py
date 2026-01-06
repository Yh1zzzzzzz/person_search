#!/usr/bin/env python3
"""Benchmark whether HF image_processor preprocessing becomes a CPU bottleneck.

This script measures (per step):
  1) DataLoader fetch time (includes image decode + PIL aug + HF image_processor)
  2) H2D transfer time (if CUDA)
  3) Optional compute time (dummy GPU work, or real vision_tower forward if requested)

Usage examples:
  python scripts/benchmark_hf_image_processor_dataloader.py \
    --dataset-name CUHK-PEDES --root-dir training_data --hf-model-name-or-path T5_270M_Base \
    --t5-image-size 896 --batch-size 32 --num-workers 8 --steps 200

  # With dummy GPU compute to see overlap
  python scripts/benchmark_hf_image_processor_dataloader.py --device cuda --dummy-compute

Notes:
- Requires torch + transformers installed.
- If you want a fair test, run on the same machine and keep I/O consistent.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from types import SimpleNamespace
from typing import Dict, List, Optional
from contextlib import nullcontext


# Make sure local imports (e.g. this repo's `datasets/`) work even when the script
# is launched from a different working directory.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    values_sorted = sorted(values)
    k = int(round((p / 100.0) * (len(values_sorted) - 1)))
    k = max(0, min(k, len(values_sorted) - 1))
    return values_sorted[k]


def _make_args(ns: argparse.Namespace):
    # build_dataloader expects attribute-style args.
    # Keep only what build_dataloader and downstream datasets use.
    return SimpleNamespace(
        # general
        num_workers=int(ns.num_workers),
        dataset_name=str(ns.dataset_name),
        root_dir=str(ns.root_dir),
        backbone=str(ns.backbone),
        hf_model_name_or_path=str(ns.hf_model_name_or_path),
        hf_use_fast=bool(ns.hf_use_fast),
        training=True,
        distributed=False,
        sampler=str(ns.sampler),
        batch_size=int(ns.batch_size),
        test_batch_size=int(ns.batch_size),
        num_instance=int(ns.num_instance),
        val_dataset=str(ns.val_dataset),
        # t5-related
        t5_image_size=int(ns.t5_image_size),
        img_aug=bool(ns.img_aug),
        text_length=int(ns.text_length),
        mm_max_length=int(ns.mm_max_length),
        loss_names=str(ns.loss_names),
        gen_prompt=str(ns.gen_prompt),
        gen_prompt_length=int(ns.gen_prompt_length),
        # legacy flags used by non-t5 branch (kept for safety)
        MLM=False,
        img_size=(384, 128),
    )


def _sync_if_cuda(torch_mod, device: str) -> None:
    if device.startswith("cuda") and torch_mod.cuda.is_available():
        torch_mod.cuda.synchronize()


def _dummy_gpu_compute(torch_mod, pixel_values):
    # Simple conv-like workload to approximate some GPU work.
    # Keep it lightweight and deterministic.
    x = pixel_values
    # [B,3,H,W] -> flatten last two dims then matmul
    b, c, h, w = x.shape
    x2 = x.view(b, c, h * w)
    w2 = torch_mod.randn((c, c), device=x.device, dtype=x.dtype)
    y = torch_mod.matmul(w2, x2)  # [c,c] x [b,c,hw] -> broadcast-ish with matmul rules
    # reduce to scalar to prevent lazy optimizations
    return y.sum()


def _move_batch_to_device(torch_mod, batch, device: str):
    if device.startswith("cuda") and torch_mod.cuda.is_available():
        out = {}
        for k, v in batch.items():
            if torch_mod.is_tensor(v):
                out[k] = v.to(device, non_blocking=True)
            else:
                out[k] = v
        return out
    return batch


def _build_model_for_benchmark(torch_mod, ns: argparse.Namespace, num_classes: int = 1):
    if ns.backbone == "t5gemma2_vion_tower":
        from model.t5Geema2_vion_tower import build_person_search_t5gemma2_vion_tower

        return build_person_search_t5gemma2_vion_tower(ns, num_classes=num_classes)

    from model.T5Gemma2_270 import build_person_search_t5gemma2

    return build_person_search_t5gemma2(ns, num_classes=num_classes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-name", required=True, help="e.g. CUHK-PEDES / ICFG-PEDES / RSTPReid")
    ap.add_argument("--root-dir", required=True, help="dataset root, passed to dataset factory")
    ap.add_argument("--hf-model-name-or-path", default="T5_270M_Base")
    ap.add_argument(
        "--hf-use-fast",
        dest="hf_use_fast",
        action="store_true",
        help="use HuggingFace fast image processor/tokenizer when available",
    )
    ap.add_argument(
        "--hf-use-slow",
        dest="hf_use_fast",
        action="store_false",
        help="force slow processor (may be significantly slower)",
    )
    ap.set_defaults(hf_use_fast=True)
    ap.add_argument("--backbone", default="t5gemma2", choices=["t5gemma2", "t5gemma2_vion_tower"], help="must be a T5 backbone")

    ap.add_argument("--t5-image-size", type=int, default=896, help="must match model vision_config.image_size when model forbids adaptation")
    ap.add_argument("--text-length", type=int, default=77)
    ap.add_argument("--mm-max-length", type=int, default=512)
    ap.add_argument("--loss-names", type=str, default="sdm+id", help="string; include 'mlm' to enable MLM path")
    ap.add_argument("--gen-prompt", type=str, default="Caption")
    ap.add_argument("--gen-prompt-length", type=int, default=32)

    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--sampler", type=str, default="random", choices=["random", "identity"])
    ap.add_argument("--num-instance", type=int, default=4)
    ap.add_argument("--val-dataset", type=str, default="val", choices=["val", "test"])

    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--device", type=str, default="cuda", help="cuda / cpu")
    ap.add_argument("--img-aug", action="store_true", help="enable PIL augmentation in dataset")

    ap.add_argument("--amp", action="store_true", help="enable torch.autocast during compute (recommended)")
    ap.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"], help="autocast dtype")

    ap.add_argument("--dummy-compute", action="store_true", help="run a small dummy GPU compute to estimate GPU busy time")
    ap.add_argument(
        "--real-forward",
        type=str,
        default="none",
        choices=["none", "vision", "full"],
        help="run real model compute: 'vision' uses model.encode_image_only; 'full' runs model(batch)",
    )
    ap.add_argument(
        "--real-backward",
        action="store_true",
        help="when --real-forward=full, also run backward on total_loss (training-like)",
    )
    ap.add_argument("--print-every", type=int, default=50)

    # Model knobs (match builder defaults; only used when --real-forward != none)
    ap.add_argument("--feature-dim", type=int, default=1024)
    ap.add_argument("--projector-hidden-dim", type=int, default=2048)
    ap.add_argument("--bnneck", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.02)
    ap.add_argument("--gen-loss-weight", type=float, default=0.0)
    ap.add_argument("--id-loss-weight", type=float, default=1.0)
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--attn-implementation", type=str, default="sdpa")

    ns = ap.parse_args()

    # Lazy imports so the script can be inspected without deps.
    import torch

    from datasets.build import build_dataloader

    device = ns.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but torch.cuda.is_available()=False; falling back to CPU")
        device = "cpu"

    args = _make_args(ns)

    print("=== Benchmark Config ===")
    print(f"dataset={args.dataset_name} root_dir={args.root_dir}")
    print(f"hf_model={args.hf_model_name_or_path} backbone={args.backbone}")
    print(f"t5_image_size={args.t5_image_size} batch_size={args.batch_size} num_workers={args.num_workers}")
    print(f"img_aug={args.img_aug} steps={ns.steps} warmup={ns.warmup} device={device}")
    print(f"dummy_compute={bool(ns.dummy_compute)}")
    print(f"real_forward={ns.real_forward} real_backward={bool(ns.real_backward)}")
    print(f"amp={bool(ns.amp)} amp_dtype={ns.amp_dtype}")

    train_loader, _, _, num_classes = build_dataloader(args)

    model = None
    use_real = (ns.real_forward != "none")
    if use_real:
        print("Loading real model for benchmark...")
        model = _build_model_for_benchmark(torch, ns, num_classes=int(num_classes))
        if device.startswith("cuda"):
            model = model.to(device)
        if ns.real_forward == "full":
            model.train(bool(ns.real_backward))
        else:
            model.eval()

    use_amp = bool(ns.amp) and device.startswith("cuda")
    amp_dtype = torch.bfloat16 if str(ns.amp_dtype).lower() == "bf16" else torch.float16
    autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()

    fetch_ms: List[float] = []
    h2d_ms: List[float] = []
    compute_ms: List[float] = []

    it = iter(train_loader)

    # Warmup
    for _ in range(int(ns.warmup)):
        t0 = time.perf_counter()
        batch = next(it)
        _ = time.perf_counter() - t0

        if device.startswith("cuda"):
            _sync_if_cuda(torch, device)
            batch_dev = _move_batch_to_device(torch, batch, device)
            _sync_if_cuda(torch, device)

            if use_real:
                if ns.real_forward == "vision":
                    with torch.no_grad():
                        with autocast_ctx:
                            _ = model.encode_image_only(batch_dev["pixel_values"])
                else:
                    with autocast_ctx:
                        out = model(batch_dev)
                    if ns.real_backward:
                        loss = out.get("total_loss")
                        if loss is None:
                            loss = sum(v for k, v in out.items() if torch.is_tensor(v) and "loss" in k)
                        model.zero_grad(set_to_none=True)
                        loss.backward()
            elif ns.dummy_compute:
                pv = batch_dev.get("pixel_values")
                if pv is not None:
                    with autocast_ctx:
                        loss = _dummy_gpu_compute(torch, pv)
                    _ = float(loss.detach().cpu().item())

            _sync_if_cuda(torch, device)

    # Timed steps
    start_wall = time.perf_counter()
    for step in range(int(ns.steps)):
        t0 = time.perf_counter()
        batch = next(it)
        t1 = time.perf_counter()

        pv = batch.get("pixel_values")
        if pv is None:
            raise RuntimeError("Batch has no 'pixel_values'. Are you using the T5Gemma2 dataset?")

        # H2D
        if device.startswith("cuda"):
            _sync_if_cuda(torch, device)
            t2 = time.perf_counter()
            batch = _move_batch_to_device(torch, batch, device)
            _sync_if_cuda(torch, device)
            t3 = time.perf_counter()
        else:
            t2 = t3 = time.perf_counter()

        # Optional compute (real model preferred)
        if device.startswith("cuda") and (use_real or ns.dummy_compute):
            _sync_if_cuda(torch, device)
            t4 = time.perf_counter()

            if use_real:
                if ns.real_forward == "vision":
                    with torch.no_grad():
                        with autocast_ctx:
                            _ = model.encode_image_only(batch["pixel_values"])
                else:
                    with autocast_ctx:
                        out = model(batch)
                    if ns.real_backward:
                        loss = out.get("total_loss")
                        if loss is None:
                            loss = sum(v for k, v in out.items() if torch.is_tensor(v) and "loss" in k)
                        model.zero_grad(set_to_none=True)
                        loss.backward()
            else:
                pv = batch.get("pixel_values")
                with autocast_ctx:
                    loss = _dummy_gpu_compute(torch, pv)
                _ = float(loss.detach().cpu().item())

            _sync_if_cuda(torch, device)
            t5 = time.perf_counter()
        else:
            t4 = t5 = time.perf_counter()

        fetch_ms.append((t1 - t0) * 1000.0)
        h2d_ms.append((t3 - t2) * 1000.0)
        compute_ms.append((t5 - t4) * 1000.0)

        if ns.print_every > 0 and (step + 1) % int(ns.print_every) == 0:
            print(
                f"step {step+1}/{ns.steps} | "
                f"fetch={fetch_ms[-1]:.1f}ms h2d={h2d_ms[-1]:.1f}ms compute={compute_ms[-1]:.1f}ms"
            )

    total_wall = time.perf_counter() - start_wall
    images = int(ns.steps) * int(args.batch_size)
    ips = images / max(total_wall, 1e-9)

    def _summ(name: str, xs: List[float]) -> Dict[str, float]:
        return {
            f"{name}_mean_ms": statistics.mean(xs),
            f"{name}_p50_ms": _percentile(xs, 50),
            f"{name}_p90_ms": _percentile(xs, 90),
            f"{name}_p99_ms": _percentile(xs, 99),
        }

    summary = {}
    summary.update(_summ("fetch", fetch_ms))
    summary.update(_summ("h2d", h2d_ms))
    summary.update(_summ("compute", compute_ms))

    print("\n=== Summary ===")
    print(f"wall_time_s={total_wall:.3f} | images={images} | images_per_s={ips:.2f}")
    for k in sorted(summary.keys()):
        print(f"{k}={summary[k]:.3f}")

    # Quick heuristic
    fetch_mean = summary["fetch_mean_ms"]
    compute_mean = summary["compute_mean_ms"]
    if device.startswith("cuda") and (ns.dummy_compute or use_real):
        ratio = fetch_mean / max(compute_mean, 1e-9)
        print("\n=== Heuristic ===")
        print(f"fetch/compute ~= {ratio:.2f}")
        if ratio > 1.5:
            print("Likely CPU/preprocess bottleneck (GPU may wait for batches).")
        elif ratio < 0.7:
            print("Likely GPU/compute bottleneck (DataLoader keeps up).")
        else:
            print("Mixed; tune num_workers/prefetch and re-check.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
