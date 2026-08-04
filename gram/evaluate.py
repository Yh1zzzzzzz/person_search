"""Evaluate dual-encoder retrieval and the GRAM decoder reranker."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

from .data import (
    GalleryCollator,
    GalleryDataset,
    QueryCollator,
    QueryDataset,
    load_manifest,
    move_to_device,
)
from .metrics import retrieval_metrics
from .model import GRAM
from .rerank import rerank_topk


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[GRAM, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_args = dict(checkpoint["model_args"])
    num_classes = int(checkpoint["model"]["classifier.weight"].shape[0])
    model = GRAM(num_classes=num_classes, **model_args)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


def amp_context(device: torch.device, dtype: str):
    if device.type != "cuda" or dtype == "fp32":
        return nullcontext()
    amp_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


@torch.inference_mode()
def gallery_embeddings(
    model: GRAM,
    loader: DataLoader,
    device: torch.device,
    dtype: str,
    cache_device: torch.device | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    features = []
    person_ids = []
    image_cache = []
    prompt_input_ids = None
    prompt_attention_mask = None
    for batch in tqdm(loader, desc="gallery"):
        batch.pop("paths")
        mm_input_ids = batch.pop("mm_input_ids", None)
        mm_attention_mask = batch.pop("mm_attention_mask", None)
        if cache_device is not None and prompt_input_ids is None:
            assert mm_input_ids is not None and mm_attention_mask is not None
            prompt_input_ids = mm_input_ids[0].clone()
            prompt_attention_mask = mm_attention_mask[0].clone()
        batch = move_to_device(batch, device)
        with amp_context(device, dtype):
            if cache_device is None:
                embedding = model.encode_image(batch["pixel_values"])
                cached_tokens = None
            else:
                embedding, cached_tokens = model.encode_image_with_cache(batch["pixel_values"])
        features.append(F.normalize(embedding.float(), dim=-1).cpu())
        person_ids.append(batch["person_ids"].cpu())
        if cached_tokens is not None:
            image_cache.append(cached_tokens.detach().to(cache_device))
    return (
        torch.cat(features),
        torch.cat(person_ids),
        torch.cat(image_cache) if image_cache else None,
        prompt_input_ids,
        prompt_attention_mask,
    )


@torch.inference_mode()
def query_embeddings(
    model: GRAM, loader: DataLoader, device: torch.device, dtype: str
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    features = []
    person_ids = []
    captions: list[str] = []
    for batch in tqdm(loader, desc="queries"):
        captions.extend(batch.pop("captions"))
        batch = move_to_device(batch, device)
        with amp_context(device, dtype):
            embedding = model.encode_text(batch["input_ids"], batch["attention_mask"])
        features.append(F.normalize(embedding.float(), dim=-1).cpu())
        person_ids.append(batch["person_ids"].cpu())
    return torch.cat(features), torch.cat(person_ids), captions


@torch.inference_mode()
def decoder_scores(
    model: GRAM,
    processor,
    captions: list[str],
    image_cache: torch.Tensor,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    topk_indices: torch.Tensor,
    max_text_length: int,
    generation_temperature: float,
    device: torch.device,
    dtype: str,
) -> tuple[torch.Tensor, float]:
    """Score top-K candidates from cached visual tokens, without image I/O."""

    rows = []
    pad_token_id = int(processor.tokenizer.pad_token_id or 0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for caption, indices in tqdm(
        zip(captions, topk_indices.tolist(), strict=True),
        total=len(captions),
        desc="reranking",
    ):
        cache_indices = torch.tensor(indices, device=image_cache.device)
        candidate_cache = image_cache.index_select(0, cache_indices).to(device)
        input_ids = prompt_input_ids.unsqueeze(0).expand(len(indices), -1).to(device)
        attention_mask = prompt_attention_mask.unsqueeze(0).expand(len(indices), -1).to(device)
        target = processor.tokenizer(
            [caption] * len(indices),
            padding=True,
            truncation=True,
            max_length=max_text_length,
            return_tensors="pt",
        )
        labels = target["input_ids"].to(device)
        with amp_context(device, dtype):
            scores = model.generation_scores_from_cache(
                candidate_cache,
                input_ids,
                attention_mask,
                labels,
                pad_token_id,
                generation_temperature,
            )
        rows.append(scores.float().cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_ms = 1000.0 * (time.perf_counter() - start) / max(len(captions), 1)
    return torch.stack(rows), latency_ms


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, checkpoint = load_model(args.checkpoint, device)
    model_args = checkpoint["model_args"]
    config = checkpoint.get("config", {})
    dtype = str(model_args.get("dtype", "bf16"))
    processor = AutoProcessor.from_pretrained(model_args["model_name"])
    prompt = str(config.get("prompt", "Describe the person in this image in detail:"))
    if args.vision_cache_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--vision-cache-device cuda requires a CUDA device")
    cache_device = torch.device(args.vision_cache_device)

    records = load_manifest(args.manifest, split=args.split)
    gallery_loader = DataLoader(
        GalleryDataset(records),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=GalleryCollator(processor, prompt if args.rerank else None),
    )
    query_loader = DataLoader(
        QueryDataset(records, args.language),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=QueryCollator(processor.tokenizer, args.max_text_length),
    )
    (
        gallery_features,
        gallery_ids,
        image_cache,
        prompt_input_ids,
        prompt_attention_mask,
    ) = gallery_embeddings(
        model, gallery_loader, device, dtype, cache_device if args.rerank else None
    )
    query_features, query_ids, captions = query_embeddings(model, query_loader, device, dtype)
    coarse_scores = query_features @ gallery_features.T
    results: dict[str, object] = {
        "coarse": retrieval_metrics(query_ids, gallery_ids, scores=coarse_scores),
    }

    if args.rerank:
        assert image_cache is not None
        assert prompt_input_ids is not None and prompt_attention_mask is not None
        cache_gib = image_cache.numel() * image_cache.element_size() / (1024**3)
        results["vision_cache"] = {
            "device": str(cache_device),
            "shape": list(image_cache.shape),
            "size_gib": round(cache_gib, 3),
        }
        topk = min(args.topk, coarse_scores.shape[1])
        topk_indices = coarse_scores.topk(topk, dim=1).indices
        generation_scores, latency_ms = decoder_scores(
            model,
            processor,
            captions,
            image_cache,
            prompt_input_ids,
            prompt_attention_mask,
            topk_indices,
            args.max_text_length,
            args.generation_temperature,
            device,
            dtype,
        )
        ranking = rerank_topk(coarse_scores, generation_scores, topk_indices, args.alpha)
        results["reranked"] = retrieval_metrics(query_ids, gallery_ids, ranking=ranking)
        results["rerank_latency_ms_per_query"] = round(latency_ms, 3)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--language", default="en")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-text-length", type=int, default=96)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--topk", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--generation-temperature", type=float, default=2.0)
    parser.add_argument(
        "--vision-cache-device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Store decoder-ready gallery visual tokens on CPU (safe) or CUDA (lowest latency)",
    )
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
