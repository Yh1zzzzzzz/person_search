"""Two-stage training entry point."""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_cosine_schedule_with_warmup

from .config import TrainConfig
from .data import (
    Stage1Collator,
    Stage1Dataset,
    Stage2Collator,
    Stage2Dataset,
    load_manifest,
    move_to_device,
)
from .model import GRAM


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    return torch.load(Path(path), map_location=device, weights_only=False)


def build_model_from_checkpoint(
    checkpoint: dict[str, Any], num_classes: int, device: torch.device
) -> GRAM:
    model_args = dict(checkpoint["model_args"])
    model_args["num_classes"] = num_classes
    model = GRAM(**model_args)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device)


def optimizer_for(model: GRAM, config: TrainConfig) -> torch.optim.Optimizer:
    heads = [model.vision_projection, model.text_projection, model.classifier]
    head_parameters = [p for module in heads for p in module.parameters() if p.requires_grad]
    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    groups = []
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": config.backbone_lr})
    if head_parameters:
        groups.append({"params": head_parameters, "lr": config.head_lr})
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def save_checkpoint(
    path: Path,
    model: GRAM,
    config: TrainConfig,
    epoch: int,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "model_args": config.model_args(),
            "config": config.to_dict(),
            "epoch": epoch,
        },
        path,
    )


def train(config: TrainConfig) -> None:
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2, ensure_ascii=False)
    with (output_dir / "environment.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
            handle,
            indent=2,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_manifest(config.manifest, split="train")
    processor = AutoProcessor.from_pretrained(config.model_name)

    teacher = None
    if config.stage == 1:
        dataset = Stage1Dataset(records)
        collator = Stage1Collator(processor, config.prompt, config.max_text_length)
        model = GRAM(num_classes=dataset.num_classes, **config.model_args()).to(device)
    else:
        dataset = Stage2Dataset(records, config.languages or [])
        collator = Stage2Collator(processor, config.languages or [], config.max_text_length)
        checkpoint = load_checkpoint(config.checkpoint or "", device)
        num_classes = int(checkpoint["config"].get("num_classes", 1))
        if "classifier.weight" in checkpoint["model"]:
            num_classes = int(checkpoint["model"]["classifier.weight"].shape[0])
        model = build_model_from_checkpoint(checkpoint, num_classes, device)
        teacher = build_model_from_checkpoint(checkpoint, num_classes, device)
        teacher.requires_grad_(False).eval()
        model.freeze_for_distillation()

    if config.gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable()
        model.backbone.config.use_cache = False

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        drop_last=True,
        collate_fn=collator,
    )
    optimizer = optimizer_for(model, config)
    update_steps_per_epoch = math.ceil(len(loader) / config.gradient_accumulation)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_epochs * update_steps_per_epoch,
        num_training_steps=config.epochs * update_steps_per_epoch,
    )

    use_amp = device.type == "cuda" and config.dtype in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if config.dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and config.dtype == "fp16")
    pad_token_id = int(processor.tokenizer.pad_token_id or 0)
    best_loss = float("inf")
    metrics_path = output_dir / "metrics.jsonl"

    for epoch in range(1, config.epochs + 1):
        model.train()
        if config.stage == 2:
            # Frozen modules must also keep deterministic dropout/normalization behavior.
            model.encoder.vision_tower.eval()
            model.encoder.multi_modal_projector.eval()
            model.vision_projection.eval()
        optimizer.zero_grad(set_to_none=True)
        sums: dict[str, float] = {}
        progress = tqdm(loader, desc=f"epoch {epoch}/{config.epochs}")
        for step, batch in enumerate(progress, start=1):
            batch = move_to_device(batch, device)
            autocast = (
                torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
            )
            with autocast:
                if config.stage == 1:
                    losses = model.stage1_loss(
                        batch, pad_token_id, config.lambda_icc, config.lambda_lm
                    )
                else:
                    assert teacher is not None
                    losses = model.stage2_loss(
                        teacher,
                        batch,
                        config.lambda_mst,
                        config.lambda_mdm,
                        config.distillation_temperature,
                    )
                loss = losses["loss"] / config.gradient_accumulation

            scaler.scale(loss).backward()
            should_update = step % config.gradient_accumulation == 0 or step == len(loader)
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            for name, value in losses.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach().float())
            progress.set_postfix(loss=f"{float(losses['loss'].detach()):.4f}")

        epoch_metrics = {name: value / len(loader) for name, value in sums.items()}
        epoch_metrics.update({"epoch": epoch, "lr": scheduler.get_last_lr()[0]})
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_metrics) + "\n")
        save_checkpoint(output_dir / "last.pt", model, config, epoch)
        if epoch_metrics["loss"] < best_loss:
            best_loss = epoch_metrics["loss"]
            save_checkpoint(output_dir / "best.pt", model, config, epoch)
        print(json.dumps(epoch_metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    train(TrainConfig.from_json(args.config))


if __name__ == "__main__":
    main()
