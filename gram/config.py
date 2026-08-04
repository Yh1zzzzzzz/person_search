"""Typed JSON configuration for training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    stage: int
    manifest: str
    output_dir: str
    model_name: str = "google/t5gemma-2-270m-270m"
    checkpoint: str | None = None
    languages: list[str] | None = None
    feature_dim: int = 512
    projection_hidden_dim: int = 2048
    retrieval_temperature: float = 0.02
    lambda_icc: float = 1.0
    lambda_lm: float = 0.5
    lambda_mst: float = 1.0
    lambda_mdm: float = 1.0
    distillation_temperature: float = 2.0
    backbone_lr: float = 5e-5
    head_lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 60
    warmup_epochs: int = 5
    batch_size: int = 64
    gradient_accumulation: int = 1
    num_workers: int = 8
    max_text_length: int = 96
    prompt: str = "Describe the person in this image in detail:"
    dtype: str = "bf16"
    attention_implementation: str = "sdpa"
    gradient_checkpointing: bool = True
    seed: int = 1

    @classmethod
    def from_json(cls, path: str | Path) -> TrainConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = json.load(handle)
        valid = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - valid)
        if unknown:
            raise ValueError(f"unknown configuration keys: {', '.join(unknown)}")
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        if self.stage not in (1, 2):
            raise ValueError("stage must be 1 or 2")
        if self.stage == 2 and not self.checkpoint:
            raise ValueError("Stage 2 requires a Stage 1 checkpoint")
        if self.stage == 2 and not self.languages:
            raise ValueError("Stage 2 requires at least one target language")
        if self.batch_size < 1 or self.gradient_accumulation < 1:
            raise ValueError("batch_size and gradient_accumulation must be positive")
        if self.epochs < 1 or not 0 <= self.warmup_epochs <= self.epochs:
            raise ValueError("warmup_epochs must be between 0 and epochs")
        if self.retrieval_temperature <= 0 or self.distillation_temperature <= 0:
            raise ValueError("temperatures must be positive")
        if self.dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError("dtype must be bf16, fp16, or fp32")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def model_args(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "feature_dim": self.feature_dim,
            "projection_hidden_dim": self.projection_hidden_dim,
            "retrieval_temperature": self.retrieval_temperature,
            "attention_implementation": self.attention_implementation,
            "dtype": self.dtype,
        }
