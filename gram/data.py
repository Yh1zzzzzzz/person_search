"""A compact JSONL data interface for English and M-PEDES training."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass(frozen=True)
class ManifestRecord:
    image: Path
    pid: int
    split: str
    captions: tuple[dict[str, str], ...]


def load_manifest(path: str | Path, split: str | None = None) -> list[ManifestRecord]:
    """Load and validate the repository JSONL format."""

    manifest = Path(path).expanduser().resolve()
    records: list[ManifestRecord] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            required = {"image", "pid", "split", "captions"}
            missing = required - raw.keys()
            if missing:
                raise ValueError(f"{manifest}:{line_number}: missing {sorted(missing)}")
            if split is not None and raw["split"] != split:
                continue
            image = Path(raw["image"]).expanduser()
            if not image.is_absolute():
                image = (manifest.parent / image).resolve()
            captions = tuple(dict(group) for group in raw["captions"])
            if not captions or any(not group for group in captions):
                raise ValueError(f"{manifest}:{line_number}: captions cannot be empty")
            records.append(
                ManifestRecord(
                    image=image,
                    pid=int(raw["pid"]),
                    split=str(raw["split"]),
                    captions=captions,
                )
            )
    if not records:
        suffix = f" for split={split!r}" if split is not None else ""
        raise ValueError(f"no records found in {manifest}{suffix}")
    return records


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


class Stage1Dataset(Dataset):
    """Flatten every English caption into a Stage 1 training sample."""

    def __init__(self, records: Sequence[ManifestRecord]):
        raw_samples = [
            (record.image, record.pid, group["en"])
            for record in records
            for group in record.captions
            if "en" in group
        ]
        if not raw_samples:
            raise ValueError("Stage 1 requires English captions under the 'en' key")
        pid_map = {
            pid: index for index, pid in enumerate(sorted({item[1] for item in raw_samples}))
        }
        self.samples = [(image, pid_map[pid], caption) for image, pid, caption in raw_samples]
        self.num_classes = len(pid_map)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Image.Image, int, str]:
        path, pid, caption = self.samples[index]
        return _open_rgb(path), pid, caption


class Stage2Dataset(Dataset):
    """Parallel English/target-language samples for cross-lingual distillation."""

    def __init__(self, records: Sequence[ManifestRecord], languages: Sequence[str]):
        self.languages = tuple(languages)
        needed = {"en", *self.languages}
        raw_samples = [
            (record.image, record.pid, group)
            for record in records
            for group in record.captions
            if needed.issubset(group)
        ]
        if not raw_samples:
            raise ValueError(
                "Stage 2 found no fully aligned caption group for: " + ", ".join(sorted(needed))
            )
        pid_map = {
            pid: index for index, pid in enumerate(sorted({item[1] for item in raw_samples}))
        }
        self.samples = [(image, pid_map[pid], group) for image, pid, group in raw_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Image.Image, int, dict[str, str]]:
        path, pid, captions = self.samples[index]
        return _open_rgb(path), pid, captions


class GalleryDataset(Dataset):
    def __init__(self, records: Sequence[ManifestRecord]):
        unique: dict[Path, int] = {}
        for record in records:
            unique.setdefault(record.image, record.pid)
        self.samples = [(path, pid) for path, pid in unique.items()]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Image.Image, int, str]:
        path, pid = self.samples[index]
        return _open_rgb(path), pid, str(path)


class QueryDataset(Dataset):
    def __init__(self, records: Sequence[ManifestRecord], language: str):
        self.samples = [
            (record.pid, group[language])
            for record in records
            for group in record.captions
            if language in group
        ]
        if not self.samples:
            raise ValueError(f"no query captions found for language {language!r}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[int, str]:
        return self.samples[index]


def _tokenize(tokenizer: Any, texts: Sequence[str], max_length: int) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}


class Stage1Collator:
    def __init__(self, processor: Any, prompt: str, max_text_length: int):
        self.processor = processor
        self.prompt = prompt
        self.max_text_length = max_text_length

    def __call__(self, samples: Sequence[tuple[Image.Image, int, str]]) -> dict[str, Any]:
        images, pids, captions = zip(*samples, strict=True)
        prompts = [f"<start_of_image> {self.prompt}"] * len(images)
        multimodal = self.processor(
            images=list(images), text=prompts, padding=True, return_tensors="pt"
        )
        query = _tokenize(self.processor.tokenizer, captions, self.max_text_length)
        return {
            "pixel_values": multimodal["pixel_values"],
            "mm_input_ids": multimodal["input_ids"],
            "mm_attention_mask": multimodal["attention_mask"],
            "query_input_ids": query["input_ids"],
            "query_attention_mask": query["attention_mask"],
            "labels": query["input_ids"].clone(),
            "person_ids": torch.tensor(pids, dtype=torch.long),
        }


class Stage2Collator:
    def __init__(self, processor: Any, languages: Sequence[str], max_text_length: int):
        self.processor = processor
        self.languages = tuple(languages)
        self.max_text_length = max_text_length

    def __call__(
        self, samples: Sequence[tuple[Image.Image, int, dict[str, str]]]
    ) -> dict[str, Any]:
        images, pids, caption_groups = zip(*samples, strict=True)
        pixels = self.processor.image_processor(images=list(images), return_tensors="pt")
        english = _tokenize(
            self.processor.tokenizer,
            [group["en"] for group in caption_groups],
            self.max_text_length,
        )
        targets = {
            language: _tokenize(
                self.processor.tokenizer,
                [group[language] for group in caption_groups],
                self.max_text_length,
            )
            for language in self.languages
        }
        return {
            "pixel_values": pixels["pixel_values"],
            "person_ids": torch.tensor(pids, dtype=torch.long),
            "english": english,
            "targets": targets,
        }


class GalleryCollator:
    def __init__(self, processor: Any, prompt: str | None):
        self.processor = processor
        self.prompt = prompt

    def __call__(self, samples: Sequence[tuple[Image.Image, int, str]]) -> dict[str, Any]:
        images, pids, paths = zip(*samples, strict=True)
        if self.prompt is None:
            processed = self.processor.image_processor(images=list(images), return_tensors="pt")
        else:
            prompts = [f"<start_of_image> {self.prompt}"] * len(images)
            processed = self.processor(
                images=list(images), text=prompts, padding=True, return_tensors="pt"
            )
        result = {
            "pixel_values": processed["pixel_values"],
            "person_ids": torch.tensor(pids, dtype=torch.long),
            "paths": list(paths),
        }
        if self.prompt is not None:
            result["mm_input_ids"] = processed["input_ids"]
            result["mm_attention_mask"] = processed["attention_mask"]
        return result


class QueryCollator:
    def __init__(self, tokenizer: Any, max_text_length: int):
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

    def __call__(self, samples: Sequence[tuple[int, str]]) -> dict[str, Any]:
        pids, captions = zip(*samples, strict=True)
        tokens = _tokenize(self.tokenizer, captions, self.max_text_length)
        return {
            **tokens,
            "person_ids": torch.tensor(pids, dtype=torch.long),
            "captions": list(captions),
        }


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value
