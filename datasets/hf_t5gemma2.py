import random
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset

from utils.iotools import read_image


@dataclass
class HFTextMaskingConfig:
    mlm_probability: float = 0.15


def _build_mlm_inputs_and_labels(
    input_ids: torch.LongTensor,
    pad_token_id: int,
    special_token_ids: set,
    mask_token_id: Optional[int],
    rng: random.Random,
    cfg: HFTextMaskingConfig,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """BERT-style MLM: replace some tokens with mask_token_id (or unk/eos fallback).

    Returns:
      mlm_input_ids: same shape as input_ids
      mlm_labels: same shape; 0 for non-masked (to be compatible with this repo's compute_mlm)
    """
    mlm_input_ids = input_ids.clone()
    mlm_labels = torch.zeros_like(input_ids)

    if mask_token_id is None:
        # Fallback: keep input token id unchanged, but still supervise it.
        # This degrades into a token reconstruction loss; good enough for wiring validation.
        mask_token_id = pad_token_id

    candidate_positions = []
    for i, token_id in enumerate(input_ids.tolist()):
        if token_id == pad_token_id:
            continue
        if token_id in special_token_ids:
            continue
        candidate_positions.append(i)

    if not candidate_positions:
        return mlm_input_ids, mlm_labels

    masked_any = False
    for pos in candidate_positions:
        if rng.random() < cfg.mlm_probability:
            masked_any = True
            original = mlm_input_ids[pos].item()
            mlm_labels[pos] = original
            mlm_input_ids[pos] = mask_token_id

    if not masked_any:
        # Ensure at least 1 masked token
        pos = candidate_positions[0]
        original = mlm_input_ids[pos].item()
        mlm_labels[pos] = original
        mlm_input_ids[pos] = mask_token_id

    return mlm_input_ids, mlm_labels


def _ensure_eos_token(
    input_ids: torch.LongTensor,
    pad_token_id: int,
    eos_token_id: Optional[int],
) -> torch.LongTensor:
    """Ensure EOS exists at the end of the non-padding sequence.

    This repo uses fixed-length padding. Some tokenizers (including the shipped T5Gemma2 tokenizer)
    may not append EOS automatically for plain encoding, so we enforce it here.

    Strategy:
    - If eos_token_id is None: no-op.
    - If there is padding: write EOS into the first pad position.
    - If no padding (fully packed / truncated): overwrite the last token with EOS.
    """
    if eos_token_id is None:
        return input_ids
    eos_token_id = int(eos_token_id)
    if eos_token_id < 0:
        return input_ids
    if bool((input_ids == eos_token_id).any().item()):
        return input_ids

    # Find first pad position
    pad_pos = (input_ids == int(pad_token_id)).nonzero(as_tuple=False).view(-1)
    if pad_pos.numel() > 0:
        input_ids = input_ids.clone()
        input_ids[int(pad_pos[0].item())] = eos_token_id
        return input_ids

    # No padding: overwrite last token
    input_ids = input_ids.clone()
    input_ids[-1] = eos_token_id
    return input_ids


class ImageTextDatasetT5Gemma2(Dataset):
    """Train dataset for T5Gemma2 backbone.

    Returns tensors that are directly movable to GPU by the existing training loop.

    Keys:
      - pids: int
      - image_ids: int
      - pixel_values: FloatTensor[3,H,W]
      - input_ids: LongTensor[text_length]
      - mlm_input_ids / mlm_labels (optional, when MLM=True in build)
    """

    def __init__(
        self,
        dataset,
        processor,
        text_length: int = 77,
        mm_max_length: Optional[int] = None,
        truncate: bool = True,
        enable_mlm: bool = False,
        masking_cfg: Optional[HFTextMaskingConfig] = None,
        train_transforms=None,
        enable_gen: bool = False,
        gen_prompt: str = "Caption",
        gen_prompt_length: int = 32,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.processor = processor
        self.text_length = text_length
        # Multimodal inputs must be long enough to hold image placeholder tokens.
        # T5Gemma2 defaults to 256 tokens per image; keep a safe floor.
        self.mm_max_length = int(mm_max_length) if mm_max_length is not None else max(int(text_length), 512)
        self.truncate = truncate
        self.enable_mlm = enable_mlm
        self.enable_gen = enable_gen
        self.gen_prompt = str(gen_prompt)
        self.gen_prompt_length = int(gen_prompt_length)
        self.masking_cfg = masking_cfg or HFTextMaskingConfig()
        self.rng = random.Random(seed)

        self.train_transforms = train_transforms

        self.tokenizer = processor.tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        self.special_token_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        self.mask_token_id = getattr(self.tokenizer, "mask_token_id", None)

        if self.pad_token_id is None:
            # Very defensive; most encoder-decoder tokenizers define pad.
            self.pad_token_id = 0

        # Pre-tokenize generation prompt once to avoid repeated work.
        self._gen_prompt_ids = None
        if self.enable_gen:
            tok = self.tokenizer(
                self.gen_prompt,
                padding="max_length",
                truncation=True,
                max_length=self.gen_prompt_length,
                return_tensors="pt",
            )
            self._gen_prompt_ids = tok.input_ids.squeeze(0)
            self._gen_prompt_ids = _ensure_eos_token(
                self._gen_prompt_ids,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        image = read_image(img_path)

        # pixel_values
        if self.train_transforms is not None:
            pv = self.train_transforms(image)
        else:
            image_processor = getattr(self.processor, "image_processor", None)
            if image_processor is not None:
                pv = image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
            else:
                pv = self.processor(text="", images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        # text input ids
        tok = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.text_length,
            return_tensors="pt",
        )
        input_ids = tok.input_ids.squeeze(0)
        input_ids = _ensure_eos_token(input_ids, pad_token_id=self.pad_token_id, eos_token_id=self.eos_token_id)

        ret = {
            "pids": pid,
            "image_ids": image_id,
            # Also provide the HF-style key expected by PersonSearchT5Gemma2.
            "pixel_values": pv,
            "input_ids": input_ids,
        }

        if self.enable_gen and self._gen_prompt_ids is not None:
            # Model forward concatenates image tokens internally, so here we only provide the text-part prompt.
            ret["masked_input_ids"] = self._gen_prompt_ids.clone()

        if self.enable_mlm:
            # IMPORTANT: when using pixel_values in the model forward, T5Gemma2 expects
            # image placeholder tokens inside input_ids. So MLM must use processor(text+image)
            # rather than tokenizer(text-only).
            mm = self.processor(
                text="<start_of_image> " + caption,
                images=image,
                padding="max_length",
                truncation=True,
                max_length=self.mm_max_length,
                return_tensors="pt",
            )
            mm_input_ids = mm["input_ids"].squeeze(0)

            mlm_input_ids, mlm_labels = _build_mlm_inputs_and_labels(
                input_ids=mm_input_ids,
                pad_token_id=self.pad_token_id,
                special_token_ids=self.special_token_ids,
                mask_token_id=self.mask_token_id,
                rng=self.rng,
                cfg=self.masking_cfg,
            )
            ret.update({
                "mlm_input_ids": mlm_input_ids,
                "mlm_labels": mlm_labels,
            })

        return ret


class ImageDatasetT5Gemma2(Dataset):
    def __init__(self, image_pids, img_paths, processor, transforms=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.processor = processor
        self.transforms = transforms

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        image = read_image(img_path)
        if self.transforms is not None:
            pv = self.transforms(image)
        else:
            image_processor = getattr(self.processor, "image_processor", None)
            if image_processor is not None:
                pv = image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
            else:
                pv = self.processor(text="", images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return pid, pv


class TextDatasetT5Gemma2(Dataset):
    def __init__(self, caption_pids, captions, tokenizer, text_length: int = 77, truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = tokenizer
        self.pad_token_id = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]
        tok = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.text_length,
            return_tensors="pt",
        )
        input_ids = tok.input_ids.squeeze(0)
        input_ids = _ensure_eos_token(input_ids, pad_token_id=self.pad_token_id, eos_token_id=self.eos_token_id)
        return pid, input_ids
