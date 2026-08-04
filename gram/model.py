"""Minimal T5Gemma2 implementation of GRAM."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .losses import identity_loss, mdm_loss, mst_loss, sdm_loss
from .rerank import sequence_log_likelihood


def inject_cached_image_features(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    image_features: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """Replace image-placeholder embeddings with cached visual tokens."""

    placeholder = input_ids.eq(image_token_id)
    expected = placeholder.sum(dim=1)
    actual = image_features.shape[1]
    if not bool(expected.eq(actual).all()):
        raise ValueError(
            f"image placeholder count {expected.tolist()} does not match cached tokens {actual}"
        )
    mask = placeholder.unsqueeze(-1).expand_as(inputs_embeds)
    cached = image_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    return inputs_embeds.masked_scatter(mask, cached)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        target_dtype = self.layers[0].weight.dtype
        return self.layers(features.to(target_dtype))


class GRAM(nn.Module):
    """Shared vision/text encoders plus the T5Gemma2 caption decoder."""

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        feature_dim: int = 512,
        projection_hidden_dim: int = 2048,
        retrieval_temperature: float = 0.02,
        attention_implementation: str = "sdpa",
        dtype: str = "bf16",
    ):
        super().__init__()
        if retrieval_temperature <= 0:
            raise ValueError("retrieval_temperature must be positive")

        from transformers import AutoModelForSeq2SeqLM

        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        load_kwargs: dict[str, Any] = {
            "attn_implementation": attention_implementation,
            "dtype": dtype_map[dtype],
        }
        try:
            self.backbone = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_kwargs)
        except TypeError:
            load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
            self.backbone = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_kwargs)

        encoder_config = self.backbone.config.encoder
        vision_dim = int(encoder_config.vision_config.hidden_size)
        text_dim = int(encoder_config.text_config.hidden_size)
        self.vision_projection = ProjectionHead(vision_dim, projection_hidden_dim, feature_dim)
        self.text_projection = ProjectionHead(text_dim, projection_hidden_dim, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / retrieval_temperature)))

        self.model_name = model_name
        self.feature_dim = feature_dim
        self.retrieval_temperature = retrieval_temperature
        self.num_classes = num_classes

    @property
    def encoder(self) -> nn.Module:
        return self.backbone.get_encoder()

    @property
    def temperature(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0).reciprocal()

    @staticmethod
    def _mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Mean-pool SigLIP patch features and project to the retrieval space."""

        vision_output = self.encoder.vision_tower(pixel_values=pixel_values, return_dict=True)
        patches = vision_output.last_hidden_state
        return self.vision_projection(patches.mean(dim=1))

    def encode_image_with_cache(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode once for retrieval and cache decoder-ready visual tokens.

        The second tensor is the output of T5Gemma2's multimodal projector. It
        can be inserted into the encoder prompt later without calling the
        vision tower again.
        """

        vision_output = self.encoder.vision_tower(pixel_values=pixel_values, return_dict=True)
        patches = vision_output.last_hidden_state
        retrieval_features = self.vision_projection(patches.mean(dim=1))
        cached_image_features = self.encoder.multi_modal_projector(patches)
        return retrieval_features, cached_image_features

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return self.text_projection(self._mean_pool(output.last_hidden_state, attention_mask))

    def caption_loss(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        pad_token_id: int,
    ) -> torch.Tensor:
        targets = labels.masked_fill(labels.eq(pad_token_id), -100)
        output = self.backbone(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=targets,
            use_cache=False,
            return_dict=True,
        )
        return output.loss

    def generation_scores(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        pad_token_id: int,
        temperature: float = 2.0,
    ) -> torch.Tensor:
        """Compute s_gen: mean conditional query log likelihood (Equation 9)."""

        targets = labels.masked_fill(labels.eq(pad_token_id), -100)
        decoder_input_ids = self.backbone.prepare_decoder_input_ids_from_labels(targets)
        output = self.backbone(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            return_dict=True,
        )
        return sequence_log_likelihood(output.logits, labels, pad_token_id, temperature)

    def generation_scores_from_cache(
        self,
        image_features: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        pad_token_id: int,
        temperature: float = 2.0,
    ) -> torch.Tensor:
        """Compute generation scores without running the vision tower.

        ``image_features`` must come from :meth:`encode_image_with_cache`.
        Only the multimodal text encoder and decoder run in this path.
        """

        encoder = self.encoder
        inputs_embeds = encoder.embed_tokens(input_ids)
        image_token_id = int(encoder.config.image_token_id)
        inputs_embeds = inject_cached_image_features(
            input_ids, inputs_embeds, image_features, image_token_id
        )
        encoder_outputs = encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )

        targets = labels.masked_fill(labels.eq(pad_token_id), -100)
        decoder_input_ids = self.backbone.prepare_decoder_input_ids_from_labels(targets)
        output = self.backbone(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            return_dict=True,
        )
        return sequence_log_likelihood(output.logits, labels, pad_token_id, temperature)

    def stage1_loss(
        self,
        batch: dict[str, torch.Tensor],
        pad_token_id: int,
        lambda_icc: float = 1.0,
        lambda_lm: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        image_features = self.encode_image(batch["pixel_values"])
        text_features = self.encode_text(batch["query_input_ids"], batch["query_attention_mask"])
        sdm = sdm_loss(image_features, text_features, batch["person_ids"], self.temperature)
        icc = identity_loss(
            self.classifier(image_features),
            self.classifier(text_features),
            batch["person_ids"],
        )
        lm = self.caption_loss(
            batch["pixel_values"],
            batch["mm_input_ids"],
            batch["mm_attention_mask"],
            batch["labels"],
            pad_token_id,
        )
        total = sdm + lambda_icc * icc + lambda_lm * lm
        return {"loss": total, "sdm": sdm, "icc": icc, "lm": lm}

    def freeze_for_distillation(self) -> None:
        """Freeze vision/decoder paths and expose only text encoder + projection."""

        for parameter in self.parameters():
            parameter.requires_grad = False
        for name, parameter in self.encoder.named_parameters():
            is_visual = name.startswith(("vision_tower.", "multi_modal_projector."))
            if not is_visual:
                parameter.requires_grad = True
        for parameter in self.text_projection.parameters():
            parameter.requires_grad = True

    def stage2_loss(
        self,
        teacher: GRAM,
        batch: dict[str, Any],
        lambda_mst: float = 1.0,
        lambda_mdm: float = 1.0,
        distillation_temperature: float = 2.0,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            image_features = self.encode_image(batch["pixel_values"])
            teacher_text = teacher.encode_text(
                batch["english"]["input_ids"], batch["english"]["attention_mask"]
            )

        totals = []
        sdm_terms = []
        mst_terms = []
        mdm_terms = []
        for target in batch["targets"].values():
            student_text = self.encode_text(target["input_ids"], target["attention_mask"])
            sdm = sdm_loss(image_features, student_text, batch["person_ids"], self.temperature)
            mst = mst_loss(student_text, teacher_text)
            mdm = mdm_loss(
                student_text,
                teacher_text,
                image_features,
                self.logit_scale.exp().clamp(max=100.0),
                distillation_temperature,
            )
            totals.append(sdm + lambda_mst * mst + lambda_mdm * mdm)
            sdm_terms.append(sdm)
            mst_terms.append(mst)
            mdm_terms.append(mdm)

        return {
            "loss": torch.stack(totals).mean(),
            "sdm": torch.stack(sdm_terms).mean(),
            "mst": torch.stack(mst_terms).mean(),
            "mdm": torch.stack(mdm_terms).mean(),
        }
