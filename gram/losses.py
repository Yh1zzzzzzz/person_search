"""Losses from the GRAM paper."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalized_similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = F.normalize(left.float(), dim=-1)
    right = F.normalize(right.float(), dim=-1)
    return left @ right.transpose(0, 1)


def sdm_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    person_ids: torch.Tensor,
    temperature: torch.Tensor | float = 0.02,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Bidirectional similarity distribution matching (Equations 1-3)."""

    similarity = _normalized_similarity(image_features, text_features)
    scale = torch.as_tensor(temperature, device=similarity.device, dtype=similarity.dtype)
    logits_i2t = similarity / scale.clamp_min(eps)
    logits_t2i = logits_i2t.transpose(0, 1)

    ids = person_ids.reshape(-1)
    positives = ids[:, None].eq(ids[None, :]).float()
    targets = positives / positives.sum(dim=1, keepdim=True).clamp_min(1.0)

    def directional_kl(logits: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=1)
        prob = log_prob.exp()
        return (prob * (log_prob - torch.log(targets + eps))).sum(dim=1).mean()

    return directional_kl(logits_i2t) + directional_kl(logits_t2i)


def identity_loss(
    image_logits: torch.Tensor, text_logits: torch.Tensor, person_ids: torch.Tensor
) -> torch.Tensor:
    """Shared identity classification constraint (Equation 4)."""

    return 0.5 * (
        F.cross_entropy(image_logits.float(), person_ids)
        + F.cross_entropy(text_logits.float(), person_ids)
    )


def mst_loss(student_text: torch.Tensor, teacher_text: torch.Tensor) -> torch.Tensor:
    """Micro semantic transmission (Equation 7)."""

    return 1.0 - F.cosine_similarity(student_text.float(), teacher_text.float(), dim=-1).mean()


def mdm_loss(
    student_text: torch.Tensor,
    teacher_text: torch.Tensor,
    image_features: torch.Tensor,
    logit_scale: torch.Tensor | float,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Macro distribution matching (Equation 8)."""

    if temperature <= 0:
        raise ValueError("distillation temperature must be positive")
    scale = torch.as_tensor(logit_scale, device=image_features.device, dtype=torch.float32)
    images = F.normalize(image_features.float(), dim=-1)
    student_logits = scale * (F.normalize(student_text.float(), dim=-1) @ images.T)
    teacher_logits = scale * (F.normalize(teacher_text.float(), dim=-1) @ images.T)
    teacher_prob = F.softmax(teacher_logits.detach() / temperature, dim=-1)
    student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    return (temperature**2) * F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
