"""Utilities for decoder-based coarse-to-fine reranking."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sequence_log_likelihood(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pad_token_id: int,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Mean token log probability for each target sequence."""

    if temperature <= 0:
        raise ValueError("generation temperature must be positive")
    log_prob = F.log_softmax(logits.float() / temperature, dim=-1)
    mask = labels.ne(pad_token_id)
    safe_labels = labels.masked_fill(~mask, 0)
    token_scores = log_prob.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_scores * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)


def minmax(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize the last dimension to [0, 1]."""

    low = values.amin(dim=-1, keepdim=True)
    high = values.amax(dim=-1, keepdim=True)
    span = high - low
    normalized = (values - low) / span.clamp_min(eps)
    return torch.where(span > eps, normalized, torch.zeros_like(normalized))


def fuse_scores(
    coarse_scores: torch.Tensor, generation_scores: torch.Tensor, alpha: float = 0.4
) -> torch.Tensor:
    """Min-max normalize and interpolate top-K scores (Equation 11)."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if coarse_scores.shape != generation_scores.shape:
        raise ValueError("coarse and generation scores must have the same shape")
    return alpha * minmax(coarse_scores) + (1.0 - alpha) * minmax(generation_scores)


def rerank_topk(
    coarse_scores: torch.Tensor,
    generation_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    alpha: float = 0.4,
) -> torch.Tensor:
    """Return a complete ranking with only the original top-K reordered."""

    fused = fuse_scores(coarse_scores.gather(1, topk_indices), generation_scores, alpha)
    top_order = fused.argsort(dim=1, descending=True)
    reordered_top = topk_indices.gather(1, top_order)

    full_order = coarse_scores.argsort(dim=1, descending=True)
    in_top = torch.zeros_like(coarse_scores, dtype=torch.bool)
    in_top.scatter_(1, topk_indices, True)
    tails = []
    for row, mask in zip(full_order, in_top, strict=True):
        tails.append(row[~mask[row]])
    return torch.cat([reordered_top, torch.stack(tails)], dim=1)
