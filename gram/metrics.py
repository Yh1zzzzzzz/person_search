"""Text-to-image retrieval metrics."""

from __future__ import annotations

import torch


def retrieval_metrics(
    query_ids: torch.Tensor,
    gallery_ids: torch.Tensor,
    *,
    scores: torch.Tensor | None = None,
    ranking: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute R@1/5/10, mAP, and mINP in percentages."""

    if (scores is None) == (ranking is None):
        raise ValueError("provide exactly one of scores or ranking")
    if ranking is None:
        ranking = scores.argsort(dim=1, descending=True)

    matches = gallery_ids[ranking].eq(query_ids[:, None])
    relevant = matches.sum(dim=1)
    if (relevant == 0).any():
        bad = int((relevant == 0).sum())
        raise ValueError(f"{bad} queries have no matching gallery identity")

    cumulative = matches.cumsum(dim=1)
    positions = torch.arange(1, matches.shape[1] + 1, device=matches.device).float()
    precision = cumulative.float() / positions[None, :]
    average_precision = (precision * matches).sum(dim=1) / relevant

    last_match = matches.shape[1] - 1 - matches.flip(1).float().argmax(dim=1)
    inp = relevant.float() / (last_match.float() + 1.0)

    result: dict[str, float] = {}
    for k in (1, 5, 10):
        clipped = min(k, matches.shape[1])
        result[f"R@{k}"] = 100.0 * matches[:, :clipped].any(dim=1).float().mean().item()
    result["mAP"] = 100.0 * average_precision.mean().item()
    result["mINP"] = 100.0 * inp.mean().item()
    return result
