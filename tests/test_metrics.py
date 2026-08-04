import torch

from gram.metrics import retrieval_metrics


def test_perfect_ranking_metrics():
    ids = torch.tensor([0, 1, 2])
    scores = torch.eye(3)
    result = retrieval_metrics(ids, ids, scores=scores)
    assert result == {"R@1": 100.0, "R@5": 100.0, "R@10": 100.0, "mAP": 100.0, "mINP": 100.0}


def test_metrics_accept_explicit_ranking():
    query_ids = torch.tensor([0])
    gallery_ids = torch.tensor([1, 0, 0])
    ranking = torch.tensor([[0, 1, 2]])
    result = retrieval_metrics(query_ids, gallery_ids, ranking=ranking)
    assert result["R@1"] == 0.0
    assert result["R@5"] == 100.0
    assert 0.0 < result["mAP"] < 100.0
