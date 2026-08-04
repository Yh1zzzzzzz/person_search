from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gram.model import GRAM, inject_cached_image_features
from gram.rerank import fuse_scores, rerank_topk, sequence_log_likelihood


def test_sequence_log_likelihood_masks_padding_and_normalizes_length():
    logits = torch.zeros(2, 3, 3)
    logits[0, 0, 1] = 5
    logits[0, 1, 2] = 5
    logits[1, 0, 1] = 5
    logits[1, 1, 2] = 5
    labels = torch.tensor([[1, 2, 0], [1, 2, 0]])
    scores = sequence_log_likelihood(logits, labels, pad_token_id=0, temperature=1.0)
    assert torch.allclose(scores[0], scores[1])


def test_fusion_uses_both_normalized_signals():
    coarse = torch.tensor([[1.0, 2.0, 3.0]])
    generation = torch.tensor([[3.0, 2.0, 1.0]])
    assert torch.allclose(fuse_scores(coarse, generation, alpha=0.5), torch.full_like(coarse, 0.5))


def test_reranking_changes_only_topk():
    coarse = torch.tensor([[0.9, 0.8, 0.7, 0.6]])
    topk = torch.tensor([[0, 1]])
    generation = torch.tensor([[0.0, 1.0]])
    ranking = rerank_topk(coarse, generation, topk, alpha=0.0)
    assert ranking.tolist() == [[1, 0, 2, 3]]


def test_cached_image_features_replace_only_placeholders():
    input_ids = torch.tensor([[1, 9, 9, 2], [1, 9, 9, 2]])
    embeddings = torch.zeros(2, 4, 3)
    cached = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    result = inject_cached_image_features(input_ids, embeddings, cached, image_token_id=9)
    assert torch.equal(result[:, 1:3], cached)
    assert torch.count_nonzero(result[:, [0, 3]]) == 0

    with pytest.raises(ValueError, match="placeholder count"):
        inject_cached_image_features(input_ids, embeddings, cached[:, :1], image_token_id=9)


def test_cached_generation_path_does_not_call_vision_tower():
    class FailingVisionTower(nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("vision tower must not run during cached reranking")

    class FakeEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(image_token_id=9)
            self.embed_tokens = nn.Embedding(20, 4)
            self.vision_tower = FailingVisionTower()

        def forward(self, *, inputs_embeds, attention_mask, return_dict):
            return SimpleNamespace(last_hidden_state=inputs_embeds)

    class FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = FakeEncoder()

        def get_encoder(self):
            return self.encoder

        def prepare_decoder_input_ids_from_labels(self, labels):
            return labels.masked_fill(labels.eq(-100), 0)

        def forward(self, *, encoder_outputs, attention_mask, decoder_input_ids, **kwargs):
            batch, length = decoder_input_ids.shape
            return SimpleNamespace(logits=torch.zeros(batch, length, 20))

    model = GRAM.__new__(GRAM)
    nn.Module.__init__(model)
    model.backbone = FakeBackbone()
    scores = model.generation_scores_from_cache(
        image_features=torch.randn(2, 2, 4),
        input_ids=torch.tensor([[1, 9, 9, 2], [1, 9, 9, 2]]),
        attention_mask=torch.ones(2, 4, dtype=torch.long),
        labels=torch.tensor([[3, 4, 0], [5, 6, 0]]),
        pad_token_id=0,
    )
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()
