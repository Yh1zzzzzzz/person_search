import torch

from gram.losses import identity_loss, mdm_loss, mst_loss, sdm_loss


def test_sdm_prefers_aligned_features():
    image = torch.eye(4)
    text = image.clone()
    ids = torch.arange(4)
    aligned = sdm_loss(image, text, ids, temperature=0.1)
    shuffled = sdm_loss(image, text.flip(0), ids, temperature=0.1)
    assert aligned < shuffled


def test_sdm_accepts_multiple_positive_captions():
    image = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    text = image.clone()
    value = sdm_loss(image, text, torch.tensor([3, 3, 8]), temperature=0.1)
    assert torch.isfinite(value)


def test_identity_loss_is_symmetric():
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    labels = torch.tensor([0, 1])
    expected = torch.nn.functional.cross_entropy(logits, labels)
    assert torch.allclose(identity_loss(logits, logits, labels), expected)


def test_distillation_losses_are_zero_for_identical_features():
    text = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    images = text.clone()
    assert torch.allclose(mst_loss(text, text), torch.tensor(0.0))
    assert torch.allclose(mdm_loss(text, text, images, 10.0), torch.tensor(0.0), atol=1e-6)
