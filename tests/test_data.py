import torch

from gram.data import GalleryCollator, Stage1Collator


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        batch = len(texts)
        return {
            "input_ids": torch.ones(batch, 2, dtype=torch.long),
            "attention_mask": torch.ones(batch, 2, dtype=torch.long),
        }


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()

    def __call__(self, *, images, text, **kwargs):
        assert images == [["image-a"], ["image-b"]]
        assert len(text) == len(images)
        return {
            "pixel_values": torch.zeros(2, 3, 4, 4),
            "input_ids": torch.ones(2, 3, dtype=torch.long),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
        }


def test_stage1_collator_batches_one_image_per_prompt():
    batch = Stage1Collator(FakeProcessor(), "Describe:", 16)(
        [("image-a", 0, "caption a"), ("image-b", 1, "caption b")]
    )
    assert batch["pixel_values"].shape[0] == 2
    assert batch["mm_input_ids"].shape[0] == 2


def test_gallery_collator_batches_one_image_per_prompt():
    batch = GalleryCollator(FakeProcessor(), "Describe:")(
        [("image-a", 0, "a.png"), ("image-b", 1, "b.png")]
    )
    assert batch["pixel_values"].shape[0] == 2
    assert batch["mm_input_ids"].shape[0] == 2
