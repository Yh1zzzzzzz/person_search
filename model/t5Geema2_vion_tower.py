import torch
import torch.nn as nn

from transformers import AutoConfig

from .T5Gemma2_270 import PersonSearchT5Gemma2
"""
图片只通过 vision tower的T5 Gemma2
"""

class PersonSearchT5Gemma2VisionTower(PersonSearchT5Gemma2):
    def __init__(
        self,
        config,
        hf_model_name_or_path: str,
        num_classes: int = 11003,
        feature_dim: int = 512,
        projector_hidden_dim: int = 1024,
        temperature: float = 0.02,
        gen_loss_weight: float = 1.0,
        id_loss_weight: float = 1.0,
        image_size: int = 448,
        attn_implementation: str = "sdpa",
    ):
        super().__init__(
            config=config,
            hf_model_name_or_path=hf_model_name_or_path,
            num_classes=num_classes,
            feature_dim=feature_dim,
            temperature=temperature,
            gen_loss_weight=gen_loss_weight,
            id_loss_weight=id_loss_weight,
            image_size=image_size,
            attn_implementation=attn_implementation,
        )

        # SigLIP vision hidden size (e.g., 1152) -> retrieval feature_dim
        vision_cfg = getattr(getattr(self.encoder, "vision_tower", None), "config", None)
        vision_hidden = int(getattr(vision_cfg, "hidden_size", 0) or 0)
        if vision_hidden <= 0:
            # fallback: best-effort; if this fails, we'll error early with a clear message.
            fallback_hidden = getattr(
                getattr(getattr(config, "encoder", None), "vision_config", None),
                "hidden_size",
                0,
            )
            vision_hidden = int(fallback_hidden or 0)
        if vision_hidden <= 0:
            raise ValueError("Unable to infer SigLIP vision hidden_size from config")

        hidden_dim = int(projector_hidden_dim)
        if hidden_dim <= 0:
            raise ValueError(f"projector_hidden_dim must be positive, got {hidden_dim}")

        # Projector Linear: Linear(input_dim, feature_dim)
        self.vision_tower_align = nn.Linear(vision_hidden, int(feature_dim), bias=False)
        nn.init.xavier_uniform_(self.vision_tower_align.weight)

    def encode_image_only(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self._validate_pixel_values_shape(pixel_values)

        # SigLIP vision tower: returns [B, num_patches, vision_hidden]
        vision_out = self.encoder.vision_tower(pixel_values=pixel_values)

        last_hidden = getattr(vision_out, "last_hidden_state", None)
        if last_hidden is None:
            raise ValueError("vision_tower output has no last_hidden_state")

        # Mean-pool over patch tokens
        pooled = last_hidden.mean(dim=1)

        pooled = pooled.to(dtype=self.vision_tower_align.weight.dtype)
        return self.vision_tower_align(pooled)


def build_person_search_t5gemma2_vion_tower(args, num_classes: int):
    """Factory used by model/build.py when --backbone=t5gemma2_vion_tower."""

    hf_path = getattr(args, "hf_model_name_or_path", "T5_270M_Base")
    config = AutoConfig.from_pretrained(hf_path, local_files_only=True)
    # config._attn_implementation = "sdpa"
    model = PersonSearchT5Gemma2VisionTower(
        config=config,
        hf_model_name_or_path=hf_path,
        num_classes=int(num_classes),
        feature_dim=int(getattr(args, "feature_dim", 1024)),
        projector_hidden_dim=int(getattr(args, "projector_hidden_dim", 2048)),
        temperature=float(getattr(args, "temperature", 0.02)),
        gen_loss_weight=float(getattr(args, "gen_loss_weight", 1.0)),
        id_loss_weight=float(getattr(args, "id_loss_weight", 1.0)),
        image_size=int(getattr(args, "t5_image_size", 448)),
        attn_implementation=str(getattr(args, "attn_implementation", "sdpa")),
    )

    # Keep the same training-time memory settings as the main t5gemma2 model.
    if hasattr(model.backbone, "config") and hasattr(model.backbone.config, "use_cache"):
        model.backbone.config.use_cache = False
    if hasattr(model.backbone, "generation_config") and hasattr(model.backbone.generation_config, "use_cache"):
        model.backbone.generation_config.use_cache = False
    if bool(getattr(args, "gradient_checkpointing", False)):
        if hasattr(model.backbone, "gradient_checkpointing_enable"):
            model.backbone.gradient_checkpointing_enable()

    return model
