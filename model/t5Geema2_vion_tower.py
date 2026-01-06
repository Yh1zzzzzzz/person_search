import torch
import torch.nn as nn

from transformers import AutoConfig

from .T5Gemma2_270 import PersonSearchT5Gemma2
"""
图片只通过 vision tower的T5 Gemma2
"""

class GeGLU(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.1):
        super().__init__()
        self.gate_proj = nn.Linear(in_features, hidden_features)
        self.in_proj = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.out_proj(self.dropout(self.act(self.gate_proj(x)) * self.in_proj(x)))

class PersonSearchT5Gemma2VisionTower(PersonSearchT5Gemma2):
    def __init__(
        self,
        config,
        hf_model_name_or_path: str,
        num_classes: int = 11003,
        feature_dim: int = 640,
        projector_hidden_dim: int = 2048,
        temperature: float = 0.02,
        gen_loss_weight: float = 1.0,
        id_loss_weight: float = 1.0,
        image_size: int = 896,
        attn_implementation: str = "sdpa",
        torch_dtype=None,
    ):
        super().__init__(
            config=config,
            hf_model_name_or_path=hf_model_name_or_path,
            num_classes=num_classes,
            feature_dim=int(feature_dim),
            temperature=temperature,
            gen_loss_weight=gen_loss_weight,
            id_loss_weight=id_loss_weight,
            image_size=image_size,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )

        vision_cfg = getattr(getattr(self.encoder, "vision_tower", None), "config", None)
        vision_hidden = int(getattr(vision_cfg, "hidden_size", 0) or 0)
        if vision_hidden <= 0:
            raise ValueError("Unable to infer SigLIP vision hidden_size from config")

        hidden_dim = int(projector_hidden_dim)
        if hidden_dim <= 0:
            raise ValueError(f"projector_hidden_dim must be positive, got {hidden_dim}")

        # Projector Linear: GeGLU(input_dim, hidden_dim, feature_dim)
        self.vision_tower_align = GeGLU(vision_hidden, self.vision_intermidiate_size, feature_dim, dropout=0.1)
        nn.init.xavier_uniform_(self.vision_tower_align.gate_proj.weight)
        nn.init.xavier_uniform_(self.vision_tower_align.in_proj.weight)
        nn.init.xavier_uniform_(self.vision_tower_align.out_proj.weight)
        nn.init.constant_(self.vision_tower_align.gate_proj.bias, 0)
        nn.init.constant_(self.vision_tower_align.in_proj.bias, 0)
        nn.init.constant_(self.vision_tower_align.out_proj.bias, 0)

        
        if hasattr(self, "vision_proj"):
            del self.vision_proj

    def encode_image_only(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self._validate_pixel_values_shape(pixel_values)

        # SigLIP vision tower: returns [B, num_patches, vision_hidden]
        vision_out = self.encoder.vision_tower(pixel_values=pixel_values)

        last_hidden = getattr(vision_out, "last_hidden_state", None)
        if last_hidden is None:
            raise ValueError("vision_tower output has no last_hidden_state")

        # Mean-pool over patch tokens
        pooled = last_hidden.mean(dim=1)

        pooled = pooled.to(dtype=self.vision_tower_align.gate_proj.weight.dtype)
        return self.vision_tower_align(pooled)


def build_person_search_t5gemma2_vion_tower(args, num_classes: int):

    hf_path = getattr(args, "hf_model_name_or_path", "T5_270M_Base")
    config = AutoConfig.from_pretrained(hf_path, local_files_only=True)
    config.drop_path_rate = 0.1
    if hasattr(config, "vision_config"):
        config.vision_config.drop_path_rate = 0.1
    if hasattr(config, "text_config"):
        config.text_config.drop_path_rate = 0.1

    model = PersonSearchT5Gemma2VisionTower(
        config=config,
        hf_model_name_or_path=hf_path,
        num_classes=int(num_classes),
        feature_dim=int(getattr(args, "feature_dim", 640)),
        projector_hidden_dim=int(getattr(args, "projector_hidden_dim", 2048)),
        temperature=float(getattr(args, "temperature", 0.02)),
        gen_loss_weight=float(getattr(args, "gen_loss_weight", 1.0)),
        id_loss_weight=float(getattr(args, "id_loss_weight", 1.0)),
        attn_implementation=str(getattr(args, "attn_implementation", "sdpa")),
        torch_dtype=(
            torch.bfloat16
            if bool(getattr(args, "amp", False)) and str(getattr(args, "amp_dtype", "bf16")).lower() == "bf16"
            else (
                torch.float16
                if bool(getattr(args, "amp", False)) and str(getattr(args, "amp_dtype", "bf16")).lower() == "fp16"
                else None
            )
        ),
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
