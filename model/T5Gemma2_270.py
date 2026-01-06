import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from transformers import AutoModelForSeq2SeqLM, PreTrainedModel

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

"""
图片通过enencoder的T5-Gemma2模型

"""
# =============================================================================
# Loss Functions
# =============================================================================
class Objectives:
    @staticmethod
    def compute_sdm(image_features, text_features, pid, logit_scale, epsilon=1e-8):
        # 1. 特征归一化 (L2 Normalize)
        # 传入前已经归一化，无需重复归一化
        # image_features = F.normalize(image_features, dim=1)
        # text_features = F.normalize(text_features, dim=1)

   
        logits = logit_scale * image_features @ text_features.t()

        # 3. 构建真实标签分布 Q (True Matching Distribution)
        # 处理 Multi-positive (同一 batch 内可能有相同 ID 的多张图或多段文本)
        pid = pid.view(-1, 1)
        # 生成 bool mask: 如果 pid 相同则为 True
        mask = torch.eq(pid, pid.t()).float() 
        # 归一化：将 binary mask 转换为概率分布 Q，使其行和为 1
        labels_distribute = mask / mask.sum(dim=1, keepdim=True)

        # 4. 计算 Image-to-Text Loss (I2T)
        # P_i2t: 模型预测的概率分布 (Softmax)
        # log_P_i2t: 模型预测的对数概率 (LogSoftmax)
        pred_i2t = F.softmax(logits, dim=1)
        log_pred_i2t = F.log_softmax(logits, dim=1)
        
        # KL Divergence: P * (log(P) - log(Q + eps))
        # epsilon 用于防止 log(0)
        loss_i2t = pred_i2t * (log_pred_i2t - torch.log(labels_distribute + epsilon))
        loss_i2t = torch.mean(torch.sum(loss_i2t, dim=1))

        # 5. 计算 Text-to-Image Loss (T2I)
        # 对称操作，转置 Logits 和 Labels
        logits_t2i = logits.t()
        labels_distribute_t2i = labels_distribute.t()
        
        pred_t2i = F.softmax(logits_t2i, dim=1)
        log_pred_t2i = F.log_softmax(logits_t2i, dim=1)
        
        loss_t2i = pred_t2i * (log_pred_t2i - torch.log(labels_distribute_t2i + epsilon))
        loss_t2i = torch.mean(torch.sum(loss_t2i, dim=1))

        # 6. 总 Loss
        return loss_i2t + loss_t2i

    @staticmethod
    def compute_id(image_logits, text_logits, pids):
        # 计算身份分类损失
        loss_img = F.cross_entropy(image_logits, pids)
        loss_txt = F.cross_entropy(text_logits, pids)
        return (loss_img + loss_txt) / 2

objectives = Objectives()

# =============================================================================
# Main Model Class
# =============================================================================
class PersonSearchT5Gemma2(PreTrainedModel):
    def __init__(
        self,
        config,
        hf_model_name_or_path: str,
        num_classes: int = 11003,
        feature_dim: int = 640,
        projector_hidden_dim: int = 2048,
        bnneck: bool = False,
        temperature: float = 0.02,
        gen_loss_weight: float = 1.0, 
        id_loss_weight: float = 1.0,
        image_size: int = 896,
        vision_intermidiate_size: int = 4304,
        attn_implementation: str = "sdpa",
        torch_dtype=None,
    ):
         
        super().__init__(config)
        self.num_classes = num_classes #CUHK-PEDES默认11003个id(训练集)
        self.feature_dim = feature_dim
        self.gen_loss_weight = gen_loss_weight
        self.id_loss_weight = float(id_loss_weight)
        self.use_bnneck = bool(bnneck)
        self.vision_intermidiate_size = int(vision_intermidiate_size)
        # 1. Load Backbone
        print(f"Loading backbone from: {hf_model_name_or_path} ...")
        self.attn_implementation = str(attn_implementation)
        load_kwargs = dict(
            trust_remote_code=True,
            local_files_only=True,
            config=config,
        )
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        try:
            self.backbone = AutoModelForSeq2SeqLM.from_pretrained(
                hf_model_name_or_path,
                attn_implementation=self.attn_implementation,
                **load_kwargs,
            )
            print(f"[T5Gemma2] attn_implementation={self.attn_implementation}")
        except TypeError:
            # Older transformers may not accept the arg at all.
            # Retry without optional load_kwargs that may not be supported.
            _load_kwargs = dict(load_kwargs)
            _load_kwargs.pop("torch_dtype", None)
            self.backbone = AutoModelForSeq2SeqLM.from_pretrained(hf_model_name_or_path, **_load_kwargs)
            print(
                f"[T5Gemma2] attn_implementation request '{self.attn_implementation}' was ignored "
                f"(transformers too old / unsupported). Using default attention implementation."
            )
        except ValueError as e:
            # Newer transformers may reject sdpa for some custom architectures.
            msg = str(e)
            if "scaled_dot_product_attention" in msg or "attn_implementation=\"eager\"" in msg:
                print(
                    f"[T5Gemma2] attn_implementation='{self.attn_implementation}' unsupported; "
                    "falling back to 'eager'."
                )
                self.attn_implementation = "eager"
                try:
                    self.backbone = AutoModelForSeq2SeqLM.from_pretrained(
                        hf_model_name_or_path,
                        attn_implementation=self.attn_implementation,
                        **load_kwargs,
                    )
                except TypeError:
                    _load_kwargs = dict(load_kwargs)
                    _load_kwargs.pop("torch_dtype", None)
                    self.backbone = AutoModelForSeq2SeqLM.from_pretrained(hf_model_name_or_path, **_load_kwargs)
            else:
                raise
        self.encoder = self.backbone.get_encoder()
        self.decoder = self.backbone.get_decoder() 
        self.lm_head = self.backbone.lm_head

        # 3. Config Parsing
        cfg = self.backbone.config
        self.hidden_size = None
        # 尝试获取 hidden_size
        if self.hidden_size is None and hasattr(cfg, "encoder"):
            enc = cfg.encoder
            if hasattr(enc, "text_config"):
                text_cfg = enc.text_config
                if hasattr(text_cfg, "hidden_size"): 
                    self.hidden_size = text_cfg.hidden_size

        # Token ids used for pooling/indexing (robust to different configs/tokenizers)
        self.pad_token_id = int(getattr(cfg, "pad_token_id", 0) or 0)
        eos_from_cfg = getattr(cfg, "eos_token_id", None)
        if eos_from_cfg is None and hasattr(cfg, "encoder") and hasattr(cfg.encoder, "text_config"):
            eos_from_cfg = getattr(cfg.encoder.text_config, "eos_token_id", None)
        self.eos_token_id = int(eos_from_cfg) if eos_from_cfg is not None else None


        def get_cfg_attr(attr_name, default_val):
            if hasattr(cfg, attr_name): 
                print(f">>> Found config attribute: {attr_name} = {getattr(cfg, attr_name)}")
                return getattr(cfg, attr_name)

            if hasattr(cfg, "encoder"):
                enc = cfg.encoder
                if hasattr(enc, attr_name): 
                    print(f">>> Found encoder attribute: {attr_name} = {getattr(enc, attr_name)}")
                    return getattr(enc, attr_name)
                if isinstance(enc, dict) and attr_name in enc: 
                    print(f">>> Found encoder dict attribute: {attr_name} = {enc[attr_name]}")
                    return enc[attr_name]
            return default_val

        self.boi_token_id = get_cfg_attr("boi_token_index", 255999)
        self.eoi_token_id = get_cfg_attr("eoi_token_index", 256000)
        self.image_token_id = get_cfg_attr("image_token_index", 256001)
        self.num_image_tokens = get_cfg_attr("mm_tokens_per_image", 256)

        self.expected_image_size = int(getattr(getattr(self.encoder.vision_tower, "config", None), "image_size", 896))

        # Enforce a fixed resolution. Image resizing/normalization should be done by the HF processor
        # before feeding pixel_values to the model.
        image_size = int(image_size)
        if image_size != self.expected_image_size:
            raise ValueError(
                f"pixel_values resolution mismatch: model expects image_size={self.expected_image_size} "
                f"but got configured image_size={image_size}. "
                "Please set processor.image_processor.size/crop_size to match the model, "
                "or re-initialize the model with the matching image_size."
            )
        
        print(f">>> Model Configured: Hidden={self.hidden_size} | Default Image Tokens={self.num_image_tokens}")


        # 视觉投影
        # NOTE: In this architecture, `encode_image_only()` pools the multimodal encoder
        # output over [IMG] tokens, whose hidden dim matches the text encoder hidden size
        # (e.g., 640), not the raw SigLIP vision tower hidden size (e.g., 1152).
        vision_hidden = int(self.hidden_size)
        if vision_hidden <= 0:
            raise ValueError(f"Invalid hidden_size for vision projection: {vision_hidden}")
        self.vision_proj = GeGLU(vision_hidden, vision_intermidiate_size, feature_dim)

        nn.init.xavier_uniform_(self.vision_proj.gate_proj.weight)
        nn.init.xavier_uniform_(self.vision_proj.in_proj.weight)
        nn.init.xavier_uniform_(self.vision_proj.out_proj.weight)
        nn.init.constant_(self.vision_proj.gate_proj.bias, 0)
        nn.init.constant_(self.vision_proj.in_proj.bias, 0)
        nn.init.constant_(self.vision_proj.out_proj.bias, 0)


        #文本投影头
        self.text_proj = GeGLU(self.hidden_size, projector_hidden_dim, feature_dim)

        nn.init.xavier_uniform_(self.text_proj.gate_proj.weight)
        nn.init.xavier_uniform_(self.text_proj.in_proj.weight)
        nn.init.xavier_uniform_(self.text_proj.out_proj.weight)
        nn.init.constant_(self.text_proj.gate_proj.bias, 0)
        nn.init.constant_(self.text_proj.in_proj.bias, 0)
        nn.init.constant_(self.text_proj.out_proj.bias, 0)


        self.logit_scale = nn.Parameter(torch.ones([]) * (1.0 / temperature))
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)

        # BNNeck for ID classification (separate stats for image/text)
        # Used ONLY for the ID branch to stabilize feature distribution.
        if self.use_bnneck:
            self.bn_i = nn.BatchNorm1d(feature_dim)
            self.bn_t = nn.BatchNorm1d(feature_dim)
            nn.init.constant_(self.bn_i.weight, 1.0)
            nn.init.constant_(self.bn_i.bias, 0.0)
            nn.init.constant_(self.bn_t.weight, 1.0)
            nn.init.constant_(self.bn_t.bias, 0.0)

        nn.init.xavier_uniform_(self.classifier.weight)
    
    def _mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.to(dtype=last_hidden_state.dtype).unsqueeze(-1)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def _mean_pool_image_tokens(self, last_hidden_state: torch.Tensor, num_image_tokens: int) -> torch.Tensor:
        """Mean-pool only the [IMG] tokens (exclude BOI/EOI).

        Expected input sequence: [BOI] + [IMG]*num_image_tokens + [EOI]
        last_hidden_state: [B, 2+num_image_tokens, H]
        """
        num_image_tokens = int(num_image_tokens)
        if num_image_tokens <= 0:
            raise ValueError(f"num_image_tokens must be positive, got {num_image_tokens}")
        if last_hidden_state.size(1) < (num_image_tokens + 2):
            raise ValueError(
                f"Unexpected sequence length={last_hidden_state.size(1)} for num_image_tokens={num_image_tokens}"
            )
        img_tokens = last_hidden_state[:, 1 : 1 + num_image_tokens, :]
        return img_tokens.mean(dim=1)

    def _validate_pixel_values_shape(self, pixel_values: torch.Tensor) -> None:
        if pixel_values is None:
            raise ValueError("pixel_values is required")
        if pixel_values.ndim != 4:
            raise ValueError(f"pixel_values must be 4D [B,3,H,W], got shape={tuple(pixel_values.shape)}")
        h = int(pixel_values.shape[-2])
        w = int(pixel_values.shape[-1])
        if h != self.expected_image_size or w != self.expected_image_size:
            raise ValueError(
                f"Method2 expects fixed image size {self.expected_image_size}x{self.expected_image_size}, "
                f"but got {h}x{w}. Please fix preprocessing (resize/pad)."
            )

    def construct_image_tokens(self, batch_size, device, num_image_tokens: Optional[int] = None):
        if num_image_tokens is None:
            num_image_tokens = self.num_image_tokens
        boi = torch.full((batch_size, 1), self.boi_token_id, dtype=torch.long, device=device)
        img = torch.full((batch_size, num_image_tokens), self.image_token_id, dtype=torch.long, device=device)
        eoi = torch.full((batch_size, 1), self.eoi_token_id, dtype=torch.long, device=device)
        return torch.cat([boi, img, eoi], dim=1)

    def encode_image_only(self, pixel_values):
        B = pixel_values.shape[0]
        device = pixel_values.device

        self._validate_pixel_values_shape(pixel_values)
        num_tokens = int(self.num_image_tokens)
        input_ids = self.construct_image_tokens(B, device, num_image_tokens=num_tokens)
        attention_mask = torch.ones_like(input_ids)
        
        enc_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values 
        )
        # Only pool the 256 [IMG] tokens (exclude BOI/EOI) as requested
        pooled = self._mean_pool_image_tokens(enc_out.last_hidden_state, num_tokens)
        pooled = pooled.to(dtype=self.vision_proj.gate_proj.weight.dtype)
        return self.vision_proj(pooled)

    def encode_text_only(self, input_ids):
        mask = (input_ids != int(self.pad_token_id)).long()
        enc_out = self.encoder(input_ids=input_ids, attention_mask=mask)
        pooled = self._mean_pool(enc_out.last_hidden_state, mask)
        pooled = pooled.to(dtype=self.text_proj.gate_proj.weight.dtype)
        return self.text_proj(pooled)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.encode_image_only(image)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        return self.encode_text_only(text)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ret = {}

        ret["temperature"] = (1.0 / self.logit_scale).detach()
        
        reuse_mm_encoder_for_image = bool(batch.get("reuse_mm_encoder_for_image", False))

        # 1) Generation (optionally first, so we can reuse encoder output for image features)
        mm_encoder_last_hidden_state = None
        if (
            self.training
            and float(getattr(self, "gen_loss_weight", 0.0)) > 0.0
            and "masked_input_ids" in batch
            and "pixel_values" in batch
        ):
            device = batch["pixel_values"].device
            B = batch["pixel_values"].shape[0]

            self._validate_pixel_values_shape(batch["pixel_values"])
            num_tokens = int(self.num_image_tokens)
            img_part_ids = self.construct_image_tokens(B, device, num_image_tokens=num_tokens)
            text_part_ids = batch["masked_input_ids"]

            multimodal_input_ids = torch.cat([img_part_ids, text_part_ids], dim=1)
            img_mask = torch.ones_like(img_part_ids)
            text_mask = (text_part_ids != int(self.pad_token_id)).long()
            multimodal_attention_mask = torch.cat([img_mask, text_mask], dim=1)

            labels = batch["input_ids"].clone()
            labels[labels == int(self.pad_token_id)] = -100

            outputs = self.backbone(
                input_ids=multimodal_input_ids,
                attention_mask=multimodal_attention_mask,
                pixel_values=batch["pixel_values"],
                labels=labels,
                return_dict=True,
            )
            ret["gen_loss"] = outputs.loss * self.gen_loss_weight

            if reuse_mm_encoder_for_image:
                mm_encoder_last_hidden_state = getattr(outputs, "encoder_last_hidden_state", None)

        # 2) Retrieval
        if "pixel_values" in batch:
            self._validate_pixel_values_shape(batch["pixel_values"])
            if mm_encoder_last_hidden_state is not None:
                # Pool [IMG] tokens from multimodal encoder output: [BOI] + [IMG]*N + [EOI] + prompt...
                num_tokens = int(self.num_image_tokens)
                pooled = self._mean_pool_image_tokens(mm_encoder_last_hidden_state, num_tokens)
                pooled = pooled.to(dtype=self.vision_proj.weight.dtype)
                v_feats = self.vision_proj(pooled)
            else:
                v_feats = self.encode_image_only(batch["pixel_values"])
            v_feats_norm = F.normalize(v_feats, dim=1)
            if self.training:
                if self.use_bnneck:
                    ret["v_id_logits"] = self.classifier(self.bn_i(v_feats))
                else:
                    ret["v_id_logits"] = self.classifier(v_feats)

        if "input_ids" in batch:
            t_feats = self.encode_text_only(batch["input_ids"])
            t_feats_norm = F.normalize(t_feats, dim=1)
            if self.training:
                if self.use_bnneck:
                    ret["t_id_logits"] = self.classifier(self.bn_t(t_feats))
                else:
                    ret["t_id_logits"] = self.classifier(t_feats)

        if self.training and "pixel_values" in batch and "input_ids" in batch:
            ret["sdm_loss"] = objectives.compute_sdm(v_feats_norm, t_feats_norm, batch["pids"], self.logit_scale)
            ret["id_loss"] = objectives.compute_id(ret["v_id_logits"], ret["t_id_logits"], batch["pids"]) * self.id_loss_weight

        # (Generation is handled above)

        loss_components = [v for k, v in ret.items() if "loss" in k]
        if loss_components:
            ret["total_loss"] = sum(loss_components)
        return ret


def build_person_search_t5gemma2(args, num_classes: int):
    from transformers import AutoConfig
    import torch

    hf_path = getattr(args, "hf_model_name_or_path", "T5_270M_Base")
    config = AutoConfig.from_pretrained(hf_path, local_files_only=True)

    config.drop_path_rate = 0.1
    if hasattr(config, "vision_config"):
        config.vision_config.drop_path_rate = 0.1
    if hasattr(config, "text_config"):
        config.text_config.drop_path_rate = 0.1

    model = PersonSearchT5Gemma2(
        config=config,
        hf_model_name_or_path=hf_path,
        num_classes=int(num_classes),
        feature_dim=int(getattr(args, "feature_dim", 640)),
        projector_hidden_dim=int(getattr(args, "projector_hidden_dim", 2048)),
        bnneck=bool(getattr(args, "bnneck", False)),
        temperature=float(getattr(args, "temperature", 0.02)),
        gen_loss_weight=float(getattr(args, "gen_loss_weight", 1.0)),
        id_loss_weight=float(getattr(args, "id_loss_weight", 1.0)),
        image_size=int(getattr(args, "t5_image_size", 896)),
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

    # VRAM optimizations
    # - use_cache stores KV caches; not needed for teacher-forcing training and increases memory.
    if hasattr(model.backbone, "config") and hasattr(model.backbone.config, "use_cache"):
        model.backbone.config.use_cache = False
    if hasattr(model.backbone, "generation_config") and hasattr(model.backbone.generation_config, "use_cache"):
        model.backbone.generation_config.use_cache = False
    if bool(getattr(args, "gradient_checkpointing", False)):
        if hasattr(model.backbone, "gradient_checkpointing_enable"):
            model.backbone.gradient_checkpointing_enable()
    return model
    
# =============================================================================
if __name__ == "__main__":
    from transformers import AutoConfig
    local_path = "T5_270M_Base"
    config = AutoConfig.from_pretrained(local_path)
    print(config)
    model = PersonSearchT5Gemma2(
        config=config,
        hf_model_name_or_path=local_path,
    )
    print("Model initialized successfully.")
