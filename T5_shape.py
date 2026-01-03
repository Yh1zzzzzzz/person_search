T5Gemma2ForConditionalGeneration(
  (model): T5Gemma2Model(
    (encoder): T5Gemma2Encoder(
      (vision_tower): SiglipVisionModel(
        (vision_model): SiglipVisionTransformer(
          (embeddings): SiglipVisionEmbeddings(
            (patch_embedding): Conv2d(3, 1152, kernel_size=(14, 14), stride=(14, 14), padding=valid)
            (position_embedding): Embedding(4096, 1152)
          )
          (encoder): SiglipEncoder(
            (layers): ModuleList(
              (0-26): 27 x SiglipEncoderLayer(
                (layer_norm1): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
                (self_attn): SiglipAttention(
                  (k_proj): Linear(in_features=1152, out_features=1152, bias=True)
                  (v_proj): Linear(in_features=1152, out_features=1152, bias=True)
                  (q_proj): Linear(in_features=1152, out_features=1152, bias=True)
                  (out_proj): Linear(in_features=1152, out_features=1152, bias=True)
                )
                (layer_norm2): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
                (mlp): SiglipMLP(
                  (activation_fn): GELUTanh()
                  (fc1): Linear(in_features=1152, out_features=4304, bias=True)
                  (fc2): Linear(in_features=4304, out_features=1152, bias=True)
                )
              )
            )
          )
          (post_layernorm): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
        )
      )
      (multi_modal_projector): T5Gemma2MultiModalProjector(
        (mm_soft_emb_norm): T5Gemma2RMSNorm((1152,), eps=1e-06)
        (avg_pool): AvgPool2d(kernel_size=4, stride=4, padding=0)
      )
      (embed_tokens): T5Gemma2TextScaledWordEmbedding(262144, 640)
      (norm): T5Gemma2RMSNorm((640,), eps=1e-06)
      (layers): ModuleList(
        (0-17): 18 x T5Gemma2EncoderLayer(
          (self_attn): T5Gemma2SelfAttention(
            (q_proj): Linear(in_features=640, out_features=1024, bias=False)
            (k_proj): Linear(in_features=640, out_features=256, bias=False)
            (v_proj): Linear(in_features=640, out_features=256, bias=False)
            (o_proj): Linear(in_features=1024, out_features=640, bias=False)
            (q_norm): T5Gemma2RMSNorm((256,), eps=1e-06)
            (k_norm): T5Gemma2RMSNorm((256,), eps=1e-06)
          )
          (pre_self_attn_layernorm): T5Gemma2RMSNorm((640,), eps=1e-06)
          (post_self_attn_layernorm): T5Gemma2RMSNorm((640,), eps=1e-06)
          (mlp): T5Gemma2MLP(
            (gate_proj): Linear(in_features=640, out_features=2048, bias=False)
            (up_proj): Linear(in_features=640, out_features=2048, bias=False)
            (down_proj): Linear(in_features=2048, out_features=640, bias=False)
            (act_fn): GELUTanh()
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (pre_feedforward_layernorm): T5Gemma2RMSNorm((640,), eps=1e-06)
          (post_feedforward_layernorm): T5Gemma2RMSNorm((640,), eps=1e-06)
          (dropout): Dropout(p=0.0, inplace=False)
        )
      )
      (dropout): Dropout(p=0.0, inplace=False)
      (rotary_emb): T5Gemma2RotaryEmbedding()
    )
    (decoder): T5Gemma2Decoder(
      (embed_tokens): T5Gemma2TextScaledWordEmbedding(262144, 640, padding_idx=0)
      (norm): T5Gemma2RMSNorm((640,), eps=1e-06)
      (layers): ModuleList(
        (0-17): 18 x T5Gemma2DecoderLayer(
          (self_attn): T5Gemma2MergedAttention(
            (q_proj): Linear(in_features=640, out_features=1024, bias=False)
            (k_proj): Linear(in_features=640, out_features=256, bias=False)
            (v_proj): Linear(in_features=640, out_features=256, bias=False)
            (o_proj): Linear(in_features=1024, out_features=640, bias=False)
            (q_norm): T5Gemma2RMSNorm((256,), eps=1e-06)
            (k_norm): T5Gemma2RMSNorm((256,), eps=1e-06)
          )
          (pre_self_attn_layernorm): T5Gemma2RMSNorm((640,), eps=1e-06)
          (post_self_attn_layernorm): T5Gemma2RMSNorm((640,), eps=1e-06)
          (mlp): T5Gemma2MLP(
            (gate_proj): Linear(in_features=640, out_features=2048, bias=False)
            (up_proj): Linear(in_features=640, out_features=2048, bias=False)
            (down_proj): Linear(in_features=2048, out_features=640, bias=False)
            (act_fn): GELUTanh()
            (dropout): Dropout(p=0.0, inplace=False)
          )
          (pre_feedforward_layernorm): T5Gemma2RMSNorm((640,), eps=1e-06)
          (post_feedforward_layernorm): T5Gemma2RMSNorm((640,), eps=1e-06)
          (dropout): Dropout(p=0.0, inplace=False)
        )
      )
      (dropout): Dropout(p=0.0, inplace=False)
      (rotary_emb): T5Gemma2RotaryEmbedding()
    )
  )
  (lm_head): T5Gemma2LMHead(
    (out_proj): Linear(in_features=640, out_features=262144, bias=False)
  )
)
# Config 属性: T5Gemma2Config {
# "architectures": [
# "T5Gemma2ForConditionalGeneration"
# ],
# "attention_dropout": 0.0,
# "bos_token_id": 2,
# "classifier_dropout_rate": 0.0,
# "decoder": {
# "_sliding_window_pattern": 6,
# "attention_bias": false,
# "attention_dropout": 0.0,
# "attn_logit_softcapping": null,
# "dropout_rate": 0.0,
# "dtype": "bfloat16",
# "final_logit_softcapping": null,
# "head_dim": 256,
# "hidden_activation": "gelu_pytorch_tanh",
# "hidden_size": 640,
# "initializer_range": 0.02,
# "intermediate_size": 2048,
# "layer_types": [
# "sliding_attention",
# "sliding_attention",
# "sliding_attention",
# "sliding_attention",
# "sliding_attention",
# "full_attention",
# "sliding_attention",
# "sliding_attention",
# "sliding_attention",
# "sliding_attention",
# "sliding_attention",
# "full_attention",
# "sliding_attention",
# "sliding_attention",
# "sliding_attention",
# "sliding_attention",
# "sliding_attention",
# "full_attention"
# ],
# "max_position_embeddings": 32768,
# "model_type": "t5gemma2_decoder",
# "num_attention_heads": 4,
# "num_hidden_layers": 18,
# "num_key_value_heads": 1,
# "query_pre_attn_scalar": 256,
# "rms_norm_eps": 1e-06,
# "rope_parameters": {
# "full_attention": {
# "factor": 8.0,
# "rope_theta": 1000000,
# "rope_type": "linear"
# },
# "sliding_attention": {
# "rope_theta": 10000,
# "rope_type": "default"
# }
# },
# "sliding_window": 512,
# "use_bidirectional_attention": false,
# "use_cache": true,
# "vocab_size": 262144
# },
# "dropout_rate": 0.0,
# "dtype": "bfloat16",
# "encoder": {
# "attention_dropout": 0.0,
# "boi_token_index": 255999,
# "dropout_rate": 0.0,
# "dtype": "bfloat16",
# "eoi_token_index": 256000,
# "image_token_index": 256001,
# "initializer_range": 0.02,
# "mm_tokens_per_image": 256,
# "model_type": "t5gemma2_encoder",
# "text_config": {
# "_name_or_path": "",
# "_sliding_window_pattern": 6,
# "add_cross_attention": false,
# "architectures": null,
# "attention_bias": false,
# "attention_dropout": 0.0,
# "attn_logit_softcapping": null,
# "bos_token_id": 2,
# "chunk_size_feed_forward": 0,
# "cross_attention_hidden_size": null,
# "decoder_start_token_id": null,
# "dropout_rate": 0.0,
# "dtype": "bfloat16",
# "eos_token_id": 1,
# "final_logit_softcapping": null,
# "finetuning_task": null,
# "head_dim": 256,
# "hidden_activation": "gelu_pytorch_tanh",
# "hidden_size": 640,
# "id2label": {
# "0": "LABEL_0",
# "1": "LABEL_1"
# },
# "initializer_range": 0.02,
# "intermediate_size": 2048,
# "is_decoder": false,
# "is_encoder_decoder": false,
# "label2id": {
# "LABEL_0": 0,
# "LABEL_1": 1
# },
# "layer_types": [
# "sliding_attention",
# "sliding_attention",
# "sliding_attention",
# "sliding_attention",
# "sliding_attention",
# "full_attention",
# "sliding_attention",
# "sliding_attention",
# "sliding_attention",
# "sliding_attention",
# "sliding_attention",
# "full_attention",
# "sliding_attention",
# "sliding_attention",
# "sliding_attention",
# "sliding_attention",
# "sliding_attention",
# "full_attention"
# ],
# "max_position_embeddings": 32768,
# "model_type": "t5gemma2_text",
# "num_attention_heads": 4,
# "num_hidden_layers": 18,
# "num_key_value_heads": 1,
# "output_attentions": false,
# "output_hidden_states": false,
# "pad_token_id": 0,
# "prefix": null,
# "problem_type": null,
# "query_pre_attn_scalar": 256,
# "return_dict": true,
# "rms_norm_eps": 1e-06,
# "rope_parameters": {
# "full_attention": {
# "factor": 8.0,
# "rope_theta": 1000000,
# "rope_type": "linear"
# },
# "sliding_attention": {
# "rope_theta": 10000,
# "rope_type": "default"
# }
# },
# "sep_token_id": null,
# "sliding_window": 512,
# "task_specific_params": null,
# "tie_word_embeddings": true,
# "tokenizer_class": null,
# "use_bidirectional_attention": false,
# "use_cache": true,
# "vocab_size": 262144
# },
# "vision_config": {
# "_name_or_path": "",
# "add_cross_attention": false,
# "architectures": null,
# "attention_dropout": 0.0,
# "bos_token_id": null,
# "chunk_size_feed_forward": 0,
# "cross_attention_hidden_size": null,
# "decoder_start_token_id": null,
# "dropout_rate": 0.0,
# "dtype": "bfloat16",
# "eos_token_id": null,
# "finetuning_task": null,
# "hidden_act": "gelu_pytorch_tanh",
# "hidden_size": 1152,
# "id2label": {
# "0": "LABEL_0",
# "1": "LABEL_1"
# },
# "image_size": 896,
# "intermediate_size": 4304,
# "is_decoder": false,
# "is_encoder_decoder": false,
# "label2id": {
# "LABEL_0": 0,
# "LABEL_1": 1
# },
# "layer_norm_eps": 1e-06,
# "model_type": "siglip_vision_model",
# "num_attention_heads": 16,
# "num_channels": 3,
# "num_hidden_layers": 27,
# "output_attentions": false,
# "output_hidden_states": false,
# "pad_token_id": null,
# "patch_size": 14,
# "prefix": null,
# "problem_type": null,
# "return_dict": true,
# "sep_token_id": null,
# "task_specific_params": null,
# "tie_word_embeddings": true,
# "tokenizer_class": null,
# "vision_use_head": false,
# "vocab_size": 262144
# },
# "vocab_size": 262144
# },
# "eoi_token_index": 256000,
# "eos_token_id": 1,
# "image_token_index": 256001,
# "initializer_range": 0.02,
# "is_encoder_decoder": true,
# "model_type": "t5gemma2",
# "pad_token_id": 0,
# "transformers_version": "5.0.0.dev0",
# "vocab_size": 262144
# }