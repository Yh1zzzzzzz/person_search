import argparse


def get_args():
    parser = argparse.ArgumentParser(description="IRRA Args")
    ######################## general settings ########################
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--name", default="baseline", help="experiment name to save")
    parser.add_argument("--output_dir", default="logs")
    parser.add_argument("--log_period", default=100)
    parser.add_argument("--eval_period", default=1)
    parser.add_argument("--val_dataset", default="test") # use val set when evaluate, if test use test set
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--resume_ckpt_file", default="", help='resume from ...')

    ######################## model general settings ########################
    parser.add_argument(
        "--backbone",
        default="clip",
        choices=["clip", "t5gemma2", "t5gemma2_vion_tower"],
        help="backbone type: clip (default IRRA) or t5gemma2 (HF multimodal)"
    )
    parser.add_argument(
        "--hf_model_name_or_path",
        default="T5_270M_Base",
        help="HuggingFace model name or local path for T5Gemma2 (e.g. google/t5gemma-2-270m-270m or ./T5_270M_Base)"
    )
    parser.add_argument("--pretrain_choice", default='ViT-B/16') # whether use pretrained model
    parser.add_argument("--temperature", type=float, default=0.02, help="initial temperature value, if 0, don't use temperature")
    parser.add_argument("--img_aug", default=False, action='store_true')

    ## cross modal transfomer setting
    parser.add_argument("--cmt_depth", type=int, default=4, help="cross modal transformer self attn layers")
    parser.add_argument("--masked_token_rate", type=float, default=0.8, help="masked token rate for mlm task")
    parser.add_argument("--masked_token_unchanged_rate", type=float, default=0.1, help="masked token unchanged rate")
    parser.add_argument("--lr_factor", type=float, default=5.0, help="lr factor for random init self implement module")
    parser.add_argument("--MLM", default=False, action='store_true', help="whether to use Mask Language Modeling dataset")

    ######################## loss settings ########################
    parser.add_argument("--loss_names", default='sdm+id+mlm', help="which loss to use ['mlm', 'cmpm', 'id', 'itc', 'sdm']")
    parser.add_argument("--mlm_loss_weight", type=float, default=1.0, help="mlm loss weight")
    parser.add_argument("--id_loss_weight", type=float, default=1.0, help="id loss weight")
    parser.add_argument(
        "--bnneck",
        default=False,
        action="store_true",
        help="Enable BNNeck (separate BN for image/text) for the ID classification branch.",
    )
    parser.add_argument("--gen_loss_weight", type=float, default=1.0, help="generation loss weight (t5gemma2)")
    parser.add_argument("--gen_prompt", type=str, default="Caption", help="generation prompt text after <start_of_image>")
    parser.add_argument("--gen_prompt_length", type=int, default=32, help="max length for generation prompt token ids")
    parser.add_argument("--feature_dim", type=int, default=1024, help="projection dim for retrieval heads (t5gemma2)")
    parser.add_argument(
        "--projector_hidden_dim",
        type=int,
        default=2048,
        help="Hidden dim of the vision-tower projector MLP (t5gemma2_vion_tower).",
    )
    parser.add_argument("--t5_image_size", type=int, default=448, help="expected square image size for t5gemma2 vision tower")
    
    ######################## vison trainsformer settings ########################
    parser.add_argument("--img_size", type=tuple, default=(384, 128))
    parser.add_argument("--stride_size", type=int, default=16)

    ######################## text transformer settings ########################
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)
    parser.add_argument(
        "--mm_max_length",
        type=int,
        default=512,
        help="Max sequence length for multimodal processor inputs (must be >= image tokens, e.g. 256 for 1 image).",
    )

    ######################## solver ########################
    parser.add_argument("--optimizer", type=str, default="Adam", help="[SGD, Adam, Adamw]")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--projector_lr",
        type=float,
        default=1e-3,
        help="Learning rate for vision-tower projector (e.g., vision_tower_align).",
    )
    parser.add_argument(
        "--classifier_lr",
        type=float,
        default=1e-4,
        help="Learning rate for ID classifier (and BNNeck affine params when enabled).",
    )
    parser.add_argument("--bias_lr_factor", type=float, default=2.)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)
    
    ######################## scheduler ########################
    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument("--milestones", type=int, nargs='+', default=(20, 50))
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--lrscheduler", type=str, default="cosine")
    parser.add_argument("--target_lr", type=float, default=0)
    parser.add_argument("--power", type=float, default=0.9)

    ######################## dataset ########################
    parser.add_argument("--dataset_name", default="CUHK-PEDES", help="[CUHK-PEDES, ICFG-PEDES, RSTPReid]")
    parser.add_argument("--sampler", default="random", help="choose sampler from [idtentity, random]")
    parser.add_argument("--num_instance", type=int, default=4)
    parser.add_argument("--root_dir", default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--test", dest='training', default=True, action='store_false')

    ######################## training efficiency ########################
    parser.add_argument(
        "--amp",
        default=False,
        action="store_true",
        help="Enable PyTorch AMP autocast in training (recommended for t5gemma2).",
    )
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="AMP dtype when --amp is enabled (bf16 preferred if supported).",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch = batch_size * grad_accum_steps).",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.0,
        help="Clip grad norm if > 0.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        default=False,
        action="store_true",
        help="Enable HF gradient checkpointing for t5gemma2 backbone to save VRAM.",
    )
    parser.add_argument(
        "--reuse_mm_encoder_for_image",
        default=False,
        action="store_true",
        help=(
            "When loss_names includes gen, reuse the multimodal encoder output (from the gen forward) "
            "to compute image retrieval features, avoiding a second vision/encoder pass. "
            "May slightly change retrieval features because encoder sees the prompt tokens."
        ),
    )

    parser.add_argument(
        "--tokenizers_parallelism",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Set env TOKENIZERS_PARALLELISM to silence HuggingFace tokenizers fork warnings.",
    )

    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
        help=(
            "Attention implementation for HF models (if supported). "
            "'sdpa' is usually much more memory-efficient than 'eager'."
        ),
    )

    args = parser.parse_args()

    return args