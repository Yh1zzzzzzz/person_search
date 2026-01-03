"""quick_check_t5gemma2.py

用途
- 验证 resize + letterbox pad 到固定分辨率后的 pixel_values 是否符合模型期望尺寸
- 可选：跑多模态生成推理（image + prompt -> generate）
- 可选：跑 retrieval embedding（如果你需要对齐检查）
- 可选：跑训练前向（loss 分支，较重）

运行位置
- 建议在仓库根目录运行（包含 T5_270M_Base 的目录），例如：/root/multimodal/person_search

常用命令
- 仅检查预处理（默认不跑任何推理/前向）：
    `python model/quick_check_t5gemma2.py`

- 多模态生成推理（推荐 GPU）：
    `USE_CUDA=1 RUN_MM_GENERATE=1 python model/quick_check_t5gemma2.py`

- retrieval embedding 推理（可选）：
    `USE_CUDA=1 RUN_INFER=1 python model/quick_check_t5gemma2.py`

- 训练前向（losses；很吃显存）：
    `USE_CUDA=1 RUN_TRAIN_FORWARD=1 python model/quick_check_t5gemma2.py`

生成参数（可选，配合 RUN_MM_GENERATE=1）
- MAX_NEW_TOKENS=64
- NUM_BEAMS=1
- DO_SAMPLE=1 TEMPERATURE=0.7 TOP_P=0.9

可视化预处理结果（可选，默认不落盘）
- SHOW_PREPROCESSED=1 尝试弹出显示（无 GUI/无 DISPLAY 时可能失败）
- PRINT_PREPROCESSED_BASE64=1 打印 PNG 的 base64（不会写硬盘；内容较长）
- SAVE_PREPROCESSED=1 保存 resize+pad 后的图片到硬盘（不推荐；默认关闭）
- PREPROCESSED_OUT=preprocessed_448.png 指定保存路径（仅 SAVE_PREPROCESSED=1 时生效）

VS Code SSH 推荐查看方式（不占硬盘）
- 方案 A（推荐）：保存到内存文件系统 /dev/shm，然后在 VS Code 里直接打开
    `USE_CUDA=1 RUN_MM_GENERATE=1 SAVE_PREPROCESSED=1 PREPROCESSED_OUT=/dev/shm/preprocessed.png python model/quick_check_t5gemma2.py`
    说明：/dev/shm 通常是 tmpfs（RAM），重启/断开后不保留，不占磁盘配额。

- 方案 B：打印 base64 到终端，然后在本地解码成 png 查看（远端不写盘）
    远端：`PRINT_PREPROCESSED_BASE64=1 ...`
    本地解码示例（把复制出来的 base64 放到 PREPROCESSED.b64）：
    `python - <<'PY'\nimport base64\nopen('preprocessed.png','wb').write(base64.b64decode(open('PREPROCESSED.b64','rb').read()))\nprint('wrote preprocessed.png')\nPY`

修改输入
- LOCAL_MODEL_PATH: 本地 HF 模型目录（默认 T5_270M_Base）
- TEST_IMAGE_PATH: 测试图片路径（不存在会生成一张 dummy 图）
- TEST_PROMPT: 多模态推理的 prompt
- TARGET_H/TARGET_W: 固定分辨率（与你的模型适配一致，当前 448）
"""

import os
import sys
import io
import base64
# Allow running as a script: `python model/quick_check_t5gemma2.py`
# by ensuring the project root is on sys.path.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from PIL import Image
from transformers import AutoConfig, AutoTokenizer
from torchvision import transforms
from model.T5Gemma2_270 import PersonSearchT5Gemma2
from utils.letterbox import LetterboxResize

# ==========================================
# 用户配置区
# ==========================================
LOCAL_MODEL_PATH = "T5_270M_Base"  # 你的模型路径
TEST_IMAGE_PATH = "0363004.png"    # 测试图片路径
TEST_CAPTION = "A man wearing a black shirt and blue jeans."
TEST_PROMPT = "A pedestrian with <unused0> hair is wearing red and white shoes, a <unused1> hooded sweatshirt, and <unused2> pants."  # 多模态推理用的 prompt

# 【关键设置】
# 方法2：统一预处理到模型原生支持的 896x896。
TARGET_H, TARGET_W = 448, 448

def main():
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print(__doc__)
        return

    print(">>> [Start] Initializing Test Script...")
    
    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"!! Error: Model path not found: {LOCAL_MODEL_PATH}")
        return
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print("Creating dummy image...")
        Image.new('RGB', (128, 384), color='blue').save(TEST_IMAGE_PATH)
    
    # 1. Config & Tokenizer
    print(">>> Loading Config & Tokenizer...")
    config = AutoConfig.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)

    # 2. Model
    print(">>> Loading Model Weights...")
    model = PersonSearchT5Gemma2(
        config=config,
        hf_model_name_or_path=LOCAL_MODEL_PATH,
        num_classes=100,
        feature_dim=512
        ,image_size=TARGET_H
    )

    # 3. Device & dtype
    # Default: only validate preprocessing; do NOT run any model compute.
    # - Run multimodal generation inference: RUN_MM_GENERATE=1
    # - Run retrieval-style inference (encode image/text + similarity): RUN_INFER=1
    # - Run full train-forward (losses; heavy): RUN_TRAIN_FORWARD=1 (CUDA recommended)
    # - Select CUDA device move: USE_CUDA=1
    use_cuda = os.environ.get("USE_CUDA", "0") == "1"
    run_infer = os.environ.get("RUN_INFER", "0") == "1"
    run_mm_generate = os.environ.get("RUN_MM_GENERATE", "0") == "1"
    run_train_forward = os.environ.get("RUN_TRAIN_FORWARD", "0") == "1"

    show_preprocessed = os.environ.get("SHOW_PREPROCESSED", "0") == "1"
    print_preprocessed_base64 = os.environ.get("PRINT_PREPROCESSED_BASE64", "0") == "1"
    save_preprocessed = os.environ.get("SAVE_PREPROCESSED", "0") == "1"
    preprocessed_default_dir = "/dev/shm" if os.path.isdir("/dev/shm") else "."
    preprocessed_out = os.environ.get(
        "PREPROCESSED_OUT",
        os.path.join(preprocessed_default_dir, f"preprocessed_letterbox_{TARGET_H}.png"),
    )
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("USE_CUDA=1 but CUDA is not available")
    device = "cuda" if use_cuda else "cpu"
    target_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f">>> Moving to {device} ({target_dtype})...")
    model.to(device=device, dtype=target_dtype)


    # 5. Data Processing 
    print(">>> Processing Real Data...")
    
    image = Image.open(TEST_IMAGE_PATH).convert('RGB')
    
    # 5.1 Letterbox (resize + pad) so H=W=TARGET_H
    letterboxed = LetterboxResize(target_size=TARGET_H)(image)

    # Optional visualization without writing to disk
    if show_preprocessed:
        try:
            letterboxed.show(title=f"letterbox_{TARGET_H}")
            print("    Showing preprocessed image via PIL.Image.show()")
        except Exception as e:
            print(f"    [Warn] SHOW_PREPROCESSED failed (likely headless env): {e}")

    if print_preprocessed_base64:
        buf = io.BytesIO()
        letterboxed.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        # Print with markers + line wrapping for easy copy/paste
        print("    Preprocessed image (PNG) base64 (copy between markers):")
        print("-----BEGIN PREPROCESSED_PNG_BASE64-----")
        line_width = int(os.environ.get("BASE64_LINE_WIDTH", "120"))
        for i in range(0, len(b64), line_width):
            print(b64[i : i + line_width])
        print("-----END PREPROCESSED_PNG_BASE64-----")

    if save_preprocessed:
        try:
            letterboxed.save(preprocessed_out)
            print(f"    Saved preprocessed (letterbox) image to: {preprocessed_out}")
            if os.path.abspath(preprocessed_out).startswith("/dev/shm"):
                print("    (Tip) This path is tmpfs (/dev/shm): uses RAM, not disk")
        except Exception as e:
            print(f"    [Warn] Failed to save preprocessed image to {preprocessed_out}: {e}")

    # 5.2 ToTensor + Normalize
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    pixel_values = val_transforms(letterboxed).unsqueeze(0)
    pixel_values = pixel_values.to(device=device, dtype=target_dtype)
    print(f"    Pixel Shape: {pixel_values.shape} (Expected: 1, 3, 448, 448)")

    txt_tokens = tokenizer(
        TEST_CAPTION, 
        return_tensors="pt", 
        padding="max_length", 
        max_length=30, 
        truncation=True
    )
    input_ids = txt_tokens.input_ids.to(device)
    
    masked_input_ids = input_ids.clone()
    masked_input_ids[:, 5:8] = 0
    
    pids = torch.tensor([0], dtype=torch.long).to(device)

    batch = {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "masked_input_ids": masked_input_ids,
        "pids": pids
    }

    if not run_mm_generate and not run_infer and not run_train_forward:
        msg = "\n>>> ✅ Preprocessing OK (448x448)."
        msg += " Set RUN_MM_GENERATE=1 to run multimodal generation inference."
        msg += " Set RUN_INFER=1 to run retrieval inference."
        if not use_cuda:
            msg += " (CPU inference is supported but may be slow.)"
        msg += " Set RUN_TRAIN_FORWARD=1 (and preferably USE_CUDA=1) to run train-forward losses."
        msg += " Use --help to see full usage."
        print(msg)
        return

    # 6. Multimodal generation inference (image + prompt -> generate text)
    if run_mm_generate:
        print("\n>>> Multimodal Generation (Eval Mode): backbone.generate")
        model.eval()

        prompt_tokens = tokenizer(
            TEST_PROMPT,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=False,
        )
        prompt_input_ids = prompt_tokens.input_ids.to(device)
        prompt_attention_mask = prompt_tokens.attention_mask.to(device)

        # Construct multimodal input: [BOI][IMG x N][EOI] + prompt
        B = int(pixel_values.shape[0])
        img_part_ids = model.construct_image_tokens(B, device)
        multimodal_input_ids = torch.cat([img_part_ids, prompt_input_ids], dim=1)
        img_mask = torch.ones_like(img_part_ids)
        multimodal_attention_mask = torch.cat([img_mask, prompt_attention_mask], dim=1)

        gen_kwargs = {
            "max_new_tokens": int(os.environ.get("MAX_NEW_TOKENS", "64")),
            "num_beams": int(os.environ.get("NUM_BEAMS", "1")),
            "do_sample": os.environ.get("DO_SAMPLE", "0") == "1",
        }
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = float(os.environ.get("TEMPERATURE", "0.7"))
            gen_kwargs["top_p"] = float(os.environ.get("TOP_P", "0.9"))

        with torch.no_grad():
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=target_dtype):
                    gen_ids = model.backbone.generate(
                        input_ids=multimodal_input_ids,
                        attention_mask=multimodal_attention_mask,
                        pixel_values=pixel_values,
                        **gen_kwargs,
                    )
            else:
                gen_ids = model.backbone.generate(
                    input_ids=multimodal_input_ids,
                    attention_mask=multimodal_attention_mask,
                    pixel_values=pixel_values,
                    **gen_kwargs,
                )

        decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        print(f"    prompt: {TEST_PROMPT}")
        print(f"    generated: {decoded[0] if len(decoded) else ''}")

        if not run_infer and not run_train_forward:
            if save_preprocessed:
                print(f"\n>>> Preprocessed image saved at: {preprocessed_out}")
            print("\n>>> ✅ Multimodal generation OK!")
            return

    # 6. Inference: retrieval-style embeddings + similarity
    if run_infer:
        print("\n>>> Inference Pass (Eval Mode): encode image/text + similarity")
        model.eval()
        with torch.no_grad():
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=target_dtype):
                    v = model.encode_image_only(batch["pixel_values"])
                    t = model.encode_text_only(batch["input_ids"])
            else:
                v = model.encode_image_only(batch["pixel_values"])
                t = model.encode_text_only(batch["input_ids"])

            v_norm = torch.nn.functional.normalize(v, dim=1)
            t_norm = torch.nn.functional.normalize(t, dim=1)
            sim = (v_norm * t_norm).sum(dim=1)

        print(f"    vision_emb: {tuple(v.shape)} | text_emb: {tuple(t.shape)}")
        print(f"    cosine_sim (per-sample): {sim.detach().float().cpu().tolist()}")

    if not run_train_forward:
        if save_preprocessed:
            print(f"\n>>> Preprocessed image saved at: {preprocessed_out}")
        print("\n>>> ✅ Inference OK! resize+pad -> model embeddings aligned.")
        return

    # 7. Train-forward (losses): recommended on CUDA
    if device != "cuda":
        print("\n>>> ⚠️  RUN_TRAIN_FORWARD=1 on CPU may be very slow.")

    if device == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes < 12 * 1024**3:
            print(
                f"\n>>> ⚠️  Skip train-forward: only {free_bytes/1024**3:.2f}GB CUDA memory free. "
                "Close other GPU jobs and retry."
            )
            return

    print("\n>>> Forward Pass (Train Mode; losses)...")
    model.train()
    with torch.no_grad():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=target_dtype):
                outputs = model(batch)
        else:
            outputs = model(batch)

    print("\n>>> Output Losses:")
    for k, v in outputs.items():
        if "loss" in k:
            print(f"    {k}: {v.item():.4f}")

    if save_preprocessed:
        print(f"\n>>> Preprocessed image saved at: {preprocessed_out}")
    print("\n>>> ✅ Train-forward OK! Fixed-size pipeline is aligned.")

if __name__ == "__main__":
    main()