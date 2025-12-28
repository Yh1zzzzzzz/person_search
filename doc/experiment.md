# Understanding by Generation：基于本仓库落地的实验代码架构方案

> 目标：在 Text-Based Person Retrieval 中，通过 **Masked Generative Auxiliary Task** 倒逼 Encoder 学到更细粒度的图文对齐；推理时丢弃生成分支，只保留 Siamese Encoder 做检索。

本文档以“最少改动复用 IRRA 训练/评估管线”为原则，给出 Phase 1~4 的代码落地方案、模块划分、数据字段约定，以及建议的工程改进点。


## 0. 你现在这份 IRRA 仓库的关键事实（决定怎么改）

- 训练/评估主链路非常清晰：`train.py` → `datasets/build.py` → `model/build.py` → `processor/processor.py` → `utils/metrics.py`。
- **训练 loop 不关心模型内部**：只要 `model(batch)` 返回一个 dict（含 `xxx_loss`），训练器会把所有 `*loss*` 相加反传。
- **评估只要求模型有 `encode_image()` / `encode_text()`**（见 `utils/metrics.py:Evaluator`）。

因此：你要做 Phase 1~4，最稳的策略是**保持训练器和评估器不变**，新增一套模型类满足同样的接口，必要时扩展 dataset/colllate。


## 1. 总体架构：把“检索”和“生成辅助”拆成可插拔部件

建议新增以下目录/文件（保持与原结构风格一致）：

- `model/ubg/`（Understanding-by-Generation）
  - `model/ubg/build.py`：根据 args 构建你的 Phase1/2/3/4 模型（返回统一接口）。
  - `model/ubg/retrieval_heads.py`：投影头、温度、SDM/ITC 等统一封装。
  - `model/ubg/text_t5.py`：T5 Encoder 封装（HF），输出 token 序列与 pooled 向量。
  - `model/ubg/vision_siglip.py`：SigLIP Vision Tower 封装。
  - `model/ubg/fusion_t5.py`：Phase3 的“图像 token→T5 shared encoder”逻辑。
  - `model/ubg/gen_vq.py`：Phase4 的 VQGAN/VQ-VAE token 生成头（训练用）。

- `datasets/ubg/`
  - `datasets/ubg/tokenization.py`：把“CLIP tokenizer”和“T5 tokenizer”统一成接口。
  - `datasets/ubg/dataset_vqgen.py`：Phase4 数据集（返回 masked image + vq target tokens）。
  - `datasets/ubg/collate.py`：支持变长文本（T5）与额外字段的 collate。

- `configs/ubg/`（可选，但强烈建议）
  - `phase1.yaml`、`phase2.yaml`、`phase3.yaml`、`phase4.yaml`：避免命令行参数爆炸。

> 你也可以不新建这么多文件，而是直接把 Phase1-4 都塞进 `model/build.py`。但长期看会变得不可维护；我建议至少把 UBG 的新代码放到 `model/ubg/`，避免污染原 IRRA 基线。


## 2. 统一接口：保证训练器/评估器完全复用

你的新模型类（无论 Phase1-4）都建议遵循下面的最小接口：

- `forward(batch) -> Dict[str, Tensor]`
  - 必须包含：`sdm_loss`（或你想用的检索损失）
  - Phase4 额外包含：`gen_loss`
- `encode_image(images) -> Tensor[B, D]`
- `encode_text(text_inputs) -> Tensor[B, D]`

这样你不需要改：
- `processor/processor.py`（训练 loop）
- `utils/metrics.py`（检索评估）

唯一可能要改的是：
- `datasets/build.py`：当你启用 T5 tokenizer/变长文本时，collate 与 batch 字段需要扩展。


## 3. 4 个 Phase 如何最小改动落地

下面按你定义的 Phase 逐步说明“应该加什么代码/怎么接进现有仓库”。


### Phase 1：ViT-B(IRRA visual) + T5 Text Encoder

**目标**：只替换文本编码器，验证 “LLM/T5 encoder 文本理解” 是否带来检索提升。

**推荐实现方式（最稳）**
- 视觉塔：复用本仓库的 CLIP ViT（`model/clip_model.py`）的 **visual** 路径。
  - 直接调用现有 `build_CLIP_from_openai_pretrained()` 得到 `clip_model`。
  - `encode_image()` 用 `clip_model.encode_image()`，取 CLS（`[:,0,:]`）作为全局。
- 文本塔：用 HuggingFace 的 T5 encoder。
  - 使用 `transformers.AutoTokenizer` + `AutoModel` 或 `T5EncoderModel`（具体类名以你环境 transformers 版本为准）。
  - `encode_text()`：取 encoder last_hidden_state 的某个 pooling（例如 eos/mean pool），再通过一个 `nn.Linear` 投影到共享维度 $D$。

**你需要新增的 batch 字段**
- `caption_ids_t5`（`input_ids`）
- `caption_attn_mask_t5`（`attention_mask`）

**对现有代码的最小改动点**
1) 在 `utils/options.py` 加开关（示例）
- `--model_name`：`irra_clip`（默认） / `ubg_phase1`
- `--text_backbone`：`clip` / `t5`
- `--hf_text_name_or_path`：例如 `google/t5gemma-...`（你自己的名字）

2) 在 `datasets/build.py` 增加 conditional：
- 如果 `text_backbone==t5`：用 HF tokenizer 做 padding，返回上述两个字段。

3) 在 `model/build.py` dispatch：
- `if args.model_name.startswith('ubg_'):` → 调用 `model/ubg/build.py:build_ubg_model()`


### Phase 2：SigLIP (Standalone) + T5 Text Encoder（视觉不进 T5）

**目标**：验证 SigLIP 视觉底座是否优于 ViT-B。

**推荐实现方式**
- 视觉塔：SigLIP Vision Tower（只取 vision encoder 输出）
- 文本塔：继续 Phase1 的 T5 encoder
- 二者各自投影到共享维度 $D$，用 `SDM/ITC` 做检索损失。

**工程建议**
- 优先用“成熟轮子”来加载 SigLIP：
  - 方案 A：HuggingFace Transformers（若你版本支持 SigLIP vision model）
  - 方案 B：`open_clip`（通常更稳定、更贴近 CLIP 类模型）

你这一步不需要动 T5 shared encoder，不需要做深融合。


### Phase 3：Deep Fusion（SigLIP → T5 Shared Encoder）+（Text → T5 Shared Encoder）

**目标**：让视觉特征进入 LLM 语义空间，形成 Native Siamese（共享 encoder 权重）。

**关键设计选择（建议你尽早定下来）**
- 你希望“共享 encoder”处理两种输入：
  1) 文本：token embedding（标准 T5 流程）
  2) 图像：SigLIP patch embedding → 线性映射到 T5 hidden size → 作为 `inputs_embeds` 喂进同一个 T5 encoder

**落地方式（HF T5 支持 inputs_embeds）**
- 文本路：`encoder(input_ids, attention_mask)`
- 图像路：
  - `siglip_vision(pixel_values)` → `image_tokens`（[B, P, C]）
  - `image_tokens_proj = Linear(C → hidden_size)`
  - `encoder(inputs_embeds=image_tokens_proj, attention_mask=image_mask)`

**Siamese 输出 pooling**
- 文本：mean pool（按 attention_mask）或取最后一个非 pad token
- 图像：mean pool（按 image_mask）或取第一 token（你可以人为加一个 learnable CLS token）

> 这一步建议先不引入 decoder/生成，只验证“深融合 + 共享语义空间”是否能稳定提升检索。


### Phase 4：Ours（Phase3 + T5 Decoder + VQ-VAE / VQGAN）

**目标**：训练时增加 Masked Generative Auxiliary Task：
- Mask 输入图片
- Decoder 结合文本预测缺失部分的 VQ token id
- 推理时丢弃 decoder + VQGAN

**建议的最小可行定义（MVP）**
- 先把“生成目标”定义成 **整图** 的 VQ token（或固定 patch 区域），不要一上来就做复杂 mask 规则；否则 debug 成本巨大。
- 生成输入：
  - encoder condition：`[text_tokens]` + `[image_tokens(masked)]`（你可以在 encoder 侧做 concat，并加 modality/type embedding）
  - decoder target：`vq_code_ids`（长度为 `L_vq`）
- 生成 loss：`CrossEntropy(ignore_index=pad)`

**数据管线新增内容**
- `masked_images`：对输入图做 mask（建议先在 dataset 端做 deterministic mask，便于复现）
- `vq_target_ids`：由 frozen VQGAN 编码得到的 codebook id 序列

**强烈建议：离线预计算 VQ tokens**
- 训练时在线跑 VQGAN encoder 会非常慢，而且会占显存。
- 建议在 `training_data/<dataset>/processed_data/` 下缓存每张图的 `vq_ids.npy`（或 `.pt`），dataset 读取即可。


## 4. Batch 字段契约（避免你改着改着 “忘了 batch 里有什么”）

建议统一成：

- 通用字段（Phase1-4 都有）
  - `images`: FloatTensor[B,3,H,W]
  - `pids`: LongTensor[B]

- 文本字段（二选一）
  - CLIP tokenizer：`caption_ids`（本仓库原有，固定 77）
  - T5 tokenizer：`caption_ids_t5`, `caption_attn_mask_t5`

- Phase4 生成字段
  - `masked_images`
  - `vq_target_ids`
  - （可选）`vq_target_attn_mask`


## 5. 损失组合与训练/推理开关

你的总损失：
$$L_{total} = L_{SDM}(retrieval) + \lambda \cdot L_{Gen}(CE)$$

建议新增参数：
- `--gen_weight`（即 $\lambda$）
- `--enable_gen`（只在 Phase4 开）

推理时：
- `encode_image/encode_text` 只走 encoder 路径
- decoder/vqgan 完全不参与


## 6. 我建议你优先实现的顺序（把风险拆小）

1) **先实现 Phase1**：改 tokenizer + T5 text encoder + SDM loss
2) 再实现 Phase2：替换视觉塔为 SigLIP，保持其余不动
3) 再实现 Phase3：共享 T5 encoder（输入 embeds）+ 两路 pooling
4) 最后上 Phase4：VQ token 离线缓存 + decoder CE

每一步都应该能在 `test.py` 直接评估 Rank-1，保证你随时知道增益来自哪里。


## 7. 对当前仓库的可改进点（我建议的“高性价比”优化）

### 7.1 修复隐藏 bug：text pos embedding resize 缺失
- `model/clip_model.py` 里 `resize_text_pos_embed()` 被调用但未定义。
- 我已补齐该函数（线性插值），避免你将来改 `text_length` 时踩坑。

### 7.2 Tokenizer/Collate 需要抽象（为 T5 变长文本做准备）
- 现有 `datasets/build.py:collate()` 假设 batch 里的序列都是固定长度 tensor（CLIP 77 是 OK）。
- 一旦引入 T5（变长 + attention_mask），建议把 collate 抽到 `datasets/ubg/collate.py`：
  - 支持 tokenizer.pad
  - 支持可选字段

### 7.3 不要重复造轮子：优先复用成熟 backbone loader
- 当前仓库自带 OpenAI CLIP 实现很可改，但对 SigLIP / 新模型来说维护成本高。
- Phase2/3 推荐使用 HuggingFace 或 open_clip 直接加载 SigLIP，减少“权重键名/shape 对齐”时间。

### 7.4 性能小优化（不改算法但更快）
- DataLoader 建议开启：`pin_memory=True`, `persistent_workers=True`（当 `num_workers>0`）
- 若你后续引入 VQ token 在线生成，建议强制离线缓存，否则训练速度会被 encoder 卡死。


## 8. 依赖建议（你环境不一定需要全装，但最好记录）

- `transformers`, `accelerate`（HF 模型）
- `sentencepiece`（T5 tokenizer 常用）
- `open_clip_torch`（可选：加载 SigLIP/CLIP 族更方便）
- VQGAN/VQ-VAE：
  - 可选 `taming-transformers` 或你已有的 VQGAN 实现


## 9. 我需要你确认的 3 个关键信息（决定 Phase3/4 的具体实现）

1) 你计划用的 SigLIP checkpoint 名称/来源？（HF 还是 open_clip）
2) Phase3 的“共享 T5 encoder”输入：你希望图像 token 与文本 token **concat 后一起跑一个 encoder**，还是 **两路分别跑同一个 encoder**（权重共享但不拼接）？
3) Phase4 的 VQGAN：你打算用哪个预训练权重（以及 codebook size、token 序列长度）？这决定 decoder 输出维度与数据缓存格式。

> 你把这 3 点确定下来后，我可以进一步把“建议新增的文件 skeleton”和“需要改动的函数签名”细化到可直接开工的程度。
