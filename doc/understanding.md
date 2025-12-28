# IRRA 代码仓库档案（面向快速上手与二次开发）

> 目标：用“从哪里开始读、每个模块做什么、调用链怎么走”的视角，把 IRRA（CVPR 2023）这份实现拆开讲清楚，并标注作者如何加载/改造预训练 CLIP（ViT/ResNet）。


## 1. 你真正需要理解的主调用链

### 训练（train）
入口脚本：`train.py`

1. 解析参数：`utils/options.py:get_args()`
2. 构建数据：`datasets/build.py:build_dataloader()`
3. 构建模型：`model/build.py:build_model()`（返回 `IRRA`）
4. 构建优化器与学习率策略：`solver/build.py`、`solver/lr_scheduler.py`
5. 训练循环：`processor/processor.py:do_train()`
6. 周期性评估与保存 best：`utils/metrics.py:Evaluator` + `utils/checkpoint.py:Checkpointer`

### 测试/评估（test）
入口脚本：`test.py`

1. 加载训练时保存的配置：`utils/iotools.py:load_train_configs()`
2. 构建测试集 dataloader：`datasets/build.py:build_dataloader(args)`（`args.training=False`）
3. 构建模型并加载 `best.pth`：`model/build.py` + `utils/checkpoint.py`
4. 评估：`processor/processor.py:do_inference()` → `utils/metrics.py:Evaluator.eval()`


## 2. 目录/文件职责速查

### 顶层脚本
- `train.py`：分布式/单卡训练入口，串起 dataloader、model、optimizer、scheduler、trainer。
- `test.py`：从 `configs.yaml` 恢复训练参数后做评估。
- `run_irra.sh`：训练命令示例。
- `visualize.py`：检索可视化脚本（从 topk 结果中画 query 对应的检索图）。
- `download_T5.py`、`T5_test/`：与本仓库核心 IRRA（CLIP-based）主链路不强耦合，属于额外实验/工具。

### datasets/
- `datasets/build.py`
  - `build_transforms()`：图像增强/归一化（mean/std 与 CLIP 一致：`[0.481, 0.458, 0.408]` / `[0.269, 0.261, 0.276]`）。
  - `build_dataloader()`：按 `args.training` 切换 train/val/test dataloader；按 `args.MLM` 切换普通训练集 vs MLM 训练集。
  - `collate()`：把 list-of-dict 组装成 tensor dict。
- `datasets/bases.py`
  - `BaseDataset`：数据集统计打印。
  - `tokenize()`：CLIP BPE tokenizer（`utils/simple_tokenizer.py`）+ `<|startoftext|>`/`<|endoftext|>` 封装并 pad 到 `text_length`。
  - `ImageTextDataset`：训练集样本返回：`images` + `caption_ids` + `pids` + `image_ids`。
  - `ImageTextMLMDataset`：在训练样本基础上增加 `mlm_ids` 和 `mlm_labels`（BERT 风格 mask 规则）。
  - `ImageDataset`/`TextDataset`：测试时将 gallery/query 拆开，Evaluator 分别编码。
- `datasets/{cuhkpedes,icfgpedes,rstpreid}.py`
  - 读取对应 json 标注，训练集展开为“(pid, image_id, img_path, caption)”列表；测试/验证集组织为 image 列表和 caption 列表。
- `datasets/sampler.py`：`RandomIdentitySampler`（N identity × K instances 组成 batch）。
- `datasets/sampler_ddp.py`：分布式版本 identity sampler（本仓库标注了 TODO/bugs，默认训练命令未启用）。

### model/
- `model/build.py`
  - `IRRA`：主模型类。以“完整 CLIP”为 backbone，并按 `loss_names` 选择性启用额外 head/module。
  - `build_model()`：构建 `IRRA` 后调用 `convert_weights(model)` 将可转换权重转为 fp16。
- `model/clip_model.py`
  - OpenAI CLIP 的“可修改版本”实现（同时支持 ResNet-CLIP 与 ViT-CLIP），并在 ViT 场景对 ReID 输入分辨率/stride 做了适配。
  - 提供 `build_CLIP_from_openai_pretrained()`：从 OpenAI URL 下载并加载权重。
- `model/objectives.py`
  - 各种损失：`SDM`（本文核心之一）、`ITC`（InfoNCE）、`ID`（分类）、`MLM`、`CMPM`。

### processor/
- `processor/processor.py`
  - `do_train()`：标准训练循环 + TensorBoard + 按 epoch eval + 保存 best。
  - `do_inference()`：调用 `Evaluator.eval()` 输出检索指标。

### solver/
- `solver/build.py`：构建 optimizer/scheduler。
  - 对随机初始化模块（key 含 `cross`、`classifier`、`mlm_head`）使用更大的学习率（`args.lr_factor`）。
- `solver/lr_scheduler.py`：warmup + milestones/cosine 等。

### utils/
- `utils/metrics.py`：`Evaluator`，将 text/image 编码为 embedding，L2 normalize，点积得到相似度，算 R@k / mAP / mINP。
- `utils/checkpoint.py`：保存/加载/断点恢复（支持 strip `module.` 前缀）。
- `utils/logger.py`：stdout + 文件日志。
- `utils/iotools.py`：读图、读 json、保存/加载 configs.yaml。
- `utils/comm.py`：DDP 通信（barrier、all_gather、reduce_dict）。
- `utils/simple_tokenizer.py`：CLIP 的 BPE tokenizer（额外加入 `<|mask|>` token 以支持 MLM）。


## 3. IRRA 模型内部：forward 里发生了什么

核心文件：`model/build.py`

### 3.1 训练任务开关（loss_names）
`args.loss_names` 形如：`sdm+mlm+id`。

`IRRA._set_task()` 会把它拆成 `current_task`，`forward()` 根据任务往 `ret` 字典里塞 loss：
- `sdm_loss`：`objectives.compute_sdm()`（核心改进之一，匹配分布对齐）
- `itc_loss`：`objectives.compute_itc()`（InfoNCE）
- `cmpm_loss`：`objectives.compute_cmpm()`
- `id_loss`：image/text 两路各做一次分类 CE，再平均
- `mlm_loss`：跨模态交互后对 masked token 做 CE

训练循环里：`processor/processor.py:do_train()` 会把 `ret` 中所有 key 含 `loss` 的项求和作为 `total_loss`。

### 3.2 特征来源：CLIP 输出的是“token 序列”，不是单向量
在 `model/build.py:IRRA.forward()`：
- `image_feats, text_feats = self.base_model(images, caption_ids)`
  - `image_feats` 形状：`[B, 1 + num_patches, D]`
  - `text_feats` 形状：`[B, text_len, D]`

然后 IRRA 取全局表征：
- 图像：`i_feats = image_feats[:, 0, :]`（CLS token）
- 文本：`t_feats = text_feats[range(B), caption_ids.argmax(dim=-1)]`（EOT token，CLIP 习惯做法）

这点非常关键：IRRA 的“推理不增加额外 cost”主要体现在**最终检索仍用全局 embedding（CLS/EOT）**；跨模态关系推理模块只在训练的辅助任务（如 MLM）里使用。

### 3.3 IRRA 的“Implicit Relation Reasoning/Aligning”在代码里体现为：MLM + 跨模态 Transformer
当启用 `mlm`：
- 文本侧另造一个 `mlm_ids`（随机 mask）
- 先用 CLIP 文本编码得到 `mlm_feats`（token 级）
- `cross_former(q=mlm_feats, k=image_feats, v=image_feats)`
  - 先做一次 `nn.MultiheadAttention`（cross-attn）
  - 再过 `cross_modal_transformer`（多层 self-attn）
  - 最后 `mlm_head` 做 token 分类（vocab 级）


## 4. 作者如何加载预训练 CLIP / ViT，以及“魔改点”是什么

核心文件：`model/clip_model.py` + `model/build.py`

### 4.1 预训练权重从哪来、怎么加载
`model/build.py:IRRA.__init__()` 会调用：

```python
self.base_model, base_cfg = build_CLIP_from_openai_pretrained(
    args.pretrain_choice, args.img_size, args.stride_size
)
```

`model/clip_model.py:build_CLIP_from_openai_pretrained()` 的行为：
1. 通过 `_MODELS` 将名字（如 `ViT-B/16`）映射到 OpenAI 的公开权重 URL。
2. `_download()` 下载到 `~/.cache/clip/` 并做 sha256 校验。
3. 尝试 `torch.jit.load()`；失败则 `torch.load()` state_dict。
4. 根据 state_dict 是否含 `visual.proj` 判断是 ViT 还是 ResNet 视觉编码器。
5. 从 state_dict 反推出 embed_dim、层数、patch_size、context_length 等配置，然后创建本仓库自定义 `CLIP(**model_cfg)`。
6. 调用 `model.load_param(state_dict)` 把权重拷回去（并在必要时 resize positional embedding）。

### 4.2 ViT 适配 ReID 分辨率（384×128）与 stride：最关键的“魔改”
OpenAI CLIP 的 ViT 默认是 224×224、patch stride=patch_size（通常 16）。IRRA 为 person ReID 常用纵向长图输入（默认 `args.img_size=(384,128)`），作者做了两步适配：

1) **改输入分辨率与 stride（patch 抽样更密/更稀）**
- 在 `build_CLIP_from_openai_pretrained()`：
  - `model_cfg['image_resolution'] = image_size`（从 224×224 改到 384×128）
  - `model_cfg['stride_size'] = stride_size`（默认 16）
- 在 `VisionTransformer.__init__()`：
  - `conv1 = nn.Conv2d(..., kernel_size=patch_size, stride=stride_size)`
  - 并用 `(H - patch)/stride + 1` 计算 `num_y/num_x` 和 `num_patches`

2) **视觉 positional embedding resize**
- `CLIP.load_param()` 对 key `visual.positional_embedding` 做 shape 检测
- 若不匹配，调用 `resize_pos_embed()`：
  - 把原先的正方形网格 pos embed reshape 成 `(gs_old, gs_old)`
  - 再用 `F.interpolate(..., mode='bilinear')` 插值到 `(num_y, num_x)`
  - 最后拼回 `[1 + num_patches, D]`

这就是“把 ViT 从方形输入变成长条 ReID 输入”的核心工程点。

> 注意：`CLIP.load_param()` 里也尝试对文本 `positional_embedding` 调用 `resize_text_pos_embed()`，但当前仓库中该函数没有定义；不过默认 `text_length=77` 与 OpenAI CLIP 一致，所以通常不会触发。若你改了 `args.text_length`，这里会报错，需要补实现。

### 4.3 返回“token 序列特征”而不是只返回 CLS：为跨模态推理铺路
与许多“只取 CLS/EOT”的实现不同，这里的 CLIP 被改造成：
- `VisionTransformer.forward()`：对整段 token 序列做 `ln_post`，并对所有 token 投影到 embed_dim：返回 `[B, 1+P, D]`。
- `CLIP.encode_text()`：`x = ln_final(x)` 后对所有 token 做 `x @ text_projection`：返回 `[B, T, D]`。

IRRA 在检索时仍然只用 CLS/EOT（全局向量），但在 MLM 分支里需要**图像 patch token + 文本 token**做 cross-attn，这个“输出序列”改动就是为此服务。

### 4.4 fp16 策略：convert_weights + fp32 LayerNorm
`model/build.py:build_model()` 会对整个 IRRA 调 `convert_weights(model)`：
- Conv/Linear/MHA 的权重与 bias 转 half
- `LayerNorm` 用自定义实现：内部先 cast 到 fp32 计算，再 cast 回原 dtype（避免 fp16 LN 数值问题）


## 5. 想快速验证你的 idea：建议的阅读顺序（最省时间）

1. 先看训练/测试入口理解“跑起来”需要什么：`train.py`、`test.py`、`utils/options.py`、`run_irra.sh`
2. 看数据长什么样、batch 里有哪些 key：`datasets/build.py`、`datasets/bases.py`
3. 看模型 forward 输出哪些 loss：`model/build.py`、`model/objectives.py`
4. 看 CLIP/ViT 的适配与输出张量形状：`model/clip_model.py`
5. 看训练循环如何聚合 loss、评估、保存 best：`processor/processor.py`、`utils/metrics.py`、`utils/checkpoint.py`
6. 最后再看 sampler/DDP 工具：`datasets/sampler.py`、`utils/comm.py`、`datasets/sampler_ddp.py`


## 6. 二次开发“改哪里最划算”（按目标划分）

### 6.1 加一个新 loss（最常见的快速实验）
- 在 `model/objectives.py` 新增 `compute_xxx()`
- 在 `model/build.py:IRRA.forward()` 里按 `loss_names` 开关把它加进 `ret`
- 在 `utils/options.py` 给 `loss_names` 增加新名字（或直接用字符串匹配）

### 6.2 加一个新模块（例如更强的跨模态交互）
- 模块定义建议放在 `model/build.py:IRRA.__init__()`
- 训练时的交互路径写在 `forward()` 的某个 task 分支里（保持推理不增加 cost 可继续沿用“只用 CLS/EOT”）
- 若需要更大学习率，沿用 `solver/build.py` 的 key 规则（例如让参数名包含 `cross`）

### 6.3 改 backbone（例如换 CLIP ViT-L/14 或 ResNet-CLIP）
- 改 `args.pretrain_choice`
- 重点检查：
  - 输入分辨率是否需要改（`args.img_size`）
  - stride 是否匹配（`args.stride_size`）
  - 显存/速度与 fp16 是否兼容

### 6.4 改 text_length / vocab / tokenizer
- `datasets/bases.py:tokenize()` 会 pad 到 `args.text_length`
- **如果你修改 `args.text_length != 77`**：目前 `resize_text_pos_embed` 缺失，需要补实现或避免触发（例如保持 77）。


## 7. 常见“读代码时容易踩的坑”

- `CLIP.encode_text()` 返回的是 **所有 token 的 embedding**，不是最终 pooled 向量；最终 pooled 是在 `IRRA.encode_text()` / `IRRA.forward()` 里用 EOT index 取出来。
- `convert_weights()` 把很多模块转 half；如果你新加模块，注意 LayerNorm/Softmax 的 dtype 与稳定性。
- identity sampler 的 DDP 版本在 `datasets/build.py` 里标注了 TODO（默认训练命令未走这条路）。


## 8. 你可以如何验证我对调用链的理解

- 训练：直接跑 `run_irra.sh`（或 `python train.py ...`）应该会在 `logs/<dataset>/<time>_<name>/` 生成 `configs.yaml` 和 `train_log.txt`，并在评估提升时保存 `best.pth`。
- 测试：用 `python test.py --config_file <上述目录>/configs.yaml`，应加载 `best.pth` 并打印 R1/R5/R10/mAP/mINP。

