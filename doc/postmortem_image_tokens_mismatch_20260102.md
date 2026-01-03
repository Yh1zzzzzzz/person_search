# 复盘：T5Gemma2 图像占位 token 与图像特征不匹配（2026-01-02）

## 背景
- 任务：多模态 Person Search（图文检索 + 文本生成）
- 数据：行人图像统一为 **384×128**（约十多万张）
- 模型：T5Gemma2（encoder 侧含 SigLIP 视觉塔 + 多模态 projector，文本 hidden size=640，视觉 hidden size=1152）

## 现象（Symptoms）
运行快速自检脚本时前向报错：

```
ValueError: Image features and image tokens do not match: tokens: 14, features 14
```

堆栈指向：`transformers.models.t5gemma2.modeling_t5gemma2.T5Gemma2Encoder` 内部 `preprocess_image_features -> get_image_placeholder_mask`。

## 影响（Impact）
- 无法完成 `encode_image_only()` 前向，导致检索分支与生成分支均被阻断。
- 训练/评测无法启动，属于硬阻塞问题。

## 快速结论（TL;DR）
- 报错信息里虽然打印 **tokens=14, features=14**，但真正检查的是 **占位 token 对应的 embedding 总元素数** 与 **image_features 总元素数**。
- 你的 monkey patch 返回的图像特征最后一维仍为 **1152**（视觉维度），但文本 embedding 维度是 **640**，因此 `numel()` 不相等而报错。

## 根因（Root Cause）
### 1) Transformers 的真实校验逻辑
在 `get_image_placeholder_mask` 内部：
- `special_image_mask = (input_ids == image_token_id)`
- `n_image_tokens = special_image_mask.sum()`（报错信息里显示的 tokens）
- `n_image_features = image_features.shape[0] * image_features.shape[1]`（报错信息里显示的 features）
- 但最终判断使用的是：
  - `inputs_embeds[special_image_mask].numel()`
  - `image_features.numel()`

因此：即使 `n_image_tokens == n_image_features`，只要最后一维维度不一致（640 vs 1152），依然会失败。

### 2) Monkey patch 漏了关键投影步骤
HF 原版 `get_image_features` 做的是：
1. `vision_tower(pixel_values).last_hidden_state` 得到 `(B, N, 1152)`
2. `multi_modal_projector(vision_outputs)`：
   - reshape 成二维网格
   - avg_pool（kernel/stride 通常为 4）
   - norm
   - **再用 `mm_input_projection_weight` 做线性投影：1152 -> 640**

你重写后的实现只做了 reshape/pool/norm，漏掉了最后的 **matmul 投影到 640**。

## 修复（Fix）
修改位置：`model/modeling_ps_t5gemma2.py` 的 monkey patch `dynamic_get_image_features`。

修复要点：
- 在 norm 后补上：
  - `x = torch.matmul(x, self_encoder.multi_modal_projector.mm_input_projection_weight)`
  - 确保输出形状为 `(B, image_len, 640)`
- 增加对可能存在 CLS token 的健壮处理：
  - 若检测到 `N == H_grid*W_grid + 1`，则丢弃首 token 再 reshape。

## 验证（Verification）
执行：
- `python ./quick_check_t5gemma2.py`

结果：
- 前向通过
- 输出 `sdm_loss / id_loss / gen_loss / total_loss` 正常
- 说明：
  - 图像占位 token 数与图像特征 token 数一致
  - 最后一维维度（640）与文本 embedding 对齐

## 关于 384×128 数据：训练时选哪种方案更好？
这里的“性能”分两部分：
- **效果（accuracy / mAP / R@K）**
- **训练效率（吞吐、显存、时长）**

### 方案 A：统一 resize 到 (392, 112)，使用 14 个图像 tokens（7×2）
- 优点：
  - token 数略多（14 vs 12），图像信息更密一点
  - 与当前 quick_check 的配置一致，流程简单
- 缺点：
  - **会改变长宽比**：384×128（3:1）-> 392×112（3.5:1），存在几何形变
  - 宽度从 128 变 112，细粒度横向信息（衣服边缘/包/手部）可能更易损失
  - 额外算力：image tokens 增加约 16.7%（12->14），encoder 自注意力相关开销随序列长度略增

### 方案 B：保持原始 (384, 128)，使用 12 个图像 tokens（6×2）
- 优点：
  - **不改变长宽比**，对行人检索通常更稳
  - 输入分辨率与数据天然一致，减少预处理“分布偏移”
  - token 数更少，训练略省显存/更快
- 缺点：
  - token 数更少，信息汇聚更强（可能轻微影响上限）

### 推荐（在你的场景更倾向）
- 如果目标是 **检索效果优先且数据全是 384×128**：更推荐 **方案 B（384×128 + 12 tokens）**。
  - 原因：避免长宽比形变通常比“多 2 个 tokens”更关键，尤其是 person re-id / person search 这类强依赖几何结构的任务。
- 如果你更在意 **实现简单、尽快跑通、且后续会混入多分辨率/多长宽比**：选 **方案 A（392×112 + 14 tokens）**也可，但要接受形变带来的潜在损失。

> 实务建议：如果你愿意做一次小规模对比实验（例如 10k steps），用同等 batch/学习率分别跑 A/B，观测 R@1 与收敛速度，最终用数据说话。

## 防回归建议（Action Items）
1. **强制一致的分辨率策略**：训练/评测/抽特征统一使用同一套 `TARGET_H/TARGET_W`。
2. **在 get_image_features 输出处加断言**（可选）：确保 `image_features.shape[-1] == encoder.embed_tokens.embedding_dim`，尽早失败、减少排查成本。
3. **把 tokens 计算写成“由输入分辨率推导”**：避免手工维护 `mm_tokens_per_image` 与实际特征数量漂移。

---

## 附：此次问题的一句话教训
报错信息里看到 “tokens=14, features=14” 不代表真的对齐；要检查的是 **(token 数 × hidden size)** 与 **(feature 数 × hidden size)** 是否一致。
