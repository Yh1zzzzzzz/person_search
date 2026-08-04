# GRAM

Official implementation of **Beyond Contrastive: Generative Reranking for
Multilingual Text-Based Person Retrieval**.

## Setup

```bash
conda create -n gram python=3.11 -y
conda activate gram
pip install -r requirements.txt
huggingface-cli login
```

Access to [`google/t5gemma-2-270m-270m`](https://huggingface.co/google/t5gemma-2-270m-270m)
is required.

## Data

```bash
python tools/prepare_manifest.py \
  --dataset CUHK-PEDES \
  --root /path/to/datasets \
  --output data/cuhk_pedes.jsonl
```

## Train

```bash
python train.py --config configs/stage1_cuhk.json
python train.py --config configs/stage2_mpedes.json
```

## Evaluate

```bash
python evaluate.py \
  --manifest data/cuhk_pedes.jsonl \
  --checkpoint outputs/gram_stage1_cuhk/best.pt \
  --rerank \
  --vision-cache-device cuda
```

The gallery visual features are cached and reused during Top-K reranking.

## Citation

```bibtex
@inproceedings{yang2026gram,
  author    = {Yang, Haotian and Peng, Cheng and Wang, Haobo and Zhu, Zhen and Tang, Xiu and Yuan, Gongsheng and Xie, Zhongle and Wu, Sai and Wang, Weiqiang and Cheng, Yu},
  title     = {Beyond Contrastive: Generative Reranking for Multilingual Text-Based Person Retrieval},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  series    = {MM '26},
  year      = {2026},
  month     = nov,
  location  = {Rio de Janeiro, Brazil},
  doi       = {10.1145/3767308.3835886},
  isbn      = {979-8-4007-2213-4/2026/11},
  publisher = {Association for Computing Machinery},
  license   = {CC BY 4.0}
}
```
