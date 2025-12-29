from transformers import AutoProcessor, AutoModelForSeq2SeqLM
import os

# 1. 之前设置的缓存目录
my_cache_dir = "./T5_1B_model_cache"
model_id = "google/t5gemma-2-1b-1b" # 保持和你之前下载时一致的ID

print("正在从缓存加载模型...")
# 从缓存加载
processor = AutoProcessor.from_pretrained(model_id, cache_dir=my_cache_dir, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, cache_dir=my_cache_dir, local_files_only=True)

# 2. 定义一个干净的本地目录用来做微调的基础
save_directory = "./T5_1B_model_Base"

print(f"正在将模型转存到: {save_directory} ...")
# 这一步会把 config.json, model.safetensors 等文件整整齐齐地存到新目录
model.save_pretrained(save_directory)
processor.save_pretrained(save_directory)

print("转存完成！你可以删除 T5_model_cache 文件夹了（如果不再需要其他模型的话）。")
