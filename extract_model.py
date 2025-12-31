from transformers import AutoProcessor, AutoModelForVision2Seq
import os

# 1. 之前设置的缓存目录
my_cache_dir = "./Qwen3-VL-2B-Instruct"
model_id = "Qwen/Qwen3-VL-2B-Instruct"  # 使用完整的模型标识

print("正在从缓存加载模型...")
# 从缓存加载 - Qwen3-VL 是视觉-语言模型，需要使用 AutoModelForVision2Seq
processor = AutoProcessor.from_pretrained(model_id, cache_dir=my_cache_dir, local_files_only=True)
model = AutoModelForVision2Seq.from_pretrained(model_id, cache_dir=my_cache_dir, local_files_only=True)

# 2. 定义一个干净的本地目录用来做微调的基础
save_directory = "./Qwen3-VL-2B-Instruct_Base"

print(f"正在将模型转存到: {save_directory} ...")
# 这一步会把 config.json, model.safetensors 等文件整整齐齐地存到新目录
model.save_pretrained(save_directory)
processor.save_pretrained(save_directory)

print("转存完成！模型已保存到:", save_directory)
