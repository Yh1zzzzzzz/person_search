import requests
from PIL import Image
import os
from transformers import AutoProcessor, AutoModelForSeq2SeqLM

# -------------------------------------------------------
# 配置部分
# -------------------------------------------------------
# 1. 设置下载目录为数据盘 (AutoDL 的数据盘通常在 /root/autodl-tmp)
my_cache_dir = "./T5_model_cache"  # 可根据需要修改路径

# 2. (可选) 设置国内镜像加速，防止下载卡住
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 

# -------------------------------------------------------
# 加载模型 (自动下载到指定路径)
# -------------------------------------------------------
print(f"正在下载/加载模型到: {my_cache_dir} ...")

processor = AutoProcessor.from_pretrained(
    "google/t5gemma-2-270m-270m", 
    cache_dir=my_cache_dir  # <--- 关键参数：指定下载路径
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/t5gemma-2-270m-270m", 
    cache_dir=my_cache_dir  # <--- 关键参数：指定下载路径
)

# -------------------------------------------------------
# 推理代码 (保持不变)
# -------------------------------------------------------
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
image = Image.open(requests.get(url, stream=True).raw)
prompt = " in this image, there is"

model_inputs = processor(text=prompt, images=image, return_tensors="pt")

# 注意：如果有 GPU，建议加上 .to("cuda")
# model.to("cuda")
# model_inputs = model_inputs.to("cuda")

generation = model.generate(**model_inputs, max_new_tokens=20, do_sample=False)
print(processor.decode(generation[0]))