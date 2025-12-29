import requests
from PIL import Image
import os
from transformers import AutoProcessor, AutoModelForSeq2SeqLM

my_cache_dir = "./T5_1B_model_cache"  # 可根据需要修改路径

local_path = "./T5_model_cache/models--google--t5gemma-2-270m-270m/snapshots/7c38f16641f455ef0685b18431faf1b17722d5a1"


# -------------------------------------------------------
# 加载模型 (自动下载到指定路径)
# -------------------------------------------------------
print(f"正在下载/加载模型到: {my_cache_dir} ...")

processor = AutoProcessor.from_pretrained("google/t5gemma-2-1b-1b", cache_dir=my_cache_dir)
model = AutoModelForSeq2SeqLM.from_pretrained("google/t5gemma-2-1b-1b", cache_dir=my_cache_dir)


url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
image = Image.open(requests.get(url, stream=True).raw)
prompt = "<start_of_image> in this image, there is"

model_inputs = processor(text=prompt, images=image, return_tensors="pt")

# 注意：如果有 GPU，建议加上 .to("cuda")
# model.to("cuda")
# model_inputs = model_inputs.to("cuda")

generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
print(processor.decode(generation[0]))
