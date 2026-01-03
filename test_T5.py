import requests
from PIL import Image
import os
from transformers import AutoProcessor, AutoModelForSeq2SeqLM

local_path = "T5_270M_Base"  # 可根据需要修改路径



processor = AutoProcessor.from_pretrained(local_path)
model = AutoModelForSeq2SeqLM.from_pretrained(local_path)


image = Image.open("test_image.png")
prompt = "<start_of_image>"

model_inputs = processor(text=prompt, images=image, return_tensors="pt")

# 注意：如果有 GPU，建议加上 .to("cuda")
model.to("cuda")
model_inputs = model_inputs.to("cuda")

generation = model.generate(**model_inputs, max_new_tokens=30, do_sample=False)
print(processor.decode(generation[0]))
# print(model)