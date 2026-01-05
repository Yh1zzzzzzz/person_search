# Load model directly
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
# 使用项目根目录下的测试图像
img_path = "./test_image.png"
img = Image.open(img_path).convert("RGB")
print(f"已加载图像: {img_path}")

# 从本地加载模型
model_id = "./Qwen3-VL-2B-Instruct_Base"
print(f"正在从 {model_id} 加载模型...")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(model_id, trust_remote_code=True)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "Person Caption:"},
        ]
    },
]
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"模型已移动到设备: {device}")
inputs = processor.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
