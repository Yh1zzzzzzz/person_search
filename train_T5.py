from transformers import AutoProcessor, AutoModelForSeq2SeqLM
import requests
from PIL import Image
# 指向刚才转存的干净目录
model_270M  = "./T5_270M_Base"
model_1B = "./T5_1B_model_Base"
# 加载模型准备微调
processor_270M = AutoProcessor.from_pretrained(model_270M)
model_270M = AutoModelForSeq2SeqLM.from_pretrained(model_270M)
processor_1B = AutoProcessor.from_pretrained(model_1B)
model_1B = AutoModelForSeq2SeqLM.from_pretrained(model_1B)


image = Image.open("./test_image.png")


# prompt = "<start_of_image>Caption: The person is wearing"
prompt = "<start_of_image> The person carries a backpack which is"


model_inputs_270M = processor_270M(text=prompt, images=image, return_tensors="pt")
model_inputs_1B = processor_1B(text=prompt, images=image, return_tensors="pt")
# 注意：如果有 GPU，建议加上 .to("cuda")
# model.to("cuda")
# model_inputs = model_inputs.to("cuda")
generation_270M = model_270M.generate(**model_inputs_270M, max_new_tokens=50, do_sample=False,repetition_penalty=1.2)
generation_1B = model_1B.generate(**model_inputs_1B, max_new_tokens=50, do_sample=False,repetition_penalty=1.2)

print("270M model output:")
print(processor_270M.decode(generation_270M[0]))
print("==" * 20)
print("==" * 20)
print("\n1B model output:")
print(processor_1B.decode(generation_1B[0]))
# # ... 接下来接你的 Dataset 加载和 Trainer 代码 ...
# # 例如使用 PEFT/LoRA (推荐，因为全量微调比较吃显存)
# from peft import LoraConfig, get_peft_model, TaskType

# # 给微调做准备的简单示例
# peft_config = LoraConfig(
#     task_type=TaskType.SEQ_2_SEQ_LM, 
#     inference_mode=False, 
#     r=8, 
#     lora_alpha=32, 
#     lora_dropout=0.1
# )

# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()

# 开始训练...
