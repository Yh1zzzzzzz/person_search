import torch
from transformers import AutoProcessor, AutoModelForSeq2SeqLM
from PIL import Image
import os

# 配置路径
model_270M_path = "./T5_270M_Base"
model_1B_path = "./T5_1B_model_Base"
image_path = "./0107002.png"
# Ground Truth (真实描述)
ground_truth = "The man has short, dark hair and wears khaki pants with an oversized grey hoodie. His black backpack hangs from one shoulder."
# 测试的 Prompt 列表
prompts_categories = {
    "A. 通用场景类": [
        "A photo of",
        "An image of",
        "A picture showing",
        "In this image, there is",
        "A view of"
    ],
    "B. 人物主体类 (ReID 核心)": [
        "The person is",
        "The person is wearing",
        "This person is carrying",
        "A pedestrian who is",
        "The pedestrian is wearing"
    ],
    "C. 结构化/元数据类": [
        "Caption:",
        "Description:",
        "Visual description:",
        "Attributes:",
        "Summary:"
    ],
    "D. 细粒度引导类": [
        "The color of the upper clothing is",
        "The person has",
        "Looking at the person's clothes,",
        "Upper body:",
        "Full body shot of"
    ]
}

def load_model_and_processor(path):
    print(f"正在加载模型: {path} ...")
    try:
        processor = AutoProcessor.from_pretrained(path)
        model = AutoModelForSeq2SeqLM.from_pretrained(path)
        return processor, model
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None, None

def calculate_loss(model, processor, image, prompt, ground_truth, device):
    """计算给定 Prompt 下生成 Ground Truth 的 Loss"""
    # 准备输入 (Encoder Inputs: Image + Prompt)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    
    # 准备标签 (Decoder Targets: Ground Truth)
    # 使用 tokenizer 编码 GT，作为 labels
    labels = processor.tokenizer(text=ground_truth, return_tensors="pt").input_ids.to(device)
    
    # 运行模型前向传播计算 Loss
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss.item()
        
    return loss

def test_prompts():
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"错误: 找不到测试图片 {image_path}")
        return

    image = Image.open(image_path)
    
    # 定义要测试的模型列表
    models_to_test = [
        ("270M Model", model_270M_path),
        ("1B Model", model_1B_path)
    ]

    device =  "cpu"
    print(f"运行设备: {device}")

    for model_name, model_path in models_to_test:
        print(f"\n{'='*20} 开始测试 {model_name} {'='*20}")
        
        processor, model = load_model_and_processor(model_path)
        if model is None:
            continue
            
        model.to(device)
        model.eval()

        print(f"Ground Truth: {ground_truth}\n")

        for category, prompts in prompts_categories.items():
            print(f"\n>>> 类别: {category}")
            print(f"{'Prompt':<35} | {'Loss':<8} | {'Generated Output'}")
            print("-" * 110)

            for prompt in prompts:
                prompt = "<start_of_image> " + prompt.strip()
                # 1. 计算 Loss (越低越好)
                loss = calculate_loss(model, processor, image, prompt, ground_truth, device)

                # 2. 生成文本 (直观对比)
                inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
                
                # 生成
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs, 
                        max_new_tokens=50, 
                        do_sample=False # 使用贪婪搜索以获得稳定结果
                    )
                
                # 解码
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # 打印结果 (格式化对齐)
                print(f"{prompt:<35} | {loss:.4f}   | {generated_text}")

        # 清理显存
        del model
        del processor
        torch.cuda.empty_cache()

if __name__ == "__main__":
    test_prompts()
