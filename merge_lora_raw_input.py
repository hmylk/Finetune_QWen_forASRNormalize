import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

# ====================== 1. 基础配置 ======================
# 基础模型路径/名称
BASE_MODEL_NAME = "/data/nfs_rt18/hongmi/pretrained_model/Qwen2.5-0.5B"
# 训练好的 LoRA 适配器路径（即训练脚本里的 LORA_OUTPUT_DIR）
LORA_MODEL_DIR = "./qwen2.5-0.5b-lora-finetune3"
# 合并后模型的保存路径
MERGED_MODEL_DIR = "./qwen2.5-0.5b-merged"

# ====================== 2. 加载基础模型 + LoRA 适配器 ======================
print("加载基础模型...")
# 加载基础模型（与微调时保持一致的配置）
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # 保持半精度，与微调一致
    device_map="cpu",          # 自动分配设备
    low_cpu_mem_usage=True     # 降低 CPU 内存占用
)

print("加载 LoRA 适配器...")
# 加载 LoRA 适配器并绑定到基础模型
lora_model = PeftModel.from_pretrained(
    base_model,
    LORA_MODEL_DIR,
    torch_dtype=torch.float16
)

# ====================== 3. 合并 LoRA 与基础模型 ======================
print("开始合并模型...")
# 核心：合并 LoRA 权重到基础模型（merge_and_unload）
merged_model = lora_model.merge_and_unload()

# 验证：合并后模型不再是 PeftModel，而是原生的 CausalLM 模型
print(f"合并后模型类型：{type(merged_model)}")  # 应输出 <class 'qwen.modeling_qwen.QwenForCausalLM'>

# ====================== 4. 保存合并后的完整模型 ======================
print("保存合并后的模型...")
# 保存模型权重
merged_model.save_pretrained(
    MERGED_MODEL_DIR,
    safe_serialization=True,  # 安全序列化，兼容不同版本
    max_shard_size="2GB"      # 分片保存（0.5B 模型可忽略，大模型建议设置）
)

# 保存 tokenizer（与合并模型配套）
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_NAME,
    trust_remote_code=True,
    padding_side="right",
    use_fast=False
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.save_pretrained(MERGED_MODEL_DIR)

print(f"合并后的模型已保存至：{MERGED_MODEL_DIR}")




# ====================== 5. 验证合并后模型的推理效果 ======================
def infer_merged_model(instruction, max_new_tokens=50):
    """使用合并后的独立模型推理"""
    # 加载合并后的模型（无需加载 LoRA 适配器）
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_DIR, trust_remote_code=True)
    
    prompt = f"{instruction.strip()}\n"

    # 推理时的 prompt 需要和训练格式保持一致
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=125,
        padding=True
    ).to("cpu")
    
    # 推理
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # 只截取新生成的部分，避免把输入 prompt 一起打印出来
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response

if os.environ.get("RUN_MERGED_TEST", "1") == "1":
    print("\n验证合并后模型推理效果：")
    test_instruction = os.environ.get("TEST_TEXT", "嗯,那个我那个那个,团建的吧,那个邮件收到了吧啊?")
    response = infer_merged_model(test_instruction)
    print(f"指令：{test_instruction}")
    print(f"合并模型输出：{response}")
