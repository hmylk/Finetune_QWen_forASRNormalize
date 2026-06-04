import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL_NAME = "/data/nfs_rt18/hongmi/pretrained_model/Qwen2.5-0.5B"
LORA_MODEL_DIR = "./asr_formal_lora3"
MERGED_MODEL_DIR = "./asr_formal_merged"

PROMPT_TEMPLATE = """删除文本中的语气词、口头禅、重复词，不改变原意、不调整语序、不添加内容。
输入：{text}
输出："""


def infer_merged_model(text, max_new_tokens=50):
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = PROMPT_TEMPLATE.format(text=text).strip()
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=125,
        padding=True,
    ).to("cpu")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


print("加载基础模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)

print("加载 LoRA 适配器...")
lora_model = PeftModel.from_pretrained(
    base_model,
    LORA_MODEL_DIR,
    torch_dtype=torch.float16,
)

print("开始合并模型...")
merged_model = lora_model.merge_and_unload()

print("保存合并后的模型...")
merged_model.save_pretrained(
    MERGED_MODEL_DIR,
    safe_serialization=True,
    max_shard_size="2GB",
)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_NAME,
    trust_remote_code=True,
    padding_side="right",
    use_fast=False,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.save_pretrained(MERGED_MODEL_DIR)

print(f"合并后的模型已保存至：{MERGED_MODEL_DIR}")

if os.environ.get("RUN_MERGED_TEST", "1") == "1":
    print("\n验证合并后模型推理效果：")
    test_text = os.environ.get("TEST_TEXT", "嗯,那个我那个那个,团建的吧,那个邮件收到了吧啊?")
    response = infer_merged_model(test_text)
    print(f"输入：{test_text}")
    print(f"输出：{response}")
