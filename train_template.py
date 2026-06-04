import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model

# ===================== 1. 配置项（直接改这里）=====================
MODEL_NAME = "/data/nfs_rt18/hongmi/pretrained_model/Qwen2.5-0.5B"

#DATA_PATH = "deepseek_asr_instruction_dataset.json"         # 你的训练数据
DATA_PATH = "merged_asr_instruction_train_formalized.json"         # 你的训练数据
OUTPUT_DIR = "./asr_formal_lora"           # 模型输出路径
MAX_LENGTH = 512                           # 文本最大长度
BATCH_SIZE = 20                             # 批次大小
EPOCHS = 3                                 # 训练轮数

# ===================== 2. 加载模型 & 分词器 =====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# ===================== 3. LoRA 配置（ASR专用最优参数）=====================
lora_config = LoraConfig(
    r=8,                     # 秩（越大越强，越占显存）
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 适配Qwen/GLM
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 查看可训练参数（通常仅1%-3%）

# ===================== 4. 数据预处理 =====================
def format_prompt(sample):
    return f"""
{sample['instruction']}
输入：{sample['input']}
输出：
""".strip()

def process_func(sample):
    prompt = format_prompt(sample)
    answer = sample["output"] + tokenizer.eos_token

    # Causal LM 训练时需要把 prompt 和 answer 拼成一个序列，
    # 并将 prompt 部分的 labels 置为 -100，只在 answer 部分计算损失。
    full_text = prompt + answer
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_encoded = tokenizer(full_text, max_length=MAX_LENGTH, truncation=True)

    labels = full_encoded["input_ids"].copy()
    prompt_len = min(len(prompt_ids), MAX_LENGTH)
    labels[:prompt_len] = [-100] * prompt_len

    full_encoded["labels"] = labels
    return full_encoded

# 加载数据
dataset = load_dataset("json", data_files=DATA_PATH)
dataset = dataset["train"].map(process_func, remove_columns=dataset["train"].column_names)

# ===================== 5. 训练参数 =====================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    num_train_epochs=EPOCHS,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    optim="adamw_torch",
    fp16=True,
    bf16=False,
    gradient_accumulation_steps=4,
    remove_unused_columns=False,
    report_to="none",
    local_rank=int(os.environ.get("LOCAL_RANK", -1)),
    ddp_find_unused_parameters=False,
    ddp_bucket_cap_mb=25,
)

# 训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")
)

# ===================== 6. 启动训练 =====================
trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ ASR书面化 LoRA 模型训练完成！")
