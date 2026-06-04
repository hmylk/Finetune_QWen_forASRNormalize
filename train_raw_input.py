import os
import torch
import json
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import warnings
warnings.filterwarnings("ignore")

# ====================== 1. 基础配置 ======================
MODEL_NAME = "/data/nfs_rt18/hongmi/pretrained_model/Qwen2.5-0.5B"
LORA_OUTPUT_DIR = "./qwen2.5-0.5b-lora-finetune3"
TRAIN_BATCH_SIZE = 5
EPOCHS = 10
LEARNING_RATE = 5e-5
MAX_SEQ_LENGTH = 125  # 根据显存调整
DATA_FILE_PATH = "merged_asr_instruction_train_formalized.json"  # 你的数据文件路径

# ====================== 2. 加载并预处理纯指令格式数据 ======================
# 加载数据（优先加载文件，无则用示例）
def load_custom_data(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    except:
        return None

raw_data = load_custom_data(DATA_FILE_PATH)
dataset = Dataset.from_list(raw_data)

# 加载 tokenizer（仅基础配置，无对话模板）
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side="right",
    use_fast=False
)
# 必须设置 pad_token（Qwen 默认为 None）
tokenizer.pad_token = tokenizer.eos_token

# 纯文本改写数据预处理：Prompt = input，拼接 output 作为训练文本
def preprocess_function(examples):
    # 核心：用 input 作为条件输入，只在 output 部分计算损失。
    inputs = examples["input"]
    outputs = examples["output"]
    
    # 2. 拼接 Prompt + 输出（无对话模板）
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    for inst, out in zip(inputs, outputs):
        # 确保是字符串（防止空值）
        inst_str = str(inst).strip() if inst else ""
        out_str = str(out).strip() if out else ""
        prompt_text = f"{inst_str}\n"
        full_text = f"{prompt_text}{out_str}{tokenizer.eos_token}"

        prompt_tokens = tokenizer(
            prompt_text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            add_special_tokens=False,
        )
        full_tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            return_attention_mask=True,
            add_special_tokens=False,
        )

        labels = full_tokens["input_ids"].copy()
        prompt_len = len(prompt_tokens["input_ids"])
        labels[:prompt_len] = [-100] * prompt_len
        labels = [label if mask == 1 else -100 for label, mask in zip(labels, full_tokens["attention_mask"])]

        input_ids_list.append(full_tokens["input_ids"])
        attention_mask_list.append(full_tokens["attention_mask"])
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }

# 应用预处理
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names,  # 删除原始字段，仅保留训练所需列
)

# 转换为 PyTorch 格式
tokenized_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

# ====================== 3. 模型与 LoRA 配置 ======================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # 半精度降低显存
    #device_map="auto"           # 自动分配设备
)

# LoRA 核心配置（仅训练少量参数）
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 纯因果语言模型任务
    r=8,                           # 低秩矩阵秩（越小参数越少）
    lora_alpha=32,                 # 缩放因子
    lora_dropout=0.05,             # Dropout 防止过拟合
    # Qwen2.5 核心可训练模块
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",                   # 不训练 bias 参数
    inference_mode=False           # 训练模式
)

# 绑定 LoRA 到主模型
peft_model = get_peft_model(model, lora_config)
print("可训练参数占比：")
peft_model.print_trainable_parameters()  # 约 0.1% 左右，显存占用极低

# ====================== 4. 训练配置与启动 ======================
training_args = TrainingArguments(
    output_dir=LORA_OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_dir=f"{LORA_OUTPUT_DIR}/logs",
    logging_steps=5,               # 每5步打印日志
    save_strategy="epoch",         # 按轮保存
    save_total_limit=1,            # 仅保留最新模型
    fp16=True,                     # 半精度训练（GPU 必备）
    gradient_accumulation_steps=4, # 梯度累积提升有效批次
    remove_unused_columns=False,
    report_to="none",              # 不使用 wandb 等工具
    lr_scheduler_type="cosine",    # 余弦学习率衰减
    warmup_steps=10,               # 预热步数
    weight_decay=0.01,             # 权重衰减防止过拟合
    max_grad_norm=1.0,             # 梯度裁剪防止爆炸
    # 自动设置分布式参数（Accelerate 会识别）
    local_rank=int(os.environ.get("LOCAL_RANK", -1)),  # 单卡时保持默认值 -1
    ddp_find_unused_parameters=False,  # 关闭未使用参数查找，提升速度
    ddp_bucket_cap_mb=25,  # DDP 通信桶大小，适配小模型
)

# 初始化 Trainer
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# 启动训练
trainer.train()

# 保存 LoRA 适配器（仅几 MB，无需保存完整模型)
peft_model.save_pretrained(LORA_OUTPUT_DIR)
tokenizer.save_pretrained(LORA_OUTPUT_DIR)

# # ====================== 5. 纯指令式推理（仅输入 instruction） ======================
# def infer_with_lora(instruction, max_new_tokens=50):
#     # 加载基础模型 + LoRA 适配器
#     base_model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME,
#         trust_remote_code=True,
#         torch_dtype=torch.float16,
#         device_map="auto"
#     )
#     lora_model = PeftModel.from_pretrained(base_model, LORA_OUTPUT_DIR)
    
#     # 仅用 instruction 作为输入 Prompt（无任何对话模板）
#     inputs = tokenizer(
#         instruction,
#         return_tensors="pt",
#         truncation=True,
#         max_length=MAX_SEQ_LENGTH,
#         padding=True
#     ).to("cuda" if torch.cuda.is_available() else "cpu")
    
#     # 推理（纯生成模式）
#     with torch.no_grad():
#         outputs = lora_model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,  # 生成最大长度
#             temperature=0.1,               # 低温度保证生成稳定
#             top_p=0.9,
#             do_sample=True,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.pad_token_id
#         )
    
#     # 解码并提取结果（仅保留生成部分，去除输入 Prompt）
#     full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     # 截取生成的部分（去掉输入的 instruction）
#     response = full_output[len(instruction):].strip()
#     return response

# # 测试推理（仅输入 instruction）
# test_instruction = "台湾啊的天气"
# response = infer_with_lora(test_instruction)
# print(f"指令（instruction）：{test_instruction}")
# print(f"模型输出：{response}")  # 预期输出：台湾的天气
