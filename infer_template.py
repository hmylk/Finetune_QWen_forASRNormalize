from infer_template_utils import load_lora_model, generate_texts

# 加载训练好的LoRA模型
tokenizer, model = load_lora_model(adapter_path="./asr_formal_lora3", device="cpu")
model.eval()
print(model)
# ASR口语输入
asr_text = "嗯那个我觉得吧，明天我们一起去开会啊"
asr_text = "嗯,那个我那个那个,团建的吧,那个邮件收到了吧啊?"

result = generate_texts(
    model=model,
    tokenizer=tokenizer,
    texts=[asr_text],
    max_input_length=125,
    max_new_tokens=64,
)[0]
print("✅ 书面化结果：", result)
