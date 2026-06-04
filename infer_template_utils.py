from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


PROMPT_TEMPLATE = """删除文本中的语气词、口头禅、重复词，不改变原意、不调整语序、不添加内容。
输入：{text}
输出："""


def load_lora_model(adapter_path="./asr_formal_lora", device="cpu", torch_dtype=None):
    config = PeftConfig.from_pretrained(adapter_path)

    if torch_dtype is None:
        torch_dtype = torch.float32 if device == "cpu" else torch.float16

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "device_map": device,
    }

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
    # Decoder-only 模型做批量生成时应使用 left padding，
    # 否则不同 batch size 下右侧 padding 会影响最后一个位置的生成结果。
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return tokenizer, model


def generate_texts(
    model,
    tokenizer,
    texts,
    max_input_length=125,
    max_new_tokens=64,
):
    prompts = [PROMPT_TEMPLATE.format(text=text).strip() for text in texts]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # 对 left padding 的 decoder-only 模型，生成结果应从统一的 padded prompt 长度之后截取。
    # 如果按 attention_mask 的有效长度截取，会把一部分输入 token 当成生成结果，
    # 从而导致 batch_size=1 和 batch_size>1 的输出不一致。
    prompt_length = inputs["input_ids"].shape[1]
    generated = outputs[:, prompt_length:]
    predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return [prediction.strip() for prediction in predictions]
