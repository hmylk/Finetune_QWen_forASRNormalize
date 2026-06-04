import argparse
import json
from pathlib import Path

import torch

from infer_template_utils import load_lora_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LoRA inference results.")
    parser.add_argument("--adapter-path", default="./asr_formal_lora3", help="LoRA adapter directory")
    parser.add_argument("--text", default="", help="Run inference on a single input text")
    parser.add_argument(
        "--data-path",
        default="./merged_asr_instruction_test.json",
        help="Evaluation dataset path",
    )
    parser.add_argument("--output-dir", default="./eval_outputs_old", help="Directory for reports")
    parser.add_argument("--max-samples", type=int, default=0, help="Only evaluate first N samples, 0 means all")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--max-input-length", type=int, default=125, help="Max prompt length")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max generated tokens")
    parser.add_argument(
        "--prompt-style",
        choices=["template", "raw_input"],
        default="raw_input",
        help="Inference prompt format. Use raw_input for train_transformer2 style models.",
    )
    return parser.parse_args()


def load_data(data_path, max_samples):
    data = json.loads(Path(data_path).read_text(encoding="utf-8"))
    valid = [item for item in data if item.get("input") and item.get("output")]
    if max_samples > 0:
        valid = valid[:max_samples]
    return valid


def lcs_length(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ch_a in a:
        curr = [0]
        for j, ch_b in enumerate(b, start=1):
            if ch_a == ch_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(curr[-1], prev[j]))
        prev = curr
    return prev[-1]


def rouge_l_f1(reference, prediction):
    if not reference and not prediction:
        return 1.0
    if not reference or not prediction:
        return 0.0
    ref_chars = list(reference)
    pred_chars = list(prediction)
    lcs = lcs_length(ref_chars, pred_chars)
    precision = lcs / len(pred_chars)
    recall = lcs / len(ref_chars)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, start=1):
        curr = [i]
        for j, ch_b in enumerate(b, start=1):
            cost = 0 if ch_a == ch_b else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def cer(reference, prediction):
    if not reference and not prediction:
        return 0.0
    if not reference:
        return 1.0
    return edit_distance(list(reference), list(prediction)) / len(reference)


def build_model(adapter_path):
    return load_lora_model(adapter_path=adapter_path, device="cpu")


def batched(iterable, batch_size):
    for start in range(0, len(iterable), batch_size):
        yield iterable[start:start + batch_size]


def build_prompts(samples, prompt_style):
    prompts = []
    for item in samples:
        input_text = item["input"].strip()
        if prompt_style == "template":
            instruction = item.get("instruction", "").strip()
            prompts.append(f"{instruction}\n输入：{input_text}\n输出：".strip())
        else:
            prompts.append(f"{input_text}\n")
    return prompts


def generate_batch(model, tokenizer, prompts, max_input_length, max_new_tokens):
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

    prompt_length = inputs["input_ids"].shape[1]
    generated = outputs[:, prompt_length:]
    predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return [prediction.strip() for prediction in predictions]


def infer_single_text(model, tokenizer, text, max_input_length, max_new_tokens, prompt_style):
    if prompt_style == "template":
        prompt = f"删除文本中的语气词、口头禅、重复词，不改变原意、不调整语序、不添加内容。\n输入：{text.strip()}\n输出："
    else:
        prompt = f"{text.strip()}\n"

    prediction = generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=[prompt],
        max_input_length=max_input_length,
        max_new_tokens=max_new_tokens,
    )[0]
    return prediction


def generate_predictions(model, tokenizer, samples, batch_size, max_input_length, max_new_tokens, prompt_style):
    results = []
    for batch in batched(samples, batch_size):
        prompts = build_prompts(batch, prompt_style)
        predictions = generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
        )
        if predictions:
            print(predictions[0])

        for item, prediction in zip(batch, predictions):
            pred = prediction.strip()
            ref = item["output"].strip()
            results.append(
                {
                    "instruction": item["instruction"],
                    "input": item["input"],
                    "prediction": pred,
                    "output": ref,
                    "exact_match": pred == ref,
                    "cer": cer(ref, pred),
                    "rouge_l": rouge_l_f1(ref, pred),
                    "prediction_length": len(pred),
                    "reference_length": len(ref),
                    "added_chars": max(0, len(pred) - len(ref)),
                }
            )
    return results


def summarize(results):
    total = len(results)
    exact_match = sum(item["exact_match"] for item in results) / total if total else 0.0
    avg_cer = sum(item["cer"] for item in results) / total if total else 0.0
    avg_rouge_l = sum(item["rouge_l"] for item in results) / total if total else 0.0
    avg_pred_len = sum(item["prediction_length"] for item in results) / total if total else 0.0
    avg_ref_len = sum(item["reference_length"] for item in results) / total if total else 0.0
    added_rate = sum(1 for item in results if item["added_chars"] > 0) / total if total else 0.0
    over_delete_rate = sum(1 for item in results if item["prediction_length"] < item["reference_length"]) / total if total else 0.0
    return {
        "samples": total,
        "exact_match": round(exact_match, 6),
        "cer": round(avg_cer, 6),
        "rouge_l": round(avg_rouge_l, 6),
        "avg_prediction_length": round(avg_pred_len, 4),
        "avg_reference_length": round(avg_ref_len, 4),
        "added_rate": round(added_rate, 6),
        "over_delete_rate": round(over_delete_rate, 6),
    }


def save_outputs(output_dir, summary, results):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_path = output_path / "report.json"
    predictions_path = output_path / "predictions.jsonl"
    errors_path = output_path / "errors.jsonl"

    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with predictions_path.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with errors_path.open("w", encoding="utf-8") as f:
        for item in results:
            if not item["exact_match"]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return report_path, predictions_path, errors_path


def main():
    args = parse_args()
    tokenizer, model = build_model(args.adapter_path)

    if args.text.strip():
        prediction = infer_single_text(
            model=model,
            tokenizer=tokenizer,
            text=args.text,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
            prompt_style=args.prompt_style,
        )
        print(prediction)
        return

    samples = load_data(args.data_path, args.max_samples)
    results = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        prompt_style=args.prompt_style,
    )
    summary = summarize(results)
    report_path, predictions_path, errors_path = save_outputs(args.output_dir, summary, results)

    print("评测完成")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"report: {report_path}")
    print(f"predictions: {predictions_path}")
    print(f"errors: {errors_path}")


if __name__ == "__main__":
    main()
