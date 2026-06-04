# Qwen2.5 Fine-tuning Scripts

## Template version

- Train: `train_template.py`
- Inference utils: `infer_template_utils.py`
- Quick inference: `infer_template.py`
- Merge and test: `merge_lora_template.py`
- Adapter dir: `asr_formal_lora3/`

Prompt format:

```text
删除文本中的语气词、口头禅、重复词，不改变原意、不调整语序、不添加内容。
输入：{text}
输出：
```

## Raw input version

- Train: `train_raw_input.py`
- Merge and test: `merge_lora_raw_input.py`
- Adapter dir: `qwen2.5-0.5b-lora-finetune3/`
- Merged model dir: `qwen2.5-0.5b-merged/`

Prompt format:

```text
{input}
```

The implementation appends a trailing newline before generation.

## Evaluation

Use `eval_lora.py` with the right prompt style:

```bash
python eval_lora.py --adapter-path ./asr_formal_lora3 --prompt-style template
python eval_lora.py --adapter-path ./qwen2.5-0.5b-lora-finetune3 --prompt-style raw_input
```

## GGUF export

```bash
python export_to_llamacpp.py
```

## Multi-GPU training

Raw-input version:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_raw_input.py
```

Template version:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_template.py
```

Or run the shell script:

```bash
bash run_train.sh
```

## End-to-end pipeline

`run_train.sh` uses three stages for the raw-input pipeline:

- Stage 1: train LoRA with `train_raw_input.py`
- Stage 2: merge LoRA and run simple inference with `merge_lora_raw_input.py`
- Stage 3: export GGUF with `export_to_llamacpp.py`

Run all stages:

```bash
bash run_train.sh
```

Edit the config block at the top of `run_train.sh` to change:

- `STAGE`
- `STOP_STAGE`
- `GPU_LIST`
- `TEST_TEXT`
- `LOG_DIR`

With log file:

```bash
bash run_train.sh
tail -f logs/raw_input_stage1_*.log
```

Template pipeline:

- Stage 1: train LoRA with `train_template.py`
- Stage 2: merge LoRA and run simple inference with `merge_lora_template.py`
- Stage 3: export GGUF from `asr_formal_merged`

```bash
bash run_train_template.sh
```

Edit the config block at the top of `run_train_template.sh` to change:

- `STAGE`
- `STOP_STAGE`
- `GPU_LIST`
- `TEST_TEXT`
- `LOG_DIR`

Both pipeline scripts write logs to `./logs/` by default. Change `LOG_DIR` in the script if needed.
