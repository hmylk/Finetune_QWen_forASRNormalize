# Qwen2.5 ASR 口语归一化微调说明

本目录包含两套 LoRA 微调方案，用于把 ASR 口语文本转换成更规范的书面文本。

## 两套训练方案

### 1. Template 版本

- 训练脚本：`train_template.py`
- 推理工具：`infer_template_utils.py`
- 单句推理：`infer_template.py`
- 合并与测试：`merge_lora_template.py`
- LoRA 输出目录：`asr_formal_lora3/`

这套方案会把任务描述和输入文本一起喂给模型。

训练 / 推理 prompt 格式：

```text
删除文本中的语气词、口头禅、重复词，不改变原意、不调整语序、不添加内容。
输入：{text}
输出：
```

### 2. Raw Input 版本

- 训练脚本：`train_raw_input.py`
- 合并与测试：`merge_lora_raw_input.py`
- LoRA 输出目录：`qwen2.5-0.5b-lora-finetune3/`
- 合并模型目录：`qwen2.5-0.5b-merged/`

这套方案只把原始文本作为输入，不再显式拼接任务描述。

训练 / 推理 prompt 格式：

```text
{input}
```

实际实现里会在末尾额外补一个换行符后再生成。

## 训练数据格式

训练数据文件是一个 JSON 数组，每个元素是一条训练样本，格式如下：

```json
{
  "instruction": "删除文本中的语气词、口头禅、重复词，不改变原意、不调整语序、不添加内容。",
  "input": "嗯,那个我那个那个,团建的吧,那个邮件收到了吧啊?",
  "output": "那个团建的邮件收到了吧？"
}
```

字段说明：

- `instruction`：任务说明，`train_template.py` 会使用这个字段
- `input`：原始 ASR 风格口语文本
- `output`：目标书面化文本

### Template 版本如何使用数据

`train_template.py` 实际看到的输入是：

```text
{instruction}
输入：{input}
输出：
```

然后让模型生成 `output`。

### Raw Input 版本如何使用数据

`train_raw_input.py` 实际只使用：

```text
{input}
```

然后让模型生成 `output`。

## 当前数据文件

- `merged_asr_instruction_train_formalized.json`：训练集
- `merged_asr_instruction_test.json`：测试集 / 评测集

## 评测与单句推理

使用 `eval_lora.py` 时，需要根据模型类型指定对应的 prompt 风格。

### 评测 Template 版本

```bash
python eval_lora.py --adapter-path ./asr_formal_lora3 --prompt-style template
```

### 评测 Raw Input 版本

```bash
python eval_lora.py --adapter-path ./qwen2.5-0.5b-lora-finetune3 --prompt-style raw_input
```

### 单句推理 Template 版本

```bash
python eval_lora.py \
  --adapter-path ./asr_formal_lora3 \
  --prompt-style template \
  --text "嗯,那个我那个那个,团建的吧,那个邮件收到了吧啊?"
```

### 单句推理 Raw Input 版本

```bash
python eval_lora.py \
  --adapter-path ./qwen2.5-0.5b-lora-finetune3 \
  --prompt-style raw_input \
  --text "嗯,那个我那个那个,团建的吧,那个邮件收到了吧啊?"
```

## GGUF 导出

```bash
python export_to_llamacpp.py
```

## 多卡训练

### Raw Input 版本

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_raw_input.py
```

### Template 版本

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_template.py
```

## 一键流程脚本

### Raw Input 流程

脚本：`run_train.sh`

包含三个阶段：

- Stage 1：训练 LoRA，调用 `train_raw_input.py`
- Stage 2：合并 LoRA 并做简单推理，调用 `merge_lora_raw_input.py`
- Stage 3：导出 GGUF，调用 `export_to_llamacpp.py`

运行方式：

```bash
bash run_train.sh
```

如果想改执行阶段、GPU、测试句子、日志目录，直接修改脚本顶部的配置区：

- `STAGE`
- `STOP_STAGE`
- `GPU_LIST`
- `TEST_TEXT`
- `LOG_DIR`

日志默认写入 `./logs/`。

查看日志：

```bash
tail -f logs/raw_input_stage1_*.log
```

### Template 流程

脚本：`run_train_template.sh`

包含三个阶段：

- Stage 1：训练 LoRA，调用 `train_template.py`
- Stage 2：合并 LoRA 并做简单推理，调用 `merge_lora_template.py`
- Stage 3：从 `asr_formal_merged` 导出 GGUF

运行方式：

```bash
bash run_train_template.sh
```

如果想改执行阶段、GPU、测试句子、日志目录，直接修改脚本顶部的配置区：

- `STAGE`
- `STOP_STAGE`
- `GPU_LIST`
- `TEST_TEXT`
- `LOG_DIR`

日志默认写入 `./logs/`。
