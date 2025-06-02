# -*- coding: utf-8 -*-
"""
本代码示例展示如何：
1. 加载 /data/new_knowledge_base.json 数据，并构造训练样本
2. 使用 Hugging Face Datasets 构建数据集，并进行分词和添加 labels
3. 划分训练集与验证集
4. 加载本地预训练模型 deepseek-llm-7b-base
5. 使用 PEFT 添加 LoRA 适配器
6. 自定义 Trainer 子类，重写 compute_loss 和 evaluation_step 方法，确保返回的 loss 可反向传播
7. 配置 Trainer 进行微调训练
"""

import json
from pathlib import Path
import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorWithPadding, BitsAndBytesConfig
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ----------------------------
# Step 0. 设置环境变量（可选）
# ----------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用 tokenizers 的并行化

model_path = "/mnt/data1/zhongrongmiao/InternLM/internlm2_5-1_8b-chat"
# ----------------------------
# Step 1. 加载 JSON 数据
# ----------------------------
json_path = Path("/mnt/data1/zhongrongmiao/InternLM/data/instruction_dataset_level_03_ft.json")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
# data 为字典，键为知识点，值为相关主题列表

# ----------------------------
# Step 2. 数据预处理：构建训练样本
# ----------------------------
# training_samples = []
# for key, values in data.items():
#     prompt = f"知识点: {key}"
#     answer = "; ".join(values)
#     training_samples.append({"prompt": prompt, "answer": answer})

# print("前3个训练样本：")
# for sample in training_samples[:3]:
#     print("Prompt:", sample["prompt"])
#     print("Answer:", sample["answer"])
#     print("=" * 40)

# ----------------------------
# Step 3. 构建 Hugging Face 数据集
# ----------------------------
dataset = Dataset.from_list(data)
print("\n数据集信息：")
print(dataset)

# ----------------------------
# Step 4. 数据 Tokenization
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def tokenize_function(batch):
    """
    对每个批次，将 prompt 和 answer 拼接成一段文本，再进行分词处理
    """
    texts = [p + "\n" + a for p, a in zip(batch["prompt"], batch["answer"])]
    tokenized_output = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=64  # 增加最大长度以更好地捕捉上下文
    )
    return tokenized_output

tokenized_dataset = dataset.map(tokenize_function, batched=True)
print("\n分词后的样本示例：")
print(tokenized_dataset[0])

# ----------------------------
# Step 5. 添加 labels 字段
# ----------------------------
def add_labels(example):
    example["labels"] = example["input_ids"].copy()
    return example

tokenized_dataset = tokenized_dataset.map(add_labels)
print("\n添加 labels 后的样本示例：")
print(tokenized_dataset[0])

# ----------------------------
# Step 6. 划分训练集与验证集
# ----------------------------
split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]
print("训练集样本数：", len(train_dataset))
print("验证集样本数：", len(eval_dataset))

# ----------------------------
# Step 7. 加载预训练模型与分词器
# ----------------------------


local_rank = int(os.getenv('LOCAL_RANK', -1))
if local_rank != -1:
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 限制每张GPU卡的显存使用
torch.cuda.set_per_process_memory_fraction(0.98, device=device.index)

# 配置 8 位量化参数
# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True,                # 开启 8 位加载
#     bnb_8bit_compute_dtype=torch.float16  # 计算时仍使用 FP16 精度
# )

# 加载模型时传入量化配置，并自动映射设备
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    # quantization_config=bnb_config,
    # device_map="auto"
)

# 准备模型用于 8 位训练
# model = prepare_model_for_kbit_training(model)

# 定义 LoRA 配置
lora_config = LoraConfig(
    r=8,                    # Rank of the low-rank matrices
    lora_alpha=32,          # Scaling factor for the learned weights
    # target_modules=["q_proj", "v_proj"],  # Target modules to apply LoRA
    lora_dropout=0.05,      # Dropout probability for the LoRA layers
    bias="none",            # Bias type for the LoRA layers
    task_type="CAUSAL_LM"   # Task type for the model
)

# lora_config = dict(
#     type=LoraConfig,
#     r=64,
#     lora_alpha=16,
#     lora_dropout=0.1,
#     bias='none',
#     task_type='CAUSAL_LM')

# 将 LoRA 适配器应用到模型
model = get_peft_model(model, lora_config)

# 打印模型结构以确认适配器已添加
print(model.print_trainable_parameters())

# ----------------------------
# Step 8. 自定义 Trainer 子类，重写 compute_loss 和 evaluation_step 方法
# ----------------------------
class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, **kwargs):
        outputs = model(**inputs)
        # 检查输出对象是否包含 loss 属性（例如 CausalLMOutputWithPast）
        if hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss
        elif isinstance(outputs, dict):
            loss = outputs.get("loss", None)
        elif isinstance(outputs, (list, tuple)):
            loss = outputs[0]
        else:
            loss = outputs

        if loss is None:
            raise ValueError("The model did not return a loss. Ensure that your inputs contain the correct labels.")
        if not loss.requires_grad:
            loss = loss.clone().detach().requires_grad_(True)
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            outputs = model(**inputs)

        # 同样提取 loss 张量，避免直接返回输出对象
        if hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss
            logits = outputs.logits if hasattr(outputs, "logits") else None
        elif isinstance(outputs, (list, tuple)):
            loss = outputs[0]
            logits = outputs[1:] if not prediction_loss_only else None
        else:
            loss = outputs
            logits = None

        labels = inputs.get("labels")
        return (loss, logits, labels)



# ----------------------------
# Step 9. 设置训练参数
# ----------------------------
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,  # 减少训练轮次以减少计算量
    per_device_train_batch_size=1,  # 继续减少批量大小
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,  # 增加梯度累积步数
    eval_strategy="epoch",  # 使用 eval_strategy 替代 evaluation_strategy
    save_strategy="epoch",
    fp16=True, 
    logging_steps=50,
    report_to="tensorboard",
    dataloader_num_workers=2,  # 减少数据加载线程数
    ddp_find_unused_parameters=False,  # 关闭DDP中查找未使用的参数
    local_rank=local_rank  # 支持分布式训练
)

# ----------------------------
# Step 10. 创建数据整理器（Data Collator）
# ----------------------------
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ----------------------------
# Step 11. 构建 MyTrainer 并开始训练
# ----------------------------
trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,  # 使用 data_collator 替代 tokenizer
)

# 开始训练
trainer.train()

# 保存最终模型（包括权重和配置）
trainer.save_model("./results/final_model")
# 同时保存分词器，确保后续恢复环境一致
tokenizer.save_pretrained("./results/final_model")