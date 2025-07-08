import torch
from typing import Optional, Dict, List, Union
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import load_dataset, Dataset
import warnings

class UniversalLoRATrainer:
    """支持全系列大模型QLoRA/LoRA微调的一站式工具类"""
    
    def __init__(
        self,
        base_model_path: str,
        model_type: str = "auto",  # 可选: chatglm/llama/bloom/auto
        use_qlora: bool = False,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        **kwargs
    ):
        """
        初始化基座模型
        :param base_model_path: 模型路径或HF模型ID
        :param model_type: 模型架构类型，auto为自动检测
        :param use_qlora: 是否启用4-bit量化训练
        :param device_map: 多卡分配策略
        """
        self.base_model_path = base_model_path
        self.model_type = self._detect_model_type(model_type)
        self.use_qlora = use_qlora
        
        # 初始化tokenizer (必须优先加载)
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            use_fast=False
        )
        
        # 量化配置 (QLoRA专用)
        quantization_config = self._get_quant_config(torch_dtype) if use_qlora else None
        
        # 加载基座模型
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            device_map=device_map,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            **kwargs
        )
        
        # QLoRA特殊处理
        if use_qlora:
            self.model = prepare_model_for_kbit_training(self.model)

    def _detect_model_type(self, model_type: str) -> str:
        """自动检测模型架构类型"""
        if model_type != "auto":
            return model_type.lower()
            
        path = self.base_model_path.lower()
        if "chatglm" in path:
            return "chatglm"
        elif "llama" in path or "alpaca" in path:
            return "llama"
        elif "bloom" in path:
            return "bloom"
        else:
            warnings.warn("未能自动识别模型类型，默认使用llama配置")
            return "llama"

    def _get_quant_config(self, torch_dtype) -> BitsAndBytesConfig:
        """生成QLoRA量化配置"""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    def get_peft_config(
        self,
        r: int = 8,
        lora_alpha: int = 32,
        target_modules: Union[List[str], str] = "auto",
        lora_dropout: float = 0.05,
        bias: str = "none",
        **kwargs
    ) -> LoraConfig:
        """
        获取LoRA配置
        :param target_modules: 可指定层名或auto自动匹配
        """
        # 自动匹配目标层
        if target_modules == "auto":
            target_modules = self._get_target_modules()
            
        return LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type=TaskType.CAUSAL_LM,
            **kwargs
        )

    def _get_target_modules(self) -> List[str]:
        """根据模型类型返回默认LoRA目标层"""
        mapping = {
            "chatglm": ["query_key_value"],
            "llama": ["q_proj", "k_proj", "v_proj"],
            "bloom": ["query_key_value"],
        }
        return mapping.get(self.model_type, ["query", "value"])

    def load_data(
        self,
        data_path: str,
        format: str = "json",
        max_length: int = 2048,
        **dataset_kwargs
    ) -> Dataset:
        """
        加载并预处理数据集
        :param data_path: 数据文件路径
        :param format: json/csv/parquet等
        :param max_length: 最大序列长度
        """
        dataset = load_dataset(format, data_files=data_path, **dataset_kwargs)["train"]
        return dataset.map(
            lambda x: self.tokenize_function(x, max_length),
            batched=False,
            remove_columns=dataset.column_names
        )

    def tokenize_function(
        self,
        example: Dict,
        max_length: int,
        ignore_label_id: int = -100
    ) -> Dict:
        """
        通用tokenize处理(支持单轮/多轮/纯文本)
        输入格式:
        - 单轮: {"instruction":..., "input":..., "output":...}
        - 多轮: {"history": [[q1,a1],[q2,a2]], "query":..., "response":...}
        - 纯文本: {"text":...}
        """
        # 多轮对话处理
        if "history" in example:
            prompt = self._build_multi_turn_prompt(example)
            response = example["response"]
        # 指令微调处理
        elif "instruction" in example:
            prompt = self._build_instruction_prompt(example)
            response = example["output"]
        # 纯文本处理
        else:
            prompt = example["text"]
            response = None

        # Tokenize处理
        tokenized = self.tokenizer(
            prompt,
            truncation=True,
            max_length=max_length // 2,
            add_special_tokens=True
        )
        
        # 无监督学习时只返回input_ids
        if response is None:
            return {"input_ids": tokenized["input_ids"]}
            
        # 监督学习处理
        response_ids = self.tokenizer(
            response,
            truncation=True,
            max_length=max_length - len(tokenized["input_ids"]),
            add_special_tokens=False
        )["input_ids"]

        return {
            "input_ids": tokenized["input_ids"] + response_ids,
            "labels": [ignore_label_id] * len(tokenized["input_ids"]) + response_ids,
            "attention_mask": [1] * (len(tokenized["input_ids"]) + len(response_ids))
        }

    def _build_instruction_prompt(self, example: Dict) -> str:
        """构建指令微调prompt"""
        # ChatGLM特殊格式
        if self.model_type == "chatglm":
            return f"[gMASK]sop {example['instruction']}\n{example.get('input', '')}"
        # LLaMA系统提示
        elif self.model_type == "llama":
            return f"<<SYS>>\n{example['instruction']}\n<</SYS>>\n\n{example.get('input', '')}"
        # 通用格式
        else:
            return f"{example['instruction']}\n{example.get('input', '')}"

    def _build_multi_turn_prompt(self, example: Dict) -> str:
        """构建多轮对话prompt"""
        history = example["history"]
        current_query = example["query"]
        
        # ChatGLM格式
        if self.model_type == "chatglm":
            prompt = ""
            for i, (q, a) in enumerate(history):
                prompt += f"[Round {i+1}]\n问：{q}\n答：{a}\n"
            prompt += f"[Round {len(history)+1}]\n问：{current_query}\n答："
        # LLaMA格式
        elif self.model_type == "llama":
            prompt = "<s>"
            for q, a in history:
                prompt += f"[INST] {q} [/INST] {a} </s>"
            prompt += f"[INST] {current_query} [/INST]"
        # 通用格式
        else:
            prompt = "\n".join([f"Q: {q}\nA: {a}" for q, a in history])
            prompt += f"\nQ: {current_query}\nA:"
            
        return prompt

    def train(
        self,
        dataset: Dataset,
        output_dir: str,
        lora_config: Optional[LoraConfig] = None,
        per_device_batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 2e-4,
        num_train_epochs: int = 1,
        save_steps: int = 100,
        logging_steps: int = 10,
        **training_kwargs
    ) -> Trainer:
        """
        执行训练
        :return: 返回Trainer对象便于后续控制
        """
        # 默认配置
        if lora_config is None:
            lora_config = self.get_peft_config()
            
        # 转换为PEFT模型
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            save_steps=save_steps,
            logging_steps=logging_steps,
            fp16=True,
            remove_unused_columns=False,
            optim="paged_adamw_8bit" if self.use_qlora else "adamw_torch",
            **training_kwargs
        )

        # 数据收集器
        collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            pad_to_multiple_of=8,
            padding=True
        )

        # 开始训练
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collator,
        )
        
        self.model.config.use_cache = False
        trainer.train()
        self.model.save_pretrained(output_dir)
        
        return trainer

    def merge_and_save(
        self,
        lora_path: str,
        output_dir: str,
        **save_kwargs
    ) -> None:
        """合并LoRA权重并保存完整模型"""
        # 重新加载FP16原始模型
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 合并权重
        lora_model = PeftModel.from_pretrained(base_model, lora_path)
        merged_model = lora_model.merge_and_unload()
        
        # 保存完整模型
        merged_model.save_pretrained(output_dir, **save_kwargs)
        self.tokenizer.save_pretrained(output_dir)

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 1. 初始化训练器 (QLoRA模式)
    trainer = UniversalLoRATrainer(
        base_model_path="THUDM/chatglm3-6b",
        model_type="chatglm",
        use_qlora=True,
        device_map="auto"
    )

    # 2. 加载数据 (支持json/parquet/csv等)
    dataset = trainer.load_data(
        data_path="data/train.json",
        max_length=2048
    )

    # 3. 开始训练
    trainer.train(
        dataset=dataset,
        output_dir="./chatglm3-6b-lora",
        per_device_batch_size=4,
        num_train_epochs=1
    )

    # 4. 合并保存 (QLoRA训练后必须合并)
    trainer.merge_and_save(
        lora_path="./chatglm3-6b-lora",
        output_dir="./chatglm3-6b-merged"
    )