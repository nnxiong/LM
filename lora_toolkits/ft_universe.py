import torch
from typing import Optional, Dict, List, Union
from datasets import load_dataset, Dataset
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
import warnings
import os
from data_processor import create_processor

class UniversalLoRATrainer:
    """支持全系列大模型的模块化LoRA微调工具"""
    
    MODEL_TARGETS = {
        "internlm2": ["wqkv", "wo", "w1", "w2", "w3"],
        "llama3": ["q_proj", "k_proj", "v_proj"],
        "chatglm3": ["query_key_value"],
        "qwen": ["c_attn"],
        "default": ["query", "value"]
    }

    def __init__(
        self,
        base_model_path: str,
        model_type: str = "auto",
        use_qlora: bool = False,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        **kwargs
    ):
        """
        初始化训练器
        :param base_model_path: 基座模型路径或HF模型ID
        :param model_type: 模型类型（auto/internlm2/llama3等）
        :param use_qlora: 是否启用4-bit量化训练
        :param device_map: 设备映射策略
        :param torch_dtype: 模型精度
        """
        self.base_model_path = base_model_path
        self.model_type = self._detect_model_type(model_type)
        self.processor = create_processor(self.model_type)
        
        # 初始化tokenizer（禁用fast tokenizer以确保兼容性）
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            use_fast=False,
            padding_side="right"  # 确保解码方向正确
        )
        
        # 量化配置（QLoRA专用）
        quantization_config = self._get_quant_config(torch_dtype) if use_qlora else None
        
        # 加载模型
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
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=kwargs.get("use_gradient_checkpointing", True)
            )

    def _detect_model_type(self, model_type: str) -> str:
        """自动检测模型类型"""
        if model_type != "auto":
            return model_type.lower()
            
        path = self.base_model_path.lower()
        if "internlm" in path:
            return "internlm2"
        elif "llama" in path:
            return "llama3"
        elif "chatglm" in path:
            return "chatglm3"
        elif "qwen" in path:
            return "qwen"
        else:
            warnings.warn(f"无法自动识别{path}的模型类型，使用默认配置")
            return "default"

    def _get_quant_config(self, torch_dtype) -> BitsAndBytesConfig:
        """生成QLoRA量化配置"""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    def load_data(
        self,
        data_path: str,
        max_length: int = 2048,
        format: str = "json",
        **kwargs
    ) -> Dataset:
        """
        加载并预处理数据集
        :param data_path: 数据文件路径或目录
        :param max_length: 最大序列长度
        :param format: 数据格式（json/csv/parquet）
        """
        # 加载原始数据集
        dataset = load_dataset(format, data_files=data_path, **kwargs)["train"]
        
        # 使用处理器进行tokenize
        return dataset.map(
            lambda x: self.processor.tokenize(x, self.tokenizer, max_length),
            batched=False,
            remove_columns=dataset.column_names
        )

    def train(
        self,
        dataset: Dataset,
        output_dir: str,
        lora_config: Optional[LoraConfig] = None,
        per_device_batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 2e-4,
        num_train_epochs: int = 1,
        logging_steps: int = 20,
        save_steps: int = 200,
        **kwargs
    ) -> Trainer:
        """
        执行LoRA微调训练
        :param dataset: 已处理的数据集
        :param output_dir: 输出目录
        :param lora_config: 自定义LoRA配置
        :return: 返回Trainer对象用于后续控制
        """
        # 自动配置LoRA目标层
        if lora_config is None:
            target_modules = self.MODEL_TARGETS.get(self.model_type, self.MODEL_TARGETS["default"])
            lora_config = LoraConfig(
                r=kwargs.get("r", 8),
                lora_alpha=kwargs.get("lora_alpha", 32),
                target_modules=target_modules,
                lora_dropout=kwargs.get("lora_dropout", 0.05),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )

        # 转换为PEFT模型
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # 训练参数配置
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            logging_steps=logging_steps,
            save_steps=save_steps,
            fp16=True,
            remove_unused_columns=False,
            optim="paged_adamw_8bit" if kwargs.get("use_qlora", False) else "adamw_torch",
            report_to=["tensorboard"],
            **{k: v for k, v in kwargs.items() if k not in ["lora_kwargs", "use_qlora"]}
        )

        # 初始化Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForSeq2Seq(
                self.tokenizer,
                pad_to_multiple_of=8,
                padding=True
            )
        )
        
        # 开始训练
        print("============== start finetuning ==============")
        self.model.config.use_cache = False
        trainer.train()
        print("============== end finetuning ==============")

        print("============== save the checkpoint ==============")
        # 保存LoRA权重
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print("============== save the checkpoint successful==============")

        return trainer

    def merge_and_save(
        self,
        lora_path: str,
        output_dir: str,
        **save_kwargs
    ) -> None:
        """
        合并LoRA权重到基础模型
        :param lora_path: LoRA权重路径
        :param output_dir: 合并后保存路径
        """
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
    # # 1. 初始化训练器 (QLoRA模式)
    trainer = UniversalLoRATrainer(
        base_model_path="/mnt/data1/zhongrongmiao/InternLM/merged_v3",
        model_type="internlm2",
        use_qlora=True,
        device_map="auto"
    )
    
    # 2. 加载数据 (自动调用data_processor处理)
    dataset = trainer.load_data(
        data_path='/mnt/data1/zhongrongmiao/InternLM/data_ft/instruction_dataset_train_v3_2.json',
        max_length=2048
    )
    
    # 3. 开始训练
    trainer.train(
        dataset=dataset,
        output_dir="/mnt/data1/zhongrongmiao/InternLM/lora_toolkits/lora_output",
        per_device_batch_size=4,
        num_train_epochs=2,
        lora_kwargs={"r": 16, "lora_alpha": 64}  # 自定义LoRA参数
    )
    
    # # 4. (可选) 合并保存完整模型
    # trainer.merge_and_save(
    #     lora_path="/mnt/data1/zhongrongmiao/InternLM/lora_toolkits/lora_output/checkpoint-470",
    #     output_dir="/mnt/data1/zhongrongmiao/InternLM/lora_toolkits/merged_model/InternLM"
    # )