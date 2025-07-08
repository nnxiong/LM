from base import BaseProcessor
from typing import List, Dict, Tuple

class InternLMProcessor(BaseProcessor):
    """InternLM2处理器（严格遵循官方模板规范）"""
    
    TEMPLATE = {
        "system": "<|im_start|>system\n{system}<|im_end|>\n",
        "user": "<|im_start|>user\n{input}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{content}<|im_end|>",
        "assistant_prefix": "<|im_start|>assistant\n",
        "suffix": "<|im_end|>",
        "sep": "\n",
        "stop_words": ["<|im_end|>"]
    }

    def build_prompt(self, example: Dict) -> str:
        """严格按官方模板构建Prompt"""
        if "messages" in example:
            return self._build_messages_prompt(example)
        elif "history" in example:
            return self._build_multi_turn_prompt(example["history"], example["query"])
        else:
            return self._build_instruction_prompt(example)

    def _build_messages_prompt(self, example: Dict) -> str:
        """处理messages格式（官方推荐格式）"""
        messages = example["messages"]
        prompt = ""
        
        # 处理system消息（必须为首条）
        if messages[0]["role"] == "system":
            sys_msg = messages.pop(0)
            prompt += self.TEMPLATE["system"].format(system=sys_msg["content"])
        
        # 处理对话轮次
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "user":
                prompt += self.TEMPLATE["user"].format(input=content)
            elif role == "assistant":
                prompt += self.TEMPLATE["assistant"].format(content=content) + self.TEMPLATE["sep"]
        
        # 添加assistant前缀用于生成
        prompt += self.TEMPLATE["assistant_prefix"]
        return prompt

    def _build_multi_turn_prompt(self, history: List[Tuple[str, str]], current_query: str) -> str:
        """构建多轮对话Prompt（兼容历史格式）"""
        prompt = self.TEMPLATE["system"].format(system="你是一个AI助手")  # 默认system
        
        # 处理历史对话
        for user_msg, assistant_msg in history:
            prompt += self.TEMPLATE["user"].format(input=user_msg)
            prompt += self.TEMPLATE["assistant"].format(content=assistant_msg) + self.TEMPLATE["sep"]
        
        # 添加当前查询
        prompt += self.TEMPLATE["user"].format(input=current_query)
        prompt += self.TEMPLATE["assistant_prefix"]
        return prompt

    def _build_instruction_prompt(self, example: Dict) -> str:
        """构建单轮指令Prompt（官方INSTRUCTION模板）"""
        prompt = self.TEMPLATE["system"].format(system="你是一个AI助手")  # 默认system
        prompt += self.TEMPLATE["user"].format(input=example["instruction"])
        if example.get("input", ""):
            prompt += self.TEMPLATE["sep"] + example["input"]
        prompt += self.TEMPLATE["assistant_prefix"]
        return prompt

    def tokenize(self, example: Dict, tokenizer, max_length: int) -> Dict:
        """严格按模板要求tokenize"""
        prompt = self.build_prompt(example)
        
        # 定位生成起始位置（assistant前缀后）
        gen_start = prompt.rfind(self.TEMPLATE["assistant_prefix"])
        input_part = prompt[:gen_start + len(self.TEMPLATE["assistant_prefix"])]
        
        # 获取输出内容（自动添加suffix）
        output_part = ""
        if "messages" in example:
            last_msg = example["messages"][-1]
            if last_msg["role"] == "assistant":
                output_part = last_msg["content"] + self.TEMPLATE["suffix"]
        elif "response" in example:
            output_part = example["response"] + self.TEMPLATE["suffix"]
        elif "output" in example:
            output_part = example["output"] + self.TEMPLATE["suffix"]

        # Tokenize处理（禁用自动添加特殊token）
        tokenized_input = tokenizer(
            input_part,
            truncation=True,
            max_length=max_length // 2,
            add_special_tokens=False
        )
        tokenized_output = tokenizer(
            output_part,
            truncation=True,
            max_length=max_length - len(tokenized_input["input_ids"]),
            add_special_tokens=False
        )
        
        return {
            "input_ids": tokenized_input["input_ids"] + tokenized_output["input_ids"],
            "attention_mask": [1] * (len(tokenized_input["input_ids"]) + len(tokenized_output["input_ids"])),
            "labels": [-100] * len(tokenized_input["input_ids"]) + tokenized_output["input_ids"]
        }

    @staticmethod
    def get_stop_words() -> List[str]:
        """官方指定的停止词"""
        return ["<|im_end|>"]
