from base import BaseProcessor
from typing import List, Dict, Tuple

class QwenProcessor(BaseProcessor):
    """Qwen系列处理器（完整单轮+多轮实现）"""
    
    TEMPLATE = {
        "system": "<|im_start|>system\n{content}<|im_end|>\n",
        "user": "<|im_start|>user\n{content}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{content}<|im_end|>\n",
        "assistant_prefix": "<|im_start|>assistant\n",
        "eos_token": "<|im_end|>",
        "stop_words": ["<|im_end|>", "<|endoftext|>"]
    }

    def build_prompt(self, example: Dict) -> str:
        """自动判断数据格式并构建Prompt"""
        if "messages" in example:
            return self._build_messages_prompt(example)
        elif "history" in example:
            return self._build_multi_turn_prompt(example["history"], example["query"])
        else:
            return self._build_instruction_prompt(example)

    def _build_messages_prompt(self, example: Dict) -> str:
        """处理messages格式数据"""
        messages = example["messages"]
        prompt = ""
        
        # 处理system消息（可选）
        if messages[0]["role"] == "system":
            sys_msg = messages.pop(0)
            prompt += self.TEMPLATE["system"].format(content=sys_msg["content"])
        
        # 处理对话轮次
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "user":
                prompt += self.TEMPLATE["user"].format(content=content)
            elif role == "assistant":
                prompt += self.TEMPLATE["assistant"].format(content=content)
        
        # 添加assistant前缀用于生成
        prompt += self.TEMPLATE["assistant_prefix"]
        return prompt

    def _build_multi_turn_prompt(self, history: List[Tuple[str, str]], current_query: str) -> str:
        """构建多轮对话Prompt"""
        prompt = ""
        for user_msg, assistant_msg in history:
            prompt += self.TEMPLATE["user"].format(content=user_msg)
            prompt += self.TEMPLATE["assistant"].format(content=assistant_msg)
        
        # 添加当前查询和assistant前缀
        prompt += self.TEMPLATE["user"].format(content=current_query)
        prompt += self.TEMPLATE["assistant_prefix"]
        return prompt

    def _build_instruction_prompt(self, example: Dict) -> str:
        """构建单轮指令Prompt"""
        prompt = example["instruction"]
        if example.get("input", ""):
            prompt += "\n" + example["input"]
        return prompt

    def tokenize(self, example: Dict, tokenizer, max_length: int) -> Dict:
        """统一tokenize处理（自动适配单轮/多轮）"""
        prompt = self.build_prompt(example)
        
        # 定位生成起始位置
        gen_start = prompt.rfind(self.TEMPLATE["assistant_prefix"])
        input_part = prompt[:gen_start + len(self.TEMPLATE["assistant_prefix"])]
        
        # 获取输出内容（自动添加EOS）
        output_part = ""
        if "messages" in example:
            last_msg = example["messages"][-1]
            if last_msg["role"] == "assistant":
                output_part = last_msg["content"] + self.TEMPLATE["eos_token"]
        elif "response" in example:
            output_part = example["response"] + self.TEMPLATE["eos_token"]
        elif "output" in example:
            output_part = example["output"] + self.TEMPLATE["eos_token"]

        # Tokenize处理
        tokenized_input = tokenizer(
            input_part,
            truncation=True,
            max_length=max_length // 2,
            add_special_tokens=False  # 模板已包含特殊token
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
        """返回Qwen停止词（用于生成控制）"""
        return ["<|im_end|>", "<|endoftext|>"]