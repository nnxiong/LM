from base import BaseProcessor
from typing import List, Dict, Tuple

class DefaultProcessor(BaseProcessor):
    """默认处理器（兼容单轮+多轮基础格式）"""
    
    def build_prompt(self, example: Dict) -> str:
        """自动适配单轮/多轮格式"""
        if "history" in example:
            return self._build_multi_turn_prompt(example["history"], example["query"])
        elif "instruction" in example:
            return self._build_instruction_prompt(example)
        else:
            raise ValueError("未知数据格式，必须包含history或instruction字段")

    def _build_multi_turn_prompt(self, history: List[Tuple[str, str]], current_query: str) -> str:
        """构建通用多轮对话Prompt"""
        prompt = ""
        for i, (user_msg, assistant_msg) in enumerate(history):
            prompt += f"### Round {i+1}\nUser: {user_msg}\nAssistant: {assistant_msg}\n"
        prompt += f"### Round {len(history)+1}\nUser: {current_query}\nAssistant: "
        return prompt

    def _build_instruction_prompt(self, example: Dict) -> str:
        """构建通用单轮指令Prompt"""
        prompt = f"Instruction: {example['instruction']}"
        if example.get("input", ""):
            prompt += f"\nInput: {example['input']}"
        prompt += "\nResponse: "
        return prompt

    def tokenize(self, example: Dict, tokenizer, max_length: int) -> Dict:
        """通用tokenize处理"""
        prompt = self.build_prompt(example)
        
        # 区分输入/输出部分
        if "history" in example:
            # 多轮对话取最后回复
            output_part = example.get("response", "") + tokenizer.eos_token
            input_part = prompt[:-len("Assistant: ")]  # 移除生成前缀
        else:
            # 单轮指令
            output_part = example.get("output", "") + tokenizer.eos_token
            input_part = prompt[:-len("Response: ")]  # 移除生成前缀

        # Tokenize处理
        tokenized_input = tokenizer(
            input_part,
            truncation=True,
            max_length=max_length // 2,
            add_special_tokens=True
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
        """返回通用停止词（使用tokenizer默认EOS）"""
        return []