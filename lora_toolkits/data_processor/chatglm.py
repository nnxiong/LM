from base import BaseProcessor
from typing import List, Dict, Tuple

class ChatGLMProcessor(BaseProcessor):
    """ChatGLM3处理器（完整单轮+多轮实现）"""
    
    def build_prompt(self, example: Dict) -> str:
        """自动判断数据格式并构建Prompt"""
        if "messages" in example:
            return self._build_messages_prompt(example)
        elif "history" in example:
            return self._build_multi_turn_prompt(example["history"], example["query"])
        else:
            return self._build_instruction_prompt(example)

    def _build_messages_prompt(self, example: Dict) -> str:
        """处理新版messages格式"""
        messages = example["messages"]
        prompt = "[gMASK]sop "
        
        # 处理system消息
        if messages[0]["role"] == "system":
            sys_msg = messages.pop(0)
            prompt += f"{sys_msg['content']}\n"
        
        # 处理对话历史
        for i, msg in enumerate(messages):
            role, content = msg["role"], msg["content"]
            if role == "user":
                prompt += f"[Round {i+1}]\n问：{content}\n"
            elif role == "assistant":
                prompt += f"答：{content}\n"
        
        # 添加assistant前缀
        prompt += "答："
        return prompt

    def _build_multi_turn_prompt(self, history: List[Tuple[str, str]], current_query: str) -> str:
        """构建多轮对话Prompt"""
        prompt = "[gMASK]sop "
        for i, (user_msg, assistant_msg) in enumerate(history):
            prompt += f"[Round {i+1}]\n问：{user_msg}\n答：{assistant_msg}\n"
        prompt += f"[Round {len(history)+1}]\n问：{current_query}\n答："
        return prompt

    def _build_instruction_prompt(self, example: Dict) -> str:
        """构建单轮指令Prompt"""
        prompt = f"[gMASK]sop {example['instruction']}"
        if example.get("input", ""):
            prompt += f"\n{example['input']}"
        return prompt

    def tokenize(self, example: Dict, tokenizer, max_length: int) -> Dict:
        """统一tokenize处理"""
        prompt = self.build_prompt(example)
        
        # 定位最后一个"答："的位置
        last_answer_pos = prompt.rfind("答：")
        input_part = prompt[:last_answer_pos + 2]  # 包含"答："
        output_part = ""
        
        # 提取assistant回复
        if "messages" in example:
            last_msg = example["messages"][-1]
            if last_msg["role"] == "assistant":
                output_part = last_msg["content"]
        elif "response" in example:
            output_part = example["response"]
        elif "output" in example:
            output_part = example["output"]

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
        """返回ChatGLM停止词"""
        return ["</s>"]