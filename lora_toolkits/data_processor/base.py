from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

class BaseProcessor(ABC):
    """处理器基类（单轮和多轮共用）"""
    
    @abstractmethod
    def build_prompt(self, example: Dict) -> str:
        """构建模型输入模板"""
        pass
    
    @abstractmethod
    def tokenize(self, example: Dict, tokenizer, max_length: int) -> Dict:
        """tokenize处理"""
        pass

    @staticmethod
    def get_stop_words() -> List[str]:
        """获取停止词（多轮对话专用）"""
        return []



# class MultiTurnProcessor(BaseProcessor):
#     """多轮对话处理器基类"""
    
#     @abstractmethod
#     def build_multi_turn_prompt(self, history: List[Tuple[str, str]], current_query: str) -> str:
#         pass
    
#     @abstractmethod
#     def get_stop_words(self) -> List[str]:
#         pass

#     def build_prompt(self, example: Dict) -> str:
#         return self.build_multi_turn_prompt(example["history"], example["query"])

#     def tokenize(self, example: Dict, tokenizer, max_length: int) -> Dict:
#         history = example["history"]
#         current_query = example["query"]
#         response = example.get("response", "")
        
#         prompt = self.build_multi_turn_prompt(history, current_query)
#         input_ids = tokenizer.encode(prompt, truncation=True, max_length=max_length//2)
#         output_ids = tokenizer.encode(response, truncation=True, max_length=max_length-len(input_ids))
        
#         return {
#             "input_ids": input_ids + output_ids,
#             "attention_mask": [1] * (len(input_ids) + len(output_ids)),
#             "labels": [-100] * len(input_ids) + output_ids
#         }