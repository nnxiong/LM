import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(abspath(__file__)))  # 添加项目根目录到路径

from typing import Type, Dict
from base import BaseProcessor
from internlm import InternLMProcessor
from llama import LlamaProcessor
from chatglm import ChatGLMProcessor
from .qwen import QwenProcessor
from .default import DefaultProcessor

class ProcessorFactory:
    """处理器工厂（自动匹配模型类型）"""
    
    _PROCESSOR_MAPPING = {
        # 标准模型标识
        "internlm": InternLMProcessor,
        "internlm2": InternLMProcessor,
        "llama": LlamaProcessor,
        "llama2": LlamaProcessor,
        "llama3": LlamaProcessor,
        "chatglm": ChatGLMProcessor,
        "chatglm2": ChatGLMProcessor,
        "chatglm3": ChatGLMProcessor,
        "qwen": QwenProcessor,
        "qwen2": QwenProcessor,
        
        # 兼容HF模型ID
        "models/internlm": InternLMProcessor,
        "models/llama": LlamaProcessor,
        "THUDM/chatglm": ChatGLMProcessor,
        "Qwen/Qwen": QwenProcessor
    }

    @classmethod
    def get_processor(cls, model_type: str) -> Type[BaseProcessor]:
        """
        获取处理器类
        :param model_type: 模型标识（不区分大小写）
        :return: 处理器类（未匹配时返回DefaultProcessor）
        """
        model_type = model_type.lower()
        for key in cls._PROCESSOR_MAPPING:
            if key in model_type:
                return cls._PROCESSOR_MAPPING[key]
        return DefaultProcessor

    @classmethod
    def create_processor(cls, model_type: str, **kwargs) -> BaseProcessor:
        """
        创建处理器实例
        :param model_type: 模型标识或路径
        :param kwargs: 处理器初始化参数
        """
        processor_class = cls.get_processor(model_type)
        return processor_class(**kwargs)

# 快捷导入
get_processor = ProcessorFactory.get_processor
create_processor = ProcessorFactory.create_processor