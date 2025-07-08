import sys
from os.path import dirname, abspath
# 将项目根目录添加到Python路径
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from data_processor import create_processor
from transformers import AutoTokenizer
from pprint import pprint

def process_internlm_single_turn():
    """测试单轮指令处理"""
    print("=== 单轮指令处理测试 ===")
    processor = create_processor("internlm2")
    
    # 单轮指令数据
    example = {
        "instruction": "解释量子计算",
        "input": "用简单的语言说明",
        "output": "量子计算利用量子比特..."
    }
    
    # 构建Prompt
    prompt = processor.build_prompt(example)
    print("\n生成的Prompt:")
    print(prompt)
    
    # Tokenize处理
    tokenized = processor.tokenize(example, tokenizer, max_length=1024)
    print("\nTokenized结果:")
    pprint({k: len(v) for k, v in tokenized.items()})
    print("Labels示例:", tokenized["labels"][-10:])

def process_internlm_multi_turn():
    """测试多轮对话处理（messages格式）"""
    print("\n=== 多轮对话处理测试（messages格式） ===")
    processor = create_processor("internlm2")
    
    # 多轮对话数据
    example = {
        "messages": [
            {"role": "system", "content": "你是一个AI助手"},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！我是InternLM"},
            {"role": "user", "content": "你会编程吗"},
            {"role": "assistant", "content": "我会Python等多种语言"}
        ],
        "response": "是的，我擅长Python"
    }
    
    # 构建Prompt
    prompt = processor.build_prompt(example)
    print("\n生成的Prompt:")
    print(prompt)
    
    # Tokenize处理
    tokenized = processor.tokenize(example, tokenizer, max_length=2048)
    print("\nTokenized结果:")
    pprint({k: len(v) for k, v in tokenized.items()})
    print("Labels掩码示例:", [i for i in tokenized["labels"] if i != -100][:5])

def process_internlm_history():
    """测试历史对话格式处理"""
    print("\n=== 历史对话格式处理测试 ===")
    processor = create_processor("internlm2")
    
    # 历史对话数据
    example = {
        "history": [
            ("你好", "你好！"),
            ("你会什么", "我能回答问题、编写代码等")
        ],
        "query": "用Python写个快速排序",
        "response": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)"
    }
    
    # 构建Prompt
    prompt = processor.build_prompt(example)
    print("\n生成的Prompt:")
    print(prompt)
    
    # Tokenize处理
    tokenized = processor.tokenize(example, tokenizer, max_length=2048)
    print("\nTokenized结果:")
    print(f"总长度: {len(tokenized['input_ids'])}")
    print(f"Input部分长度: {len([x for x in tokenized['labels'] if x == -100])}")
    print(f"Output部分长度: {len([x for x in tokenized['labels'] if x != -100])}")

if __name__ == "__main__":
    # 初始化tokenizer（自动加载InternLM特殊token）
    tokenizer = AutoTokenizer.from_pretrained("/mnt/data1/zhongrongmiao/InternLM/merged_v3", trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained("../merged_v3", torch_dtype=torch.float16, trust_remote_code=True).cuda()
    
    # 运行测试
    process_internlm_single_turn()
    process_internlm_multi_turn()
    process_internlm_history()