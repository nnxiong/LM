'''
在使用 PEFT（Parameter-Efficient Fine-Tuning）时，通常需要指定哪些模块需要进行适配。以下是一些常见的模块匹配规则：
# 1. 指定 target_modules 
- /home/zhongrongmiao/anaconda3/envs/ProteinMPNN/lib/python3.8/site-packages/peft/utils/constants.py 中没有该模型对应的target_modules
# 2. 不指定 target_modules
- /home/zhongrongmiao/anaconda3/envs/ProteinMPNN/lib/python3.8/site-packages/peft/utils/constants.py 中有该模型对应的target_modules

部分模块匹配规则​​：
​​注意力层​​：自动识别 q_proj、k_proj、v_proj、o_proj（LLaMA 风格）或 query、key、value、dense（BERT 风格）等。
​​前馈网络（FFN）​​：适配 gate_proj、up_proj、down_proj（SwiGLU 结构）或 intermediate.dense、output.dense（传统 MLP）。
'''

from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer)

# ​​即使加载本地模型​​，如果该模型包含自定义代码（如非 transformers 官方支持的架构或分词器），仍然需要设置 trust_remote_code=True
model = AutoModelForCausalLM.from_pretrained('/mnt/data2/model/Qwen/Qwen2-7B-Instruct', trust_remote_code=True)
# 查看模型的参数 找到 self attention 模块 和 FNN前馈网络模块
for name,param in model.named_parameters():
    print(name)

# 指定 target_modules
config = LoraConfig(
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM")  # 不指定 target_modules
peft_model = get_peft_model(model, config)
# 查看实际适配的模块
print(config.target_modules)

# ​​PEFT 库会成功匹配并适配所有指定的目标模块​​， 从 model.layers.0 —— model.layers.n-1 中的每一层都将应用 LoRA 适配器。

model = AutoModelForCausalLM.from_pretrained('/mnt/data1/zhongrongmiao/InternLM/internlm2_5-1_8b-chat', trust_remote_code=True)
# 查看模型的参数 找到 attention 模块 和 FNN前馈网络模块
for name,param in model.named_parameters():
    print(name)
# 指定 target_modules
config = LoraConfig(
    target_modules=["wqkv", "wo", "w1", "w2", "w3"],
    task_type="CAUSAL_LM")  # 不指定 target_modules
peft_model = get_peft_model(model, config)
# 查看实际适配的模块
print(config.target_modules)