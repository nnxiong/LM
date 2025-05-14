# 1 构造微调数据集
根据不同模型微调对应的不同模板，需要构建不同的微调数据集。可参考代码 `ft_data_processing.py`


# 2 微调
## 2.1 环境安装
步骤 0. 使用 conda 先构建一个 Python-3.10 的虚拟环境

```
conda create --name xtuner-env python=3.10 -y
conda activate xtuner-env
```

步骤 1. 安装 XTuner

```
pip install -U 'xtuner[deepspeed]'
```

验证：
打印配置文件： 在命令行中使用 `xtuner list-cfg` 验证是否能打印配置文件列表。


## 2.2 微调命令
使用 `xtuner train` 启动训练
```
NPROC_PER_NODE=2 xtuner train ./internlm2_chat_2_5_1_8b_lora.py --deepspeed deepspeed_zero2  --work-dir ./work_dirs3/
```
- `./internlm2_chat_2_5_1_8b_lora.py`：微调配置文件
- `--deepspeed deepspeed_zero2`：指定 deepspeed_zero2 优化微调
- `--work-dir ./work_dirs3/`: 指定训练日志及 checkpoint 保存路径

## 2.3 微调配置文件
参考 `internlm2_chat_2_5_1_8b_lora.py` 文件，重要配置参数介绍:
- `pretrained_model_name_or_path`：待微调 LLM
- `data_files`：微调指令数据集列表
- `batch_size`：训练 batch size
- `lr`：学习率
- `weight_decay`：权重衰退系数
- `evaluation_inputs`：验证数据列表



## 2.4 转换格式
当模型微调完毕，创建存放 hf 格式参数的目录:
```
mkdir .hf/ultimate_hf_v3
```

转换格式:
```
xtuner convert pth_to_hf .internlm2_chat_2_5_1_8b_lora.py \
                            .work_dirs3/iter_128.pth\
                           .hf/ultimate_hf_v3
```


## 2.5 合并参数
创建存放合并后的参数的目录：
```
mkdir .merged_v3
```

合并参数
```
xtuner convert merge .internlm2_5-1_8b-chat \
                        .hf/ultimate_hf_v3\
                        .merged_v3 \
                        --max-shard-size 2GB
```

## 2.6 对话测试
```
xtuner chat .merged_v3 --prompt-template internlm2_chat
```


# 3 FastAPI 部署
参考 `model_api.py`


