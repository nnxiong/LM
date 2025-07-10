# Finetune API框架v2 使用介绍

可以参考`/mnt/data1/bioford2/test/test_finetune_fastapi.py`，`/mnt/data1/bioford2/test/curl_test_finetune.md` 和 `/mnt/data1/bioford2/test/test_finetune.py`

## 一、finetune_api 使用示例
以下使用了 `finetune_api` API框架，以及test model的 inference函数 `run_test_finetune`

```python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from finetune_api import FinetuneAPI
from test_finetune import run_test_finetune

# 创建 API 实例
api = FinetuneAPI(
    model_name="test",
    run_finetune_func=run_test_finetune,
    model_path="/app/model_path/test/",                  
    finetune_data_dir="/app/finetune_datasets/test"  
)

if __name__ == "__main__":
    api.start(port=8001)
```


## 二、 post 字段介绍
```js
curl -X "POST" "http://0.0.0.0:8001/finetune" \
    -H "Content-Type: application/json" \
    -d '{
        "task_id": "task_1",
        "call_back_api": "http://172.17.0.1:5000/callback",
        "input_content": {
            "customized_input":{        
                "train_dataset_path": "/app/finetune_datasets/test/train",  # 必有字段，值可以是网上路径 or 宿主机路径
                "valid_dataset_path": "/app/finetune_datasets/test/test",  # 必有字段，值可以是网上路径 or 宿主机路径
                "test_dataset_path": "/app/finetune_datasets/test/test"  # 必有字段，值可以是网上路径 or 宿主机路径
            },
        },
        "param_dict": {
            "finetune_mode": "full",      # 必有字段，值str
            "config_param": {},         # 必有字段，值dict
            "show_frequecy": 2,        # 必有字段，值int
            "measurement": "epoch",      # 必有字段，值 "step" or "epoch"
            "total_epoch_number": 5,     # 必有字段，值 int 必须大于等于 show_frequecy字段
            "batch_size": 64,          # 必有字段，int
            "lr": 0.001                # 必有字段，float
            }  
    }'
```



## 三、POST 请求参数表

| 字段名                  | 是否必填 | 类型      | 说明                          |
| -------------------- | ---- | ------- | --------------------------- |
| `task_id`            | 是    | `str`   | 任务唯一标识                      |
| `call_back_api`      | 是    | `str`   | 回调 API 地址，结果通过 POST 回传      |
| `input_content`      | 是    | `dict`  | 数据路径：支持本地或网址                |
| `customized_input`   | 是    | `dict`  | 包括 train/valid/test 路径      |
| `param_dict`         | 是    | `dict`  | 调用参数设置                      |
| `finetune_mode`      | 是    | `str`   | "full"                      |
| `config_param`       | 是    | `dict`  | 预留配置                        |
| `show_frequecy`      | 是    | `int`   | 依据 measurement 指定迭代间隐秒输出频率  |
| `measurement`        | 是    | `str`   | "step" 或 "epoch"            |
| `total_epoch_number` | 是    | `int`   | 训练总轮数 (大于等于 show\_frequecy) |
| `batch_size`         | 是    | `int`   | 批量大小                        |
| `lr`                 | 是    | `float` | 学习率                         |

---

## 四、返回值说明
示例
```js
{'status': 'success', 'task_id': 'task_1', 'is_final': False, 'idx': 2, 'fig_param': {'x_axis': [2], 'x_label': 'epoch', 'losses': [106.59809218160808], 'accuracies': [0.9657666666666667], 'valid_losses': [7.502392057795078], 'valid_accuracies': [0.9724]}, 'test_accuracy': None, 'output_model_path': '/mnt/data1/bioford2/finetuned_model_path/test/20250708024147_task_1/model_files', 'output_log_path': '/mnt/data1/bioford2/finetuned_model_path/test/20250708024147_task_1/finetune.log'}
192.168.1.4 - - [08/Jul/2025 10:42:00] "POST /callback HTTP/1.1" 200 -

{'status': 'success', 'task_id': 'task_1', 'is_final': False, 'idx': 4, 'fig_param': {'x_axis': [2, 4], 'x_label': 'epoch', 'losses': [106.59809218160808, 51.49035492935218], 'accuracies': [0.9657666666666667, 0.9829], 'valid_losses': [7.502392057795078, 6.990388248581439], 'valid_accuracies': [0.9724, 0.973]}, 'test_accuracy': None, 'output_model_path': '/mnt/data1/bioford2/finetuned_model_path/test/20250708024147_task_1/model_files', 'output_log_path': '/mnt/data1/bioford2/finetuned_model_path/test/20250708024147_task_1/finetune.log'}
192.168.1.4 - - [08/Jul/2025 10:42:11] "POST /callback HTTP/1.1" 200 -

{'status': 'success', 'task_id': 'task_1', 'is_final': True, 'idx': 5, 'fig_param': {'x_axis': [2, 4, 5], 'x_label': 'epoch', 'losses': [106.59809218160808, 51.49035492935218, 39.23344399477355], 'accuracies': [0.9657666666666667, 0.9829, 0.9863666666666666], 'valid_losses': [7.502392057795078, 6.990388248581439, 6.788880517240614], 'valid_accuracies': [0.9724, 0.973, 0.975]}, 'test_accuracy': 0.9766, 'output_model_path': '/mnt/data1/bioford2/finetuned_model_path/test/20250708024147_task_1/model_files', 'output_log_path': '/mnt/data1/bioford2/finetuned_model_path/test/20250708024147_task_1/finetune.log'}
192.168.1.4 - - [08/Jul/2025 10:42:17] "POST /callback HTTP/1.1" 200 
```

| 字段                  | 类型            | 说明                     |
| ------------------- | ------------- | ---------------------- |
| `status`            | `str`         | "success" 或 "fail"     |
| `task_id`           | `str`         | 任务 ID                  |
| `is_final`          | `bool`        | 是否已训练完成                |
| `idx`               | `int`         | 当前迭代值（与 x\_axis 同）     |
| `fig_param`         | `dict`        | 图表参数，包括 x\_axis/丢断线/精度 |
| `test_accuracy`     | `float/或null` | 最终测试集 acc              |
| `output_model_path` | `str`         | 最佳模型保存路径               |
| `output_log_path`   | `str`         | 日志路径                   |
---

## 五、结果示例说明

### 1. measurement = "epoch"

* show\_frequency = 2
* total\_epoch = 5
* 返回 epoch = 2, 4, 5 (final)

### 2. measurement = "step"

* show\_frequency = 200
* total\_epoch = 2
* 步数缩小，返回 step = 200, 400, ..., final\_step

### 3. 支持网址路径

* train/valid/test 路径支持全球 HTTP(S)服务器

---

## 六、模型路径生成

根据参数创建:

```python
model_path = "/mnt/data1/bioford2/model_path/test"
output_model_path = "/mnt/data1/bioford2/finetuned_model_path/test/{timestamp}_task_x/model_files"
output_log_path = "/mnt/data1/bioford2/finetuned_model_path/test/{timestamp}_task_x/finetune.log"
```

* 存储 best\_model\_epoch/step.pth
* 返回相应路径

---

## 七、内部 run\_test\_finetune 返回格式

```python
{
  "is_final": False,               # 必有字段
  "idx": step_count or epoch,      # 必有字段
  "fig_param": {                   # 必有字段
    "x_axis": [...],               # 必有字段
    "x_label": "step" or "epoch",    # 必有字段
    "losses": [...],
    "accuracies": [...],
    "valid_losses": [...],
    "valid_accuracies": [...]
  },
  "test_accuracy": null or float,      # test验证，可以为其他test 指标
  "output_model_path": ...,            # 必有字段   返回为文件夹地址，可供inference使用
  "output_log_path": ...               # 必有字段
}
```
