# API框架v2 使用介绍
## inference_api_v2 使用示例
以下使用了 `inference_api_v2` API框架，以及test model的 inference函数 `run_test_inference`

```python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference_api_v2 import InferenceAPI
from test_inference_v2 import run_test_inference

# 创建 API 实例
api = InferenceAPI(
    model_name="test",
    run_inference_func=run_test_inference,
    model_path="/app/model_path/test/",  # 按照
    data_dir="/app/datasets/test",
    input_webpath_key = "input_path",  # 需要下载的文件路径在 input_content中的key 【如果模型有功能需要input_path就需要给出key】   
)

if __name__ == "__main__":
    api.start(port=8101)
```
其中 `input_webpath_key` 是input文件对应的key （可以结合 情况3，4，5 的IO理解），API框架将自动下载 网络文件或者找寻框架中的文件


## run_modelname_inference 函数要求

输入值顺序为 input_dict, param_dict, model_path (必须是框架内路径),output_path_name
返回值必须是一个tuple，第一项为result_type 可取 "file" 或其他任意字符串 ==注意：如果返回值是路径，那么result_type必须是"file"！！！==

如果有file输出，记得在 output_path_name 加上对应的文件后缀

例如
```python
import os

def run_test_inference(customized_input, param_dict, model_path="../model_path/test", output_path_name="../datasets/test/input_data"):
    output_path = output_path_name + ".txt"

    mode = param_dict["mode"]
    # 检查 model_path 是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
        # input
    input_path = customized_input.get("input_path", None)
    input_strings = customized_input.get("input_strings", None)
    

    if mode == "value_value":
        results = customized_input.get("input_strings", None)
        results.insert(0,"CCCCC")

        return "array", results

    elif mode == "value_file":
        results = customized_input.get("input_strings", None)
        results.insert(0,"CCCCC")

        with open(output_path, "w") as f:
            for line in results:
                f.write(line + "\n")
        return "file", output_path


    elif mode == "file_value":
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        with open(input_path, "r") as f:
            results = [line.strip() for line in f if line.strip()]
            results.insert(0,"CCCCC")
        
        return "array", results
    

    elif mode == "file_file":
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        with open(input_path, "r") as f:
            results = [line.strip() for line in f if line.strip()]
            results.insert(0,"CCCCC")

        with open(output_path, "w") as f:
            for line in results:
                f.write(line + "\n")

        return "file", output_path


    elif mode =="mixed_file":
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        with open(input_path, "r") as f:
            results = [line.strip() for line in f if line.strip()]
            results.extend(input_strings)
            results.insert(0,"CCCCC")
        with open(output_path, "w") as f:
            for line in results:
                f.write(line + "\n")

        return "file", output_path


    else:
        raise ValueError("incorrect mode!")



# Example usage
if __name__ == "__main__":

    print("========== case 1 =============")
    # 情况1     input: string    output:string
    result1 = run_test_inference({ "input_strings" :["AAAAAAA","BBBBBB"]}, {"mode":"value_value"}, model_path = "../model_path/test", output_path_name = "/app/datasets/test/results/write_test_case2")
    print("result1:",result1)

    # 情况2     input: string    output:txt
    print("========== case 2 =============")
    # post output_path/default_output_path_ext 如果输出是file，必须二选一
    result2 = run_test_inference({ "input_strings" :["AAAAAAA","BBBBBB"]}, {"mode":"value_file"}, model_path = "../model_path/test", output_path_name = "/app/datasets/test/results/write_test_case2")
    print("result2:",result2)

    # 情况3     input: txt    output:txt
    print("========== case 3 =============")
    result3 = run_test_inference({ "input_path" : "../datasets/test/input_data.txt"}, {"mode":"file_file"}, model_path = "../model_path/test", output_path_name = "/app/datasets/test/results/write_test_case3")
    print("result3:",result3)

    # 情况4     input: txt    output:string
    print("========== case 4 =============")
    result4 = run_test_inference({ "input_path" : "../datasets/test/input_data.txt"}, {"mode":"file_value"}, model_path = "../model_path/test")
    print("result4 :",result4)

    # 情况5 input：混合输入   output: txt
    print("========== case 5 =============")
    result5 = run_test_inference({ "input_path" : "../datasets/test/input_data.txt", "input_strings":["case5","case5_input_string"]}, {"mode":"mixed_file"}, model_path = "../model_path/test", output_path_name = "/app/datasets/test/results/write_test_case5")
    print("result5 :",result5)
```
对应的post也可以参考  `/mnt/data2/bioford/test/test_inference_v2.py`




## post 字段介绍
样例1
```bash
curl -X 'POST' 'http://0.0.0.0:8101/inference' \
     -H 'Content-Type: application/json' \
     -d '{
           "task_id": "task_3_1",
           "call_back_api": "http://172.17.0.1:5000/callback",
           "input_content": { "input_path" : "../datasets/test/input_data.txt"},
           "param_dict": {}
           "output_path": "../datasets/test/results/write_test3_1.txt"
         }'
```
样例2
``` bash
curl -X 'POST' 'http://0.0.0.0:8101/inference' \
     -H 'Content-Type: application/json' \
     -d '{
           "task_id": "task_3_2",
           "call_back_api": "http://172.17.0.1:5000/callback",
           "input_content": { "input_path" :"https://oxtium-bioford-public.obs.cn-south-1.myhuaweicloud.com/com/oxtium/bioford/test_examples/acme_input.txt" },
           "param_dict": {}
         }'
```

## post字段介绍表

| 字段名                 | 是否必填 | 类型          | 示例值或取值说明                                                                                                                                           |
|----------------------|----------|---------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `task_id`            | 是       | `str`         | `"task_1"`：任务唯一标识，便于追踪和回调结果                                                                                                               |
| `call_back_api`      | 是       | `str`         | `"http://172.17.0.1:5000/callback"`：回调接口地址，任务完成后POST结果到此API。若为空则直接返回结果                                                       |                                                                                         |
| `input_content`      | 是       | `dict`        | 包含实际输入数据，如：`{"input_path": "..."} 或 {"input_strings": [...]}`                                                                                  |
| `input_path`         | 条件必填 | `str`         | 本地路径或公网URL，用于指定输入文件路径。                                                                                     |                                                      |
| `param_dict`         | 是       | `dict`        | `{}`：参数字典，供模型使用的自定义推理参数                                                                                                                 |
| `output_path`        | 否       | `str`         | 本地路径，指定结果文件保存地址。优先级高于 `default_output_path_ext`                                                                                      |                                                                 |

## 返回值字段表

## 🧾 字段介绍表

| 字段名                 | 是否必填 | 类型          | 示例值或取值说明                                                                                                                                           |
|----------------------|----------|---------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `task_id`            | 是       | `str`         | `"task_1"`：任务唯一标识，便于追踪和回调结果                                                                                                               |                                                                             |
| `result_type`        | 自动生成 | `str`         | `"value"` 或 `"file"`：表示结果是直接返回值还是生成的文件                                                                                                  |
| `result`             | 自动生成 | `List[str]` 或 `str` | 推理返回结果，若为 `"value"` 类型，则为列表；若为 `"file"` 类型，则为文件的绝对路径                                                               



# 5种情况 IO详解
## 情况1     input: string    output: strings

``` bash
curl -X 'POST' 'http://0.0.0.0:8101/inference' \
    -H 'Content-Type: application/json' \
    -d '{
        "task_id": "task_1",
        "call_back_api": "http://172.17.0.1:5000/callback",
        "input_content": { "input_strings" :["AAAAAAA","BBBBBB"]},
        "param_dict": {"mode":"value_value"}
        }'

```
返回值
```
{"status":"success","task_id":"task_1","result_type":"array","result":["CCCCC","AAAAAAA","BBBBBB"]}
```


##  情况2     input: string    output:txt

### 情况 2.1  output_path 本地路径。  post 必须是docker框架内路径

```bash
curl -X 'POST' 'http://0.0.0.0:8101/inference' \
     -H 'Content-Type: application/json' \
     -d '{
           "task_id": "task_2_1",
           "call_back_api": "http://172.17.0.1:5000/callback",
           "input_content": { "input_strings" :["AAAAAAA","BBBBBB"]},
           "param_dict": {"mode":"value_file"},
           "output_path":"/app/datasets/test/results/write_test2_1"
         }'
```
返回值
```
{"status":"success","task_id":"task_2_1","result_type":"file","result":"/mnt/data2/bioford/datasets/test/results/write_test2_1.txt"}
```

### 情况 2.2  output_path 默认文件路径。 不指定 output_path时，系统生成默认格式的路径
```bash
curl -X 'POST' 'http://0.0.0.0:8101/inference' \
     -H 'Content-Type: application/json' \
     -d '{
           "task_id": "task_2_2",
           "call_back_api": "http://172.17.0.1:5000/callback",
           "input_content": { "input_strings" :["AAAAAAA","BBBBBB"]},
           "param_dict": {"mode":"value_file"}
         }'
```
返回值
```
{"status":"success","task_id":"task_2_1","result_type":"file","result":"/mnt/data2/bioford/datasets/test/results/20250423020919_task_2_1.txt"}
```


## 情况3     input: txt    output:txt

### 情况 3.1  input_path 本地路径
```bash
curl -X 'POST' 'http://0.0.0.0:8101/inference' \
     -H 'Content-Type: application/json' \
     -d '{
           "task_id": "task_3_1",
           "call_back_api": "http://172.17.0.1:5000/callback",
           "input_content": { "input_path" : "../datasets/test/input_data.txt"},
           "param_dict": {"mode":"file_file"},
           "output_path": "../datasets/test/results/write_test3_1"
         }'
```
返回值
```
{"status":"success","task_id":"task_3_1","result_type":"file","result":"/mnt/data2/bioford/datasets/test/results/write_test3_1.txt"}
```

### 情况 3.2  input_path 网络上传文件
```bash
curl -X 'POST' 'http://0.0.0.0:8101/inference' \
     -H 'Content-Type: application/json' \
     -d '{
           "task_id": "task_3_2",
           "call_back_api": "http://172.17.0.1:5000/callback",
           "input_content": { "input_path" :"https://oxtium-bioford-public.obs.cn-south-1.myhuaweicloud.com/com/oxtium/bioford/test_examples/acme_input.txt" },
           "param_dict": {"mode":"file_file"}
         }'
```
返回值
```
{"status":"success","task_id":"task_3_2","result_type":"file","result":"/mnt/data2/bioford/datasets/test/results/20250423021435_task_3_2.txt"}
```


## 情况4     input: txt    output:string
```bash
curl -X 'POST' 'http://0.0.0.0:8101/inference' \
     -H 'Content-Type: application/json' \
     -d '{
           "task_id": "task_4",
           "call_back_api": "http://172.17.0.1:5000/callback",
           "input_content": { "input_path" : "../datasets/test/input_data.txt"},
           "param_dict": {"mode":"file_value"}
         }'
```
返回值
```
{"status":"success","task_id":"task_4","result_type":"array","result":["CCCCC","input test data! from ../datasets/test/input_data.txt"]}
```


## 情况5     input：混合输入   output: txt
```bash
curl -X 'POST' 'http://0.0.0.0:8101/inference' \
     -H 'Content-Type: application/json' \
     -d '{
           "task_id": "task_5",
           "call_back_api": "http://172.17.0.1:5000/callback",
           "input_content": { "input_path" : "../datasets/test/input_data.txt", "input_strings":["abcd","AAA"]},
           "param_dict": {"mode":"mixed_file"}
         }'
```
返回值
```
{"status":"success","task_id":"task_5","result_type":"file","result":"/mnt/data2/bioford/datasets/test/results/20250423021854_task_5.txt"}
```


