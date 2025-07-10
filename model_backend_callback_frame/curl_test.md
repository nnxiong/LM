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