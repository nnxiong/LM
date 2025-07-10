# multi-functions
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

    # curl -X 'POST' 'http://0.0.0.0:8101/inference' \
    #     -H 'Content-Type: application/json' \
    #     -d '{
    #         "task_id": "task_1",
    #         "call_back_api": "http://172.17.0.1:5000/callback",
    #         "input_content": { "input_strings" :["AAAAAAA","BBBBBB"]},
    #         "param_dict": {"mode":"value_value"}
    #         }'

    # {"status":"success","task_id":"task_1","result_type":"array","result":["CCCCC","AAAAAAA","BBBBBB"]}



    # 情况2     input: string    output:txt
    print("========== case 2 =============")
    # post output_path/default_output_path_ext 如果输出是file，必须二选一
    result2 = run_test_inference({ "input_strings" :["AAAAAAA","BBBBBB"]}, {"mode":"value_file"}, model_path = "../model_path/test", output_path_name = "/app/datasets/test/results/write_test_case2")
    print("result2:",result2)

    # 情况 2.1  output_path 本地路径。  post 必须是docker框架内路径

    # ```bash
    # curl -X 'POST' 'http://0.0.0.0:8101/inference' \
    #      -H 'Content-Type: application/json' \
    #      -d '{
    #            "task_id": "task_2_1",
    #            "call_back_api": "http://172.17.0.1:5000/callback",
    #            "input_content": { "input_strings" :["AAAAAAA","BBBBBB"]},
    #            "param_dict": {"mode":"value_file"},
    #            "output_path":"/app/datasets/test/results/write_test2_1"
    #          }'
    # ```
    # 返回值
    # ```
    # {"status":"success","task_id":"task_2_1","result_type":"file","result":"/mnt/data2/bioford/datasets/test/results/write_test2_1.txt"}
    # ```

    # # 情况 2.2  output_path 默认文件路径。 不指定 output_path时，系统生成默认格式的路径
    # ```bash
    # curl -X 'POST' 'http://0.0.0.0:8101/inference' \
    #      -H 'Content-Type: application/json' \
    #      -d '{
    #            "task_id": "task_2_2",
    #            "call_back_api": "http://172.17.0.1:5000/callback",
    #            "input_content": { "input_strings" :["AAAAAAA","BBBBBB"]},
    #            "param_dict": {"mode":"value_file"}
    #          }'
    # ```
    # 返回值
    # ```
    # {"status":"success","task_id":"task_2_1","result_type":"file","result":"/mnt/data2/bioford/datasets/test/results/20250423020919_task_2_1.txt"}
    # ```

    # 情况3     input: txt    output:txt
    print("========== case 3 =============")
    result3 = run_test_inference({ "input_path" : "../datasets/test/input_data.txt"}, {"mode":"file_file"}, model_path = "../model_path/test", output_path_name = "/app/datasets/test/results/write_test_case3")
    print("result3:",result3)

    #  # 情况 3.1  input_path 本地路径
    # ```bash
    # curl -X 'POST' 'http://0.0.0.0:8101/inference' \
    #      -H 'Content-Type: application/json' \
    #      -d '{
    #            "task_id": "task_3_1",
    #            "call_back_api": "http://172.17.0.1:5000/callback",
    #            "input_content": { "input_path" : "../datasets/test/input_data.txt"},
    #            "param_dict": {"mode":"file_file"},
    #            "output_path": "../datasets/test/results/write_test3_1"
    #          }'
    # ```
    # 返回值
    # ```
    # {"status":"success","task_id":"task_3_1","result_type":"file","result":"/mnt/data2/bioford/datasets/test/results/write_test3_1.txt"}
    # ```

    # ### 情况 3.2  input_path 网络上传文件
    # ```bash
    # curl -X 'POST' 'http://0.0.0.0:8101/inference' \
    #      -H 'Content-Type: application/json' \
    #      -d '{
    #            "task_id": "task_3_2",
    #            "call_back_api": "http://172.17.0.1:5000/callback",
    #            "input_content": { "input_path" :"https://oxtium-bioford-public.obs.cn-south-1.myhuaweicloud.com/com/oxtium/bioford/test_examples/acme_input.txt" },
    #            "param_dict": {"mode":"file_file"}
    #          }'
    # ```
    # 返回值
    # ```
    # {"status":"success","task_id":"task_3_2","result_type":"file","result":"/mnt/data2/bioford/datasets/test/results/20250423021435_task_3_2.txt"}
    # ```


    # 情况4     input: txt    output:string
    print("========== case 4 =============")
    result4 = run_test_inference({ "input_path" : "../datasets/test/input_data.txt"}, {"mode":"file_value"}, model_path = "../model_path/test")
    print("result4 :",result4)

    # ## 情况4     input: txt    output:string
    # ```bash
    # curl -X 'POST' 'http://0.0.0.0:8101/inference' \
    #      -H 'Content-Type: application/json' \
    #      -d '{
    #            "task_id": "task_4",
    #            "call_back_api": "http://172.17.0.1:5000/callback",
    #            "input_content": { "input_path" : "../datasets/test/input_data.txt"},
    #            "param_dict": {"mode":"file_value"}
    #          }'
    # ```
    # 返回值
    # ```
    # {"status":"success","task_id":"task_4","result_type":"array","result":["CCCCC","input test data! from ../datasets/test/input_data.txt"]}
    # ```


    # 情况5 input：混合输入   output: txt
    print("========== case 5 =============")
    result5 = run_test_inference({ "input_path" : "../datasets/test/input_data.txt", "input_strings":["case5","case5_input_string"]}, {"mode":"mixed_file"}, model_path = "../model_path/test", output_path_name = "/app/datasets/test/results/write_test_case5")
    print("result5 :",result5)
    
    # ```bash
    # curl -X 'POST' 'http://0.0.0.0:8101/inference' \
    #      -H 'Content-Type: application/json' \
    #      -d '{
    #            "task_id": "task_5",
    #            "call_back_api": "http://172.17.0.1:5000/callback",
    #            "input_content": { "input_path" : "../datasets/test/input_data.txt", "input_strings":["abcd","AAA"]},
    #            "param_dict": {"mode":"mixed_file"}
    #          }'
    # ```
    # 返回值
    # ```
    # {"status":"success","task_id":"task_5","result_type":"file","result":"/mnt/data2/bioford/datasets/test/results/20250423021854_task_5.txt"}
    # ```



# python test_inference_v2.py
# ========== case 1 =============
# result1: ('array', ['CCCCC', 'AAAAAAA', 'BBBBBB'])
# ========== case 2 =============
# result2: ('file', '/app/datasets/test/results/write_test_case2.txt')
# ========== case 3 =============
# result3: ('file', '/app/datasets/test/results/write_test_case3.txt')
# ========== case 4 =============
# result4 : ('array', ['CCCCC', 'input test data! from ../datasets/test/input_data.txt'])
# ========== case 5 =============
# result5 : ('file', '/app/datasets/test/results/write_test_case5.txt')