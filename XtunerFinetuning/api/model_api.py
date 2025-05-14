from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from target import JSONTagExtractor
import time
# 初始化 FastAPI 应用
app = FastAPI()

# 加载模型和分词器
# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and cause OOM Error.
# tokenizer = AutoTokenizer.from_pretrained("../merged_v2", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("../merged_v2", torch_dtype=torch.float16, trust_remote_code=True).cuda()
tokenizer = AutoTokenizer.from_pretrained("../merged_v3", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("../merged_v3", torch_dtype=torch.float16, trust_remote_code=True).cuda()
model = model.eval()
extractor = JSONTagExtractor(operation_file_path="../data/operation.xlsx", target_file_path="../data/target.xlsx")

# 定义请求模型
class RequestBody(BaseModel):
  user_input: str
  ope_threshold:float = 0.88
  tar_threshold:float = 0.86

# 定义 API 端点
@app.post("/extract_input")
def extract_input(request_body: RequestBody):
  # 阶段 1
  # No manual Threshold
  start_time1 = time.time()
  # result = extractor.extract_tags(request_body.user_input)
  # # manual Threshold
  result = extractor.extract_tags(request_body.user_input, request_body.ope_threshold, request_body.tar_threshold)
  operation = result["original_operation"]
  target = result["original_target"]
  operation_in_database = result["operation"]
  target_in_database = result["target"]
  operation_score = str(result["operation_score"])
  target_score = str(result["target_score"])

  end_time1 = time.time()
  execution_time1 = end_time1 - start_time1
  print(f"阶段 1 执行时间：{execution_time1}秒")

  response = ""
  if result["op_type"] == True:
    # 阶段 2
    with torch.no_grad():
      start_time2 = time.time()
      if result["original_target"] != "":
        query = f'你是一名医疗场景下的操作输入提取器，需要将用户操作指令中向目标对象输入的文本提取出来，只需要输出提取出的文本，这对我很重要。现有用户操作指令：“{request_body.user_input}”，其中用户的动作为“{operation}”，目标对象为“{target}”，现在请提取出操作指令中向目标对象输入的文本。'
        response, _ = model.chat(tokenizer, query, history=None)

      else:
        query = f'你是一名医疗场景下的目标对象与输入内容提取器，需要提取出用户操作指令中的目标对象与向目标对象输入的文本，这对我很重要。现有用户操作指令：“{request_body.user_input}”，其中用户的动作为“{target}”，现在请提取出操作指令中的目标对象与向目标对象输入的文本，两者以空格分开输出。'
        output, _ = model.chat(tokenizer, query, history=None)
        print(output)
        tar_input = output.split(" ")
        target = tar_input[0]
        response = tar_input[1]

      # print(str(response).split("\n"))
      end_time2 = time.time()
      execution_time2 = end_time2 - start_time2
      print(f"阶段 2 执行时间：{execution_time2}秒")
      return {"operation": operation,"target": target, "operation_in_database": operation_in_database,
              "target_in_database": target_in_database,"operation_score": operation_score,"target_score": target_score,
              "input_text": response}

  else:
    return {"operation": operation,"target": target, "operation_in_database": operation_in_database,
            "target_in_database": target_in_database,"operation_score": operation_score,"target_score": target_score,
            "input_text": response}
  
    # 启动 API 服务
if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8930)

