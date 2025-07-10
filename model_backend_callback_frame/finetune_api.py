import os
import logging
import requests
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
from urllib.parse import urlparse
import copy
import datetime

import pytz

def get_now_time_path_name(task_id):
    beijing_tz = pytz.timezone('Asia/Shanghai')
    now = datetime.datetime.now(beijing_tz)
    return now.strftime("%Y%m%d%H%M%S") + f"_{task_id}"

# def get_now_time_path_name(task_id):
#     return datetime.datetime.now().strftime("%Y%m%d%H%M%S") + f"_{task_id}"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FintuneRequest(BaseModel):
    task_id: str
    call_back_api: Optional[str] = None
    input_content: Dict
    param_dict: Dict = {}
    model_path: Optional[str] = None
    output_model_path: Optional[str] = None
    output_log_path: Optional[str] = None

class FinetuneAPI:
    def __init__(self, model_name: str, run_finetune_func, model_path: str, finetune_data_dir: str):
        self.model_name = model_name
        self.run_finetune_func = run_finetune_func
        self.model_path = model_path          # "/app/model_path/model_name"
        self.finetune_data_dir = finetune_data_dir     #  "/app/finetune_data_dirsets/model_name"

        self.host_mount_path = os.getenv("HOST_MOUNT_PATH", "/mnt/data1/bioford2")
        
        self.app = FastAPI()
        self.setup_routes()

    def setup_routes(self):
        @self.app.get("/")
        async def read_root():
            return {"message": f"Welcome to {self.model_name} Finetune API!"}

        @self.app.post("/finetune")
        async def finetune(request: FintuneRequest):
            return self.handle_finetune(request)

    def handle_finetune(self, request: FintuneRequest):
        try:
            logging.info(f"Received finetune request: {request}")

            task_name = get_now_time_path_name(request.task_id)

            customized_input = copy.deepcopy(request.input_content["customized_input"])
            host_docker_path = "/"+ self.model_path.split("/")[1]   # /app

            # 自动路径设置          
            task_folder = os.path.join(host_docker_path, f"finetuned_model_path/{self.model_name}/{task_name}")  # /app/finetuned_model_path/model_name/20250423020919_task_2_1
            output_model_path = request.output_model_path or os.path.join(task_folder, "model_files")
            output_log_path = request.output_log_path or os.path.join(task_folder, "finetune.log")

            model_param_path = request.model_path or self.model_path
            response_stream = []


            # 下载datasets 到   "/app/finetune_data_dirsets/model_name/20250423020919_task_2_1/train.extesion" 或者直接使用 request传入的路径
            customized_input["train_dataset_path"]  = self.download_file(task_name, customized_input["train_dataset_path"])
            customized_input["valid_dataset_path"]  = self.download_file(task_name, customized_input["valid_dataset_path"])
            customized_input["test_dataset_path"]  = self.download_file(task_name, customized_input["test_dataset_path"])

            logging.info(f"Customized input after downloading datasets: {customized_input}")

            # use docker paths as inputs
            for output in self.run_finetune_func(customized_input, request.param_dict, model_param_path, output_model_path, output_log_path):
                
                output_model_path = output["output_model_path"].replace(host_docker_path,self.host_mount_path)   # 替换为宿主机位置供后端取得
                output_log_path = output["output_log_path"].replace(host_docker_path,self.host_mount_path)
                
                result = {
                    "status": "success",
                    "task_id": request.task_id,
                    "is_final": output["is_final"],
                    "idx":output["idx"],
                    "fig_param":output["fig_param"],  # dict
                    "test_accuracy":output["test_accuracy"],
                    "output_model_path":output_model_path,
                    "output_log_path":output_log_path
                }
                response_stream.append(result)

                if request.call_back_api:
                    try:
                        requests.post(request.call_back_api, json=result, timeout=10)
                    except Exception as e:
                        logging.error(f"Callback error: {e}")

            return response_stream[-1] if response_stream else {"status": "failure", "task_id": request.task_id, "detail": "No output returned."}

        except Exception as e:
            logging.error(f"Finetune error: {e}")
            return {"status": "failure", "task_id": request.task_id, "detail": str(e)}
        
    def download_file(self, task_name, file_url: str) -> str:
        parsed_url = urlparse(file_url)
        if not parsed_url.scheme.startswith("http"):
            return file_url

        filename = os.path.basename(parsed_url.path)   # train or valid or test.extesion
        
        
        task_dir = os.path.join(self.finetune_data_dir, task_name)
        save_path = os.path.join(task_dir, filename)     # "/app/finetune_data_dirsets/model_name/20250423020919_task_2_1/train.extesion"

        os.makedirs(task_dir, exist_ok=True)

        logging.info(f"Downloading file from {file_url} to {save_path}")
        try:
            response = requests.get(file_url, stream=True)
            response.raise_for_status()
            with open(save_path, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            logging.info(f"File downloaded successfully: {save_path}")
            return save_path
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download file: {e}")
            raise HTTPException(status_code=400, detail=f"Error downloading file: {str(e)}")


    def start(self, host="0.0.0.0", port=8000):
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)
