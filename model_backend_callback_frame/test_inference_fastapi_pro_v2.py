import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference_api_v2 import InferenceAPI
from test_inference_v2 import run_test_inference

# 创建 API 实例
api = InferenceAPI(
    model_name="test",
    run_inference_func=run_test_inference,
    model_path="/app/model_path/test/",
    data_dir="/app/datasets/test",
    input_webpath_key = "input_path",  # 需要下载的文件路径在 input_content中的key 【如果模型有功能需要input_path就需要给出key】   
)

if __name__ == "__main__":
    api.start(port=8101)
