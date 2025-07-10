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
