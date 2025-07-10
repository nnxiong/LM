docker run -it --rm \
  -p 8101:8101 \
  -v /mnt/data2/bioford:/app \
  -e HOST_MOUNT_PATH="/mnt/data2/bioford" \
  ablang_api

-----
docker run -it \
  -v /mnt/data1/bioford2:/app \
  --network=host \
  -e HOST_MOUNT_PATH="/mnt/data1/bioford2" \
  -w /app/test \
  --name bf_test_env \
  --rm \
  test_model_api

python3 test_finetune.py
python3 test_finetune_fastapi.py 

