model_path = "/app/model_path/test/"

host_docker_path = "/app"

host_mount_path = "/mnt/data1/bioford2"



print(model_path.replace(host_docker_path,host_mount_path))
