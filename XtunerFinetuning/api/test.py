
import time
def call_model_api(user_input):
    # 初始化提取器 - 从指定路径加载target库
    import requests
    url = "http://localhost:8930/extract_input"  # 替换为你的服务器IP（如果不是本地）
    # 阶段 2
    payload = {
        "user_input": user_input
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # 检查HTTP错误
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"input_text 提取API调用失败: {e}")
        return None
    
if __name__ == "__main__":
    # main()
    # start_time = time.time()
    user_input = '向藏点中写入适合工作'
    resp = call_model_api(user_input)
    print('user_input:', user_input, resp)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f"执行时间：{execution_time}秒")
