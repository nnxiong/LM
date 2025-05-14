import pandas as pd
import json
import time
import os
import random
def create_target_prompt_1(user_input, ope, tar, text):
    conversation = {
        "messages": [
            {   "role": "user", 
                "content": f'你是一名医疗场景下的操作输入提取器，需要将用户操作指令中向目标对象输入的文本提取出来，只需要输出提取出的文本，这对我很重要。现有用户操作指令：“{user_input}”，其中用户的动作为“{ope}”，目标对象为“{tar}”，现在请提取出操作指令中向目标对象输入的文本。'
            },
            {
                "role": "assistant", 
                "content": f'{text}'
            }
        ]
    }
    return conversation

def create_target_prompt_2(user_input, ope, tar, text):
    conversation = {
        "messages": [
            {   "role": "user", 
                "content": f'你是一名医疗场景下的目标对象与输入内容提取器，需要提取出用户操作指令中的目标对象与向目标对象输入的文本，这对我很重要。现有用户操作指令：“{user_input}”，其中用户的动作为“{ope}”，现在请提取出操作指令中的目标对象与向目标对象输入的文本，必须要先输出目标对象，并且两者以空格分开输出。'
            },
            {
                "role": "assistant", 
                "content": f'{tar} {text}'
            }
        ]
    }
    return conversation

def construct_instruct(file_path, save_path):
    # 读取Excel文件
    df = pd.read_excel(file_path)
    prompt = []
    recipe_fine_tunning_num = 0
    # 遍历每一行
    for index, row in df.iterrows():
        if index < 470:
            user_input = str(row[0])
            try:
                # 解析第二列为字典
                second_column_json = row[1]
                if pd.isna(second_column_json):  # 处理空值
                    continue
                data_dict = json.loads(second_column_json)
                
                # 提取三个key的值
                operation = data_dict.get("operation", "")
                target = data_dict.get("target", "")
                input_text = data_dict.get("input_text", "")
                
            except json.JSONDecodeError as e:
                print(f"\n错误 - 行 {index + 1}: 无法解析第二列的JSON字符串")
                print(f"错误信息: {e}\n")
                continue
            convers_1 = create_target_prompt_1(user_input, operation, target, input_text)
            prompt.append(convers_1)
            convers_2 = create_target_prompt_2(user_input, operation, target, input_text)
            prompt.append(convers_2)
            recipe_fine_tunning_num += 1
    random.shuffle(prompt)
    with open(save_path, 'w', encoding='utf-8') as outfile:
        json.dump(prompt, outfile, ensure_ascii=False, indent=4)
    
    print(F"{recipe_fine_tunning_num}条食谱对应的指令数据集构建完成。并已保存在{save_path}")

if __name__ == '__main__':
    data_path = "/mnt/data1/zhongrongmiao/InternLM/data_ft/"
    # json_path = os.path.join(data_path, "train_v2.xlsx")
    json_path = os.path.join(data_path, "train_v2.xlsx")
    # save_path = os.path.join(data_path, "instruction_dataset_train_v2.json")
    save_path = os.path.join(data_path, "instruction_dataset_train_v3_2.json")
    begin = time.time()
    construct_instruct(json_path, save_path)
    times = time.time() - begin
    print(f"指令数据集构建时间：{times}")