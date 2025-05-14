import pandas as pd
import json
import time
import os

def create_target_prompt_3(user_input, ope, tar, text):
    # conversation = {
    #     "messages": [
    #         { "role": "system", "content": "现在你是一个从给出的医疗场景下的操作指令中提取标准动作、目标对象和输入内容的人工智能." },
    #         {   "role": "user", 
    #             "content": f'根据以下的用户在医疗场景下的输入："{user_input}"。给出以下格式的的医疗场景下的操作指令："{{"operation": "[标准动作]", "target": "[目标对象]", "input_text": "[输入内容或空字符串]"}}'
    #         },
    #         {
    #             "role": "assistant", 
    #             "content": f'{{"operation": "{ope}", "target": "{tar}", "input_text": "{text}"}}'
    #         }
    #     ]
    # }

    conversation = {
        "messages": [
            {   "role": "user", 
                "content": f'你是一名医疗场景下的操作指令JSON格式化器，需要将用户的操作指令按照JSON格式以中文输出。输出内容包含三个字段：operation、target、input_text。其中operation和target字段的内容都不能为空，只有operation表示输入的意义时，input_text字段才不为空，否则input_text为空。请对用户操作指令"{user_input}"按照上述规则输出。'
            },
            {
                "role": "assistant", 
                "content": f'{{"operation": "{ope}", "target": "{tar}", "input_text": "{text}"}}'
            }
        ]
    }

    # conversation = {"prompt": f'你是一名医疗场景下的操作指令JSON格式化器，需要将用户的操作指令按照JSON格式输出。输出内容包含三个字段：operation、target、input_text。其中operation和target字段都不能为空，只有operation表示输入的意义时，input_text字段才不为空，否则input_text为空。请对用户操作指令"{user_input}"按照上述规则输出。', 
    #                 "answer": f'{{"operation": "{ope}", "target": "{tar}", "input_text": "{text}"}}'
    #                 }
    return conversation

def create_target_prompt_2(user_input, ope, tar):
    # conversation = {"prompt": f'你是一名医疗场景下的操作指令JSON格式化器，需要将用户的操作指令按照JSON格式输出。输出内容包含三个字段：operation、target。其中operation和target字段都不能为空。请对用户操作指令"{user_input}"按照上述规则以JSON格式输出。', 
    #                 "answer": f'{{"operation": "{ope}", "target": "{tar}"}}'
    #                 }
    conversation = {
        "messages": [
            {   "role": "user", 
                "content": f'你是一名医疗场景下的操作指令JSON格式化器，需要将用户的操作指令按照JSON格式以中文输出。输出内容包含两个字段：operation、target。其中operation和target字段的中文内容都不能为空。请对用户操作指令"{user_input}"按照上述规则以JSON格式输出。'
            },
            {
                "role": "assistant", 
                "content": f'{{"operation": "{ope}", "target": "{tar}"}}'
            }
        ]
    }
    return conversation
def construct_instruct03(file_path, save_path):
    # 读取Excel文件
    df = pd.read_excel(file_path)
    prompt = []
    recipe_fine_tunning_num = 0
    # 遍历每一行
    for index, row in df.iterrows():
        user_input = str(row[0])
        try:
            # 解析第二列为字典
            second_column_json = row[1]
            if pd.isna(second_column_json):  # 处理空值
                second_column_json = '{}'
            data_dict = json.loads(second_column_json)
            
            # 提取三个key的值
            operation = data_dict.get("operation", "")
            target = data_dict.get("target", "")
            input_text = data_dict.get("input_text", "")
            
        except json.JSONDecodeError as e:
            print(f"\n错误 - 行 {index + 1}: 无法解析第二列的JSON字符串")
            print(f"错误信息: {e}\n")
            continue
        # convers = create_target_prompt(user_input, operation, target, input_text)
        # if operation in ["输入", "录入", "填入","填写","登记","记录","采集","输录","编写","记载","书写","撰写","添加","写入","填写","记入"]:
        #     convers = create_target_prompt_3(user_input, operation, target, input_text)
        if target not in ["输入", "录入", "填入","填写","登记","记录","采集","输录","编写","记载","书写","撰写","添加","写入","填写","记入"]:
            convers = create_target_prompt_2(user_input, operation, target)
            prompt.append(convers)
            recipe_fine_tunning_num += 1
    
    with open(save_path, 'w', encoding='utf-8') as outfile:
        json.dump(prompt, outfile, ensure_ascii=False, indent=4)
    
    print(F"{recipe_fine_tunning_num}条食谱对应的指令数据集构建完成。并已保存在{save_path}")

def construct_instruct04(file_path, save_path):
    # 读取Excel文件
    df = pd.read_excel(file_path)
    prompt = []
    recipe_fine_tunning_num = 0
    # 遍历每一行
    for index, row in df.iterrows():
        user_input = str(row[0])
        try:
            # 解析第二列为字典
            second_column_json = row[1]
            if pd.isna(second_column_json):  # 处理空值
                second_column_json = '{}'
            data_dict = json.loads(second_column_json)
            
            # 提取三个key的值
            operation = data_dict.get("operation", "")
            target = data_dict.get("target", "")
            input_text = data_dict.get("input_text", "")
            if input_text != "":
                continue
            
        except json.JSONDecodeError as e:
            print(f"\n错误 - 行 {index + 1}: 无法解析第二列的JSON字符串")
            print(f"错误信息: {e}\n")
            continue
        convers = create_target_prompt_2(user_input, operation, target)
        prompt.append(convers)
        recipe_fine_tunning_num += 1
    
    with open(save_path, 'w', encoding='utf-8') as outfile:
        json.dump(prompt, outfile, ensure_ascii=False, indent=4)
    
    print(F"{recipe_fine_tunning_num}条食谱对应的指令数据集构建完成。并已保存在{save_path}")

if __name__ == '__main__':
    data_path = "/mnt/data1/zhongrongmiao/InternLM/data/"
    # json_path = os.path.join(data_path, "level_03_data.xlsx")
    # save_path = os.path.join(data_path, "instruction_dataset_level_03.json")
    # save_path = os.path.join(data_path, "instruction_dataset_level_03_text_input.json")
    json_path = os.path.join(data_path, "level_04_data.xlsx")
    save_path = os.path.join(data_path, "instruction_dataset_level_04.json")
    begin = time.time()
    # construct_instruct03(json_path, save_path)
    construct_instruct04(json_path, save_path)
    times = time.time() - begin
    print(f"指令数据集构建时间：{times}")