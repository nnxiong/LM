import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import jieba
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import time
import json

class JSONTagExtractor:
    """使用BGE-M3模型提取JSON标签的类"""
    
    def __init__(self, model_name: str = "/mnt/data1/bge-m3", operation_file_path: str = "operations.json", target_file_path: str = "target.xlsx"):
        """
        初始化提取器
        
        Args:
            model_name: BGE模型名称
            target_file_path: target库Excel文件路径
        """
        # 加载embedding模型
        self.model = SentenceTransformer(model_name)

        # 定义操作类型库
        # self.operations = {
        #     "点击类": ["点击", "新增", "保存", "暂存","删除", "导出", "打印", "提交"],
        #     "勾选类": ["勾选", "选中", "选择", "打勾", "勾上"],
        #     "输入类": ["输入", "录入", "填入", "填写", "写入", "修改", "编辑", "补充","备注","记录","输录","写进","写到"],
        #     "跳转类": ["打开", "跳转", "返回", "前往", "进入", "访问", "切换到"]
        # }
        self.operations = {}
        # 读取JSON文件
        if operation_file_path.endswith("json"):
            with open(operation_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 访问数据
            self.operations = data['operations']
        else:
            # 读取Excel文件（假设第一行是表头）
            df = pd.read_excel(operation_file_path)
            # 转换为字典结构
            for column in df.columns:
                # 去除空值并转换为列表
                values = df[column].dropna().astype(str).tolist()
                self.operations[column] = values
        # 扁平化操作列表，用于向量匹配
        self.operation_list = []
        for op_type, ops in self.operations.items():
            self.operation_list.extend(ops)
        
        # 预计算操作词的embedding
        self.operation_embeddings = self._compute_embeddings(self.operation_list)
        
        # 从Excel文件加载target库
        self._load_targets_from_excel(target_file_path)

    
    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """计算文本列表的embedding向量"""
        return self.model.encode(texts, normalize_embeddings=True)
    

    def _find_best_match(self, query_embedding: np.ndarray, 
                    candidate_embeddings: np.ndarray, 
                    candidates: List[str],
                    original_phrase: str,  # 添加原始短语参数
                    threshold: float = 0.5) -> Tuple[str, float, str]:
        """
        找到最佳匹配项
        
        Args:
            query_embedding: 查询词的embedding
            candidate_embeddings: 候选项的embeddings
            candidates: 候选项列表
            original_phrase: 原始查询中的短语
            threshold: 相似度阈值
            
        Returns:
            最佳匹配项、相似度分数和原始查询中的短语
        """
        # 计算余弦相似度
        similarities = np.dot(candidate_embeddings, query_embedding)
        
        # 对较短的候选项进行轻微的惩罚，避免优先匹配过短的词
        length_penalties = np.array([max(0, 1.0 - 0.05 * (5 - min(len(c), 10) / 2)) for c in candidates])

        adjusted_similarities = similarities * length_penalties
        
        # 找到最大相似度及其索引
        best_idx = np.argmax(adjusted_similarities)
        best_score = similarities[best_idx]  # 返回原始相似度分数，不是经过长度调整的

        # print(f"原始短语: {original_phrase}, 候选: {candidates[best_idx]}, 相似度: {best_score:.4f}")
        
        if best_score >= threshold:
            return candidates[best_idx], best_score, original_phrase
        else:
            return "", 0.0, original_phrase
    
    def _load_targets_from_excel(self, file_path: str):
        """
        从Excel文件加载target库
        
        Args:
            file_path: Excel文件路径
        """
        try:
            # 读取Excel文件
            df = pd.read_excel(file_path)
            
            # 检查必要的列是否存在
            if '原始目标' not in df.columns:
                raise ValueError("Excel文件必须包含'原始目标'列")
            
            # 提取targets和对应的label
            self.targets = df['原始目标'].dropna().tolist()
            
            # 检查是否有level列
            if 'level' in df.columns:
                self.target_labels = {target: label for target, label in zip(df['原始目标'].dropna(), df['level'])}
                
                # 筛选出level为3的目标并计算最大长度
                level3_targets = [target for target, level in self.target_labels.items() if level == 3]
                self.max_target_length_level3 = max([len(target) for target in level3_targets]) if level3_targets else 0
                print(f"Level 3 目标最大长度: {self.max_target_length_level3}")
            else:
                self.target_labels = {target: 1 for target in self.targets}  # 默认label为1
                self.max_target_length_level3 = 0  # 如果没有level列，设为0
            
            # 预计算target的embedding
            self.target_embeddings = self._compute_embeddings(self.targets)

            self.max_target_length = max([len(target) for target in self.targets]) if self.targets else 0
            print(f"成功加载 {len(self.targets)} 个目标词")
        except Exception as e:
            print(f"加载target文件失败: {e}")
            # 使用默认targets作为备选
            self.targets = [
                "按钮", "输入框", "下拉菜单", "复选框", "单选框", "标签页", "链接", 
                "表格", "文本框", "日期选择器", "文件上传", "搜索框", "列表项",
                "主页", "详情页", "设置页", "用户中心", "登录页", "注册页"
            ]
            self.target_labels = {target: 1 for target in self.targets}  # 默认label为1
            self.target_embeddings = self._compute_embeddings(self.targets)
            print("使用默认target库")
    
    def _get_operation_type(self, operation: str) -> str:
        """获取操作的类型"""
        for op_type, ops in self.operations.items():
            if operation in ops:
                return op_type
        return ""
    
    def extract_tags(self, query: str,  ope_threshold=0.88, tar_threshold=0.85) -> Dict[str, Union[str, None]]:
        """从查询中提取JSON标签"""
        # 定义歧义词列表和特殊目标词列表
        ambiguous_words = ["备注", "记录", "标记", "注释"]
        special_target_words = ["亮点", "藏点", "备注点", "备注"]

        # time_start = time.time()

        # 截断处理，操作最长默认4
        scope = self.max_target_length_level3+4+4

        if len(query) > 2*scope:
            query = query[:scope] + query[-scope:]
        # print(f"query: {query}")
        # time_end = time.time()
        # print(f"截断处理耗时: {time_end - time_start:.2f}秒")
        # time_start = time.time()
        # 使用jieba分词
        tokens = list(jieba.cut(query, cut_all=False))
        # time_end = time.time()
        # print(f"分词耗时: {time_end - time_start:.2f}秒")
        # 生成短语
        phrases = []
        for i in range(len(tokens)):
            phrases.append(tokens[i])
            if i < len(tokens) - 1:
                phrases.append(tokens[i] + tokens[i+1])
            if i < len(tokens) - 2:
                phrases.append(tokens[i] + tokens[i+1] + tokens[i+2])
        
        # time_start = time.time()
        # 计算短语的embeddings
        phrase_embeddings = self._compute_embeddings(phrases)
        # time_end = time.time()
        # print(f"短语embedding计算耗时: {time_end - time_start:.2f}秒")
        # 分别存储歧义词和非歧义词候选
        ambiguous_candidates = []
        op_candidates = []
        
        # time_start = time.time()
        # 识别操作词
        for i, phrase_emb in enumerate(phrase_embeddings):
            op, score, original_phrase = self._find_best_match(
                phrase_emb, 
                self.operation_embeddings, 
                self.operation_list,
                phrases[i],  # 传入原始短语
                threshold=ope_threshold  # 提高非歧义词的匹配阈值
            )
            
            if op and score > 0:
                if op in ambiguous_words:
                    ambiguous_candidates.append((op, round(score, ), original_phrase, len(original_phrase)))
                else:
                    op_candidates.append((op, round(score, 3), original_phrase, len(original_phrase)))
        
        # 优先使用非歧义词
        if not op_candidates and ambiguous_candidates:
            op_candidates = ambiguous_candidates
        
        # 如果没找到操作词，强行返回匹配到的操作
        if not op_candidates :
            for i, phrase_emb in enumerate(phrase_embeddings):
                op, score, original_phrase = self._find_best_match(
                    phrase_emb, 
                    self.operation_embeddings, 
                    self.operation_list,
                    phrases[i],  # 传入原始短语
                    threshold=0  # 传入0的阈值作强匹配
                )
                op_candidates.append((op, round(score, 3), original_phrase, len(original_phrase)))
        # print(op_candidates)
        # 选择最佳操作词
        # best_operation, _ , original_operation = sorted(op_candidates, key=lambda x: x[1], reverse=True)[0]
        best_operation, score , original_operation, _ = sorted(op_candidates, key=lambda x: (x[1], x[3]), reverse=True)[0]
        
        # time_end = time.time()
        # print(f"操作词识别耗时: {time_end - time_start:.2f}秒")

        # 创建过滤后的查询用于目标匹配
        filtered_query = query.replace(original_operation, "")  # 使用原始短语进行替换
        
        # filtered_query_length = len(filtered_query)
        # scope = self.max_target_length+8
        # print(f"filtered_query_length: {filtered_query_length}")
        # print(f"max_target_length: {self.max_target_length}")
        # print(f"scope: {scope}")

        # if filtered_query_length > 2*scope:
        #     filtered_query = filtered_query[:scope] + filtered_query[-scope:]
        # time_start = time.time()
        filtered_tokens = list(jieba.cut(filtered_query, cut_all=False))
        # time_end = time.time()
        # print(f"过滤后的分词耗时: {time_end - time_start:.2f}秒")

        # time_start = time.time()
        # 动态计算最大短语长度
        max_phrase_length = min(self.max_target_length, len(filtered_query))

        max_tokens = max(5, int(max_phrase_length / 2))
        # time_end= time.time()
        # print(f"动态计算最大短语长度耗时: {time_end - time_start:.2f}秒")

        # time_start = time.time()
        # 生成目标短语
        target_phrases = set()
        for token in filtered_tokens:
            if len(token) > 1:
                target_phrases.add(token)
        
        for phrase_length in range(2, min(max_tokens + 1, len(filtered_tokens) + 1)):
            for i in range(len(filtered_tokens) - phrase_length + 1):
                phrase = ''.join(filtered_tokens[i:i+phrase_length])
                target_phrases.add(phrase)

        if filtered_query.strip():
            target_phrases.add(filtered_query.strip())
        # time_end = time.time()
        # print(f"目标短语生成耗时: {time_end - time_start:.2f}秒")

        

        # 找出最佳目标
        best_target = ""
        best_target_score = 0
        original_target = ""

        if target_phrases:
            unique_phrases = list(target_phrases)
            
            # 计算所有短语的embeddings
            target_phrase_embeddings = self._compute_embeddings(unique_phrases)
            
            # time_end = time.time()
            # print(f"目标短语embedding计算耗时: {time_end - time_start:.2f}秒")
            
            # time_start = time.time()
            
            # 向量化计算相似度矩阵
            similarity_matrix = np.dot(target_phrase_embeddings, self.target_embeddings.T)
            
            # 创建长度惩罚向量
            length_penalties = np.array([max(0, 1.0 - 0.05 * (5 - min(len(c), 10) / 2)) for c in self.targets])
            
            # 应用长度惩罚
            adjusted_similarities = similarity_matrix * length_penalties
            
            # 为每个短语找出最佳匹配，但保持原有的比较逻辑
            for i, phrase_similarities in enumerate(similarity_matrix):
                # 找出当前短语的最佳匹配
                best_idx = np.argmax(adjusted_similarities[i])
                score = float(phrase_similarities[best_idx])  # 使用原始相似度
                
                # 应用四舍五入保持精度
                score = round(score, 5)
                
                # 应用阈值过滤
                # if score >= 0.85:
                if score >= tar_threshold:
                    target = self.targets[best_idx]
                    orig_phrase = unique_phrases[i]
                    print(target, score)
                    # 原始的比较逻辑
                    if score > best_target_score or (score == best_target_score and len(target) > len(best_target)):
                        best_target = target
                        best_target_score = score
                        original_target = orig_phrase

    
        # time_end = time.time()
        # print(f"目标匹配耗时: {time_end - time_start:.2f}秒")
        # 特殊目标词处理：如果没有找到目标且操作是写入类
        if not best_target and best_operation in self.operations["输入类"]:
            for special_word in special_target_words:
                if special_word in query:
                    best_target = special_word
                    best_target_score = 1.0
                    original_target = special_word  # 特殊词使用自身作为原始词
                    break

        elif not best_target:
            best_target = self.targets[best_idx]
            # print("not best_target: ", best_target, score)
            # orig_phrase = unique_phrases[i]
            # best_target_score = score
            original_target = filtered_query



        # 如果匹配到原句的操作词长度为1，则将其替换为最佳操作词返回给下游
        if len(original_operation) == 1:
            original_operation = best_operation
        # 下游只需要通过op_type判断是否是写入操作
        # time_end = time.time()
        # print(f"特殊目标词处理: {time_end - time_start:.2f}秒")

        return {
            "operation": best_operation,
            "target": best_target,
            "original_operation": original_operation,
            "original_target": original_target,
            "score": best_target_score,
            "op_type": True if self._get_operation_type(best_operation)=="输入类" else False
        }


# 示例用法
def main():
    # 初始化提取器 - 从指定路径加载target库
    extractor = JSONTagExtractor(target_file_path="../data/target.xlsx")
    
    # 测试案例
    test_queries = [
        "点击护士知晓用药不良反应制度及流程掌握50%以下",
        "选中经过三个月以上危重症护理在职培训(计划；考核；记录)",
        "勾选保护患者隐私（减少裸露；不被其他人参观；避免泄露病情；公开场合不讨论患者信息）",
        "勾选保护患者隐私减少裸露；不被其他人参观；避免泄露病情",
        "写入病室整体环境，环境良好无异味",
        "在病室整体环境写入环境良好无异味",
        "在病室整体环境备注了环境良好无异味",
        "在保护患者隐私（减少裸露；不被其他人参观；避免泄露病情；公开场合不讨论患者信息）的备注中写入环境良好无异味",
        "打开围手术期评估记录规范",
        "跳转到围手术期评估记范表",
        "跳转到围手术期评估记录表",
        "跳转到围手术期评估规范表",
        "在医疗垃圾分类处置按规定存放、包装规范中备注废弃物分类标准。",
        "在医疗垃圾分类放置、交接双签字执行到位补充今天已经完成了医疗垃圾分类，并且交接时双方都签字确认了今天已经完成了医疗垃圾分类，并且交接时双方都签字确认了今天已经完成了医疗垃圾分类，并且交接时双方都签字确认了今天已经完成了医疗垃圾分类，并且交接时双方都签字确认了今天已经完成了医疗垃圾分类，并且交接时双方都签字确认了今天已经完成了医疗垃圾分类，并且交接时双方都签字确认了今天已经完成了医疗垃圾分类，并且交接时双方都签字确认了今天已经完成了医疗垃圾分类，并且交接时双方都签字确认了今天已经完成了医疗垃圾分类，并且交接时双方都签字确认了今天已经完成了医疗垃圾分类，并且交接时双方都签字确认了今天已经完成了医疗垃圾分类，并且交接时双方都签字确认了今天已经完成了医疗垃圾分类，并且交接时双方都签字确认了今天已经完成了医疗垃圾分类，并且交接时双方都签字确认了今天已经完成了医疗垃圾分类，并且交接时双方都签字确认了",
        "在医疗垃圾分类放置、交接双签字执行到位处补充需明确责任人并定期进行检查与培训，首先，确保每个医疗垃圾容器上都贴有清晰的标签，标明垃圾类型，如感染性废物、病理性废物、损伤性废物等，以便医务人员正确分类，同时，每个垃圾容器都应设有明显的警示标识，提醒医务人员注意安全，避免接触有害物质，其次，交接环节必须严格执行双签字制度最后，定期进行医疗垃圾管理工作的检查，确保各项措施落实到位，通过上述措施，可以有效提高医疗垃圾管理的质量，保障医患双方的安全。",
        "往备注中写入环境良好无异味",
        "往亮点中写入患者无大碍",
        "往备注点中写入患者隐私不受影响",
        "将患者极度危险填进藏点",
        "在备注中备注患者隐私不受影响",
        "往备注点中备注患者隐私不受影响"
    ]
    # test_queries = ["点击护士知晓用药不良反应制度及流程掌握50%以下"]
    
    print("= Target库信息 =")
    print(f"已加载 {len(extractor.targets)} 个目标词")
    
    for query in test_queries:
        time_start = time.time()
        result = extractor.extract_tags(query)
        time_end = time.time()
        print(f"耗时: {time_end - time_start:.2f}秒")
        print(f"查询: {query}")
        print(f"提取结果: {result}")
        print("-" * 50)


if __name__ == "__main__":
    
    main()
