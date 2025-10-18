"""相似度计算器：实现余弦相似度、Jaccard相似度计算"""
import math


class SimilarityCalculator:
    @staticmethod
    def calculate_cosine_similarity(vector1, vector2):
        """计算两个向量的余弦相似度"""
        if not vector1 or not vector2:
            return 0.0

        # 合并所有键（词汇表）
        all_keys = set(vector1.keys()).union(set(vector2.keys()))
        # 计算点积
        dot_product = 0.0
        for key in all_keys:
            dot_product += vector1.get(key, 0) * vector2.get(key, 0)
        # 计算模长
        norm1 = math.sqrt(sum(v ** 2 for v in vector1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in vector2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0
        return round(dot_product / (norm1 * norm2), 4)

    @staticmethod
    def calculate_jaccard_similarity(vector1, vector2):
        """计算两个向量的Jaccard相似度（交集/并集）"""
        if not vector1 or not vector2:
            return 0.0

        set1 = set(vector1.keys())
        set2 = set(vector2.keys())
        intersection = set1 & set2
        union = set1 | set2

        if not union:
            return 0.0
        return round(len(intersection) / len(union), 4)