"""
相似度计算模块：实现基于余弦相似度的文本相似度计算
"""
from utils import segment_text


def calculate_cosine_similarity(text1, text2):
    """
    计算两个文本的余弦相似度（保留两位小数）
    参数：
        text1 (str): 第一个文本
        text2 (str): 第二个文本
    返回：
        float: 相似度值（0~1）
    """
    # 文本分词
    words1 = segment_text(text1)
    words2 = segment_text(text2)

    # 处理空文本场景
    if not words1 or not words2:
        return 0.0

    # 构建词汇表和词频向量
    vocab = set(words1 + words2)
    vocab_index = {word: i for i, word in enumerate(vocab)}
    vec1 = [0] * len(vocab)
    vec2 = [0] * len(vocab)

    # 统计词频
    for word in words1:
        vec1[vocab_index[word]] += 1
    for word in words2:
        vec2[vocab_index[word]] += 1

    # 计算点积和模长
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = (sum(x ** 2 for x in vec1)) ** 0.5
    norm2 = (sum(x ** 2 for x in vec2)) ** 0.5

    # 避免除零错误
    if norm1 == 0 or norm2 == 0:
        return 0.0

    # 计算并返回余弦相似度（保留两位小数）
    similarity = dot_product / (norm1 * norm2)
    return round(similarity, 2)