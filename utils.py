"""
工具函数模块：负责文件读写和文本预处理（分词、清洗）
"""
import os
import re
import jieba
from functools import lru_cache

# 停用词表（过滤无意义中文词汇）
STOPWORDS = {'的', '了', '是', '我', '都', '在', '也', '就', '和', '很', '到', '说', '要', '去', '人', '有'}


def read_file(file_path):
    """
    读取文件，兼容utf-8和gbk编码
    参数：
        file_path (str): 文件路径
    返回：
        str: 文件内容（已去除首尾空白）
    异常：
        FileNotFoundError: 文件不存在时抛出
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"错误：文件不存在 → {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='gbk') as f:
            return f.read().strip()


@lru_cache(maxsize=128)  # 缓存分词结果，避免重复计算
def segment_text(text):
    """
    文本分词与清洗：
    1. 过滤非中文/英文/数字的字符；
    2. 中文用jieba分词，英文/数字按词保留；
    3. 过滤停用词和空字符串。
    参数：
        text (str): 原始文本
    返回：
        list[str]: 清洗后的分词列表
    """
    if not text:
        return []
    # 过滤特殊字符，仅保留中文、英文、数字和空格
    text = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', text)
    # 分离中文与非中文部分
    chinese_parts = re.findall(r'[\u4e00-\u9fa5]+', text)
    non_chinese_parts = re.split(r'\s+', text)  # 英文/数字按空格分割

    words = []
    # 中文部分用jieba分词
    for ch_part in chinese_parts:
        words.extend(jieba.lcut(ch_part))
    # 非中文部分直接加入（已按空格分割）
    for part in non_chinese_parts:
        if part.strip():
            words.append(part)
    # 过滤停用词和空字符串
    return [word for word in words if word not in STOPWORDS and word.strip()]