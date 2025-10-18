"""文本处理器：负责文本清洗、分词、n-gram生成、向量化"""
import re
import jieba

class TextProcessor:
    def __init__(self, ngram_size=1):  # 调整默认ngram为1（单词级别）
        self.ngram_size = ngram_size
        self.stopwords = {'的', '了', '是', '我', '都', '在', '也', '就', '和', '很', '到', '说', '要', '去', '人', '有'}

    def clean_text(self, text):
        """清洗文本：去除所有标点和特殊字符，仅保留文字、数字和空格"""
        if text is None:
            return ""
        # 过滤规则：只保留中文、英文、数字和空格
        cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', str(text))
        return cleaned.strip()

    def segment_text(self, text):
        """分词：中文用jieba，过滤停用词和空字符串"""
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return []
        # 中文分词（精确模式）
        words = jieba.lcut(cleaned_text)
        # 过滤停用词和空词（strip()去除空格）
        return [word for word in words if word.strip() and word not in self.stopwords]

    def generate_ngrams(self, words):
        """生成n-gram序列（n=1时直接返回单词列表）"""
        if self.ngram_size == 1:
            return words  # 单词级别无需拼接
        if len(words) < self.ngram_size:
            return []
        ngrams = []
        for i in range(len(words) - self.ngram_size + 1):
            ngram = ''.join(words[i:i+self.ngram_size])  # 中文n-gram拼接为字符串
            ngrams.append(ngram)
        return ngrams

    def text_to_vector(self, text):
        """文本向量化：统计n-gram的词频（返回字典{ngram: 次数}）"""
        words = self.segment_text(text)
        ngrams = self.generate_ngrams(words)
        vector = {}
        for ngram in ngrams:
            vector[ngram] = vector.get(ngram, 0) + 1
        return vector