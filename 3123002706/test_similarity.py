"""
单元测试：覆盖文本相似度计算的核心场景
"""
import unittest
from similarity import calculate_cosine_similarity


class TestTextSimilarity(unittest.TestCase):
    # 1. 完全相同文本
    def test_identical_text(self):
        text = "今天天气很好，适合去公园散步"
        self.assertAlmostEqual(calculate_cosine_similarity(text, text), 1.0, places=2)

    # 2. 完全不同文本
    def test_different_texts(self):
        text1 = "苹果香蕉橘子"
        text2 = "西瓜葡萄哈密瓜"
        self.assertAlmostEqual(calculate_cosine_similarity(text1, text2), 0.0, places=2)

    # 3. 样例文本（实际计算≈0.55）
    def test_sample_case(self):
        orig = "今天是星期天，天气晴，今天晚上我要去看电影。"
        plagiarized = "今天是周天，天气晴朗，我晚上要去看电影。"
        self.assertAlmostEqual(calculate_cosine_similarity(orig, plagiarized), 0.55, places=2)

    # 4. 空文本
    def test_empty_text(self):
        self.assertAlmostEqual(calculate_cosine_similarity("", "测试文本"), 0.0, places=2)
        self.assertAlmostEqual(calculate_cosine_similarity("文本", ""), 0.0, places=2)
        self.assertAlmostEqual(calculate_cosine_similarity("", ""), 0.0, places=2)

    # 5. 部分重复文本（实际计算≈0.77）
    def test_partial_overlap(self):
        text1 = "机器学习是人工智能的核心领域"
        text2 = "人工智能的核心领域包括机器学习"
        self.assertAlmostEqual(calculate_cosine_similarity(text1, text2), 0.77, places=2)

    # 6. 纯字母文本（实际计算≈0.0）
    def test_different_length_texts(self):
        text1 = "a" * 100
        text2 = "a" * 50 + "b" * 50
        self.assertAlmostEqual(calculate_cosine_similarity(text1, text2), 0.0, places=2)

    # 7. 含数字文本（实际计算≈0.55）
    def test_text_with_numbers(self):
        text1 = "今天温度25度，适合户外运动"
        text2 = "25度的天气适合户外运动"
        self.assertAlmostEqual(calculate_cosine_similarity(text1, text2), 0.55, places=1)

    # 8. 含英文文本（实际计算≈0.77）
    def test_text_with_english(self):
        text1 = "Python is a popular programming language"
        text2 = "Python is a widely used programming language"
        self.assertAlmostEqual(calculate_cosine_similarity(text1, text2), 0.77, places=2)

    # 9. 停用词过滤
    def test_stopwords_filtering(self):
        text1 = "的 是 我 今天 去 公园"
        text2 = "今天 去 公园"
        self.assertAlmostEqual(calculate_cosine_similarity(text1, text2), 1.0, places=2)

    # 10. 标点符号处理
    def test_punctuation_handling(self):
        text1 = "今天，天气好！"
        text2 = "今天天气好"
        self.assertAlmostEqual(calculate_cosine_similarity(text1, text2), 1.0, places=2)

    # 11. 超长文本（验证性能）
    def test_long_text(self):
        long_text = "测试文本重复100次 " * 100
        self.assertAlmostEqual(calculate_cosine_similarity(long_text, long_text), 1.0, places=2)

    # 12. 特殊符号文本
    def test_special_characters(self):
        text1 = "!@#$%^&*()"
        text2 = "abcdefg"
        self.assertAlmostEqual(calculate_cosine_similarity(text1, text2), 0.0, places=2)


if __name__ == "__main__":
    unittest.main()