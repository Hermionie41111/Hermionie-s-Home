"""论文查重检测器：整合文本处理和相似度计算，处理文件读写"""
import os


class PlagiarismDetector:
    def __init__(self, ngram_size=2):
        from text_processor import TextProcessor
        from similarity_calculator import SimilarityCalculator
        self.processor = TextProcessor(ngram_size=ngram_size)
        self.calculator = SimilarityCalculator()

    def read_file(self, file_path):
        """读取文件内容（兼容utf-8/gbk）"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在：{file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='gbk') as f:
                return f.read()

    def process_files(self, orig_path, plag_path, output_path):
        """处理文件：计算相似度并写入结果"""
        try:
            # 读取文本
            orig_text = self.read_file(orig_path)
            plag_text = self.read_file(plag_path)

            # 向量化
            orig_vector = self.processor.text_to_vector(orig_text)
            plag_vector = self.processor.text_to_vector(plag_text)

            # 计算余弦相似度（主指标）
            similarity = self.calculator.calculate_cosine_similarity(orig_vector, plag_vector)

            # 写入结果
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"{similarity:.4f}")
            return True
        except Exception as e:
            print(f"处理失败：{e}")
            return False