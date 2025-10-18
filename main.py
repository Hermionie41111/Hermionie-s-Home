"""
程序入口：解析命令行参数，执行论文查重流程
"""
import sys
from utils import read_file
from similarity import calculate_cosine_similarity


def main():
    """
    主函数：解析命令行参数，执行查重并输出结果
    命令格式：python main.py 原文路径 抄袭文路径 结果输出路径
    """
    if len(sys.argv) != 4:
        print("错误：参数格式应为 'python main.py 原文路径 抄袭文路径 结果输出路径'")
        sys.exit(1)

    orig_path, plag_path, result_path = sys.argv[1], sys.argv[2], sys.argv[3]

    try:
        # 读取文件
        orig_text = read_file(orig_path)
        plag_text = read_file(plag_path)

        # 计算相似度
        similarity = calculate_cosine_similarity(orig_text, plag_text)

        # 写入结果
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(f"文本相似度：{similarity:.2f}")
        print(f"查重完成！结果已写入 {result_path}")

    except FileNotFoundError as e:
        print(f"文件错误：{e}")
    except Exception as e:
        print(f"程序异常：{str(e)}")


if __name__ == "__main__":
    main()