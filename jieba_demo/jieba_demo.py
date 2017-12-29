#!/usr/bin/env python3
# coding: utf-8
# File: jieba_demo.py
# Author: lxw
# Date: 12/22/17 4:03 PM
"""
### References
[jieba](https://github.com/fxsjy/jieba)
"""

import jieba.analyse
import pprint


def main():
    # 1. 分词
    # print('/'.join(jieba.cut('老铁这很稳的。', HMM=False)))    # 老/铁/这/很/稳/的/。
    print('/'.join(jieba.cut('老铁这很稳的。', HMM=True)))    # 老铁/这/很/稳/的/

    # 2. 基于 TF-IDF 算法的关键词抽取
    sentence = """基于 TextRank 算法的关键词抽取基本思想:
    将待抽取关键词的文本进行分词
    以固定窗口大小(默认为5，通过span属性调整)，词之间的共现关系，构建图
    计算图中节点的PageRank，注意是无向带权图
    """
    tags = jieba.analyse.extract_tags(sentence=sentence, topK=20, withWeight=True, allowPOS=())
    # pprint.pprint(tags)
    print(tags)

    # 3. 基于 TextRank 算法的关键词抽取
    sentence = """基于 TextRank 算法的关键词抽取基本思想:
    将待抽取关键词的文本进行分词
    以固定窗口大小(默认为5，通过span属性调整)，词之间的共现关系，构建图
    计算图中节点的PageRank，注意是无向带权图
    """
    # NOTE:这里要和TF-IDF的用法区分开:TF-IDF的用法不需要指定（表示不进行过滤，都留下）; 而textrank的用法需要指定允许的词性
    # tags = jieba.analyse.textrank(sentence=sentence, topK=20, withWeight=True, allowPOS=())
    tags = jieba.analyse.textrank(sentence=sentence, topK=20, withWeight=True, allowPOS=("ns", "n", "vn", "v"))
    print(tags)


if __name__ == '__main__':
    main()

