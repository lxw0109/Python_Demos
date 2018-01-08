#!/usr/bin/env python3
# coding: utf-8
# File: chinese_word_cloud.py
# Author: lxw
# Date: 1/8/18 11:56 PM

# Reference: http://blog.csdn.net/fontthrone/article/details/72782971
from os import path
from scipy.misc import imread
import matplotlib.pyplot as plt
import jieba
# jieba.load_userdict("txt/userdict.txt")
# 添加用户词库为主词典,原词典变为非主词典
from wordcloud import WordCloud, ImageColorGenerator

# 获取当前文件路径
dir_name = path.dirname(__file__)

stopwords = {}
isCN = 1 #默认启用中文分词
bg_path = "img/bg.jpg"    # 设置背景图片路径
text_path = "txt/chinese_content.txt"
# font_path = "fonts/simkai.ttf'    # 为matplotlib设置中文字体路径
stopwords_path = "txt/stopwords.txt"    # 停用词词表
img1 = "img/ImgShape.png"    # 保存的图片名字1(只按照背景图片形状)
img2 = "img/ImgColor.png"    # 保存的图片名字2(颜色按照背景图片颜色布局生成)

word_list = ["晓伟刘"]    # 在结巴的词库中添加新词

bg_pic = imread(path.join(dir_name, bg_path))    # 设置背景图片

# 设置词云属性
wc = WordCloud(# font_path=font_path,    # 设置字体
               background_color="white",    # 背景颜色
               max_words=2000,    # 词云显示的最大词数
               mask=bg_pic,    # 设置背景图片
               max_font_size=100,    # 字体最大值
               random_state=42,
               width=1000, height=860, margin=2,    # 设置图片默认的大小, 但若使用背景图片, 则保存的图片大小将按照其大小保存
                                                    # margin为词语边缘距离
               )

# 添加自己的词库分词
def add_word(words):
    for word in words:
        jieba.add_word(word)

add_word(word_list)

content = open(path.join(dir_name, text_path)).read()

def jieba_clear_sentence(sentence):
    wo_stop_word_list = []
    seg_generator = jieba.cut(sentence, cut_all=False)
    text_seg_str = "/".join(seg_generator)

    f_stop = open(stopwords_path)
    try:
        stop_words = f_stop.read()
    finally:
        f_stop.close()

    stop_words_set = set(stop_words.split("\n"))
    text_seg_list = text_seg_str.split("/")
    for word in text_seg_list:
        word = word.strip()
        if not(word in stop_words_set) and len(word) > 1:
            wo_stop_word_list.append(word)
    return "".join(wo_stop_word_list)

if isCN:
    content = jieba_clear_sentence(content)

# 生成词云, 可以用generate输入全部文本(wordcloud对中文分词支持不好,建议启用中文分词),也可以我们计算好词频后使用generate_from_frequencies函数
wc.generate(content)
# wc.generate_from_frequencies(txt_freq)    # txt_freq例子为[('词a', 100),('词b', 90),('词c', 80)]
# 从背景图片生成颜色值
image_colors = ImageColorGenerator(bg_pic)

plt.figure()
# 以下代码显示图片
plt.imshow(wc)
plt.axis("off")
plt.show()
# 绘制词云

# 保存图片
wc.to_file(path.join(dir_name, img1))

icg = ImageColorGenerator(bg_pic)

plt.imshow(wc.recolor(color_func=icg))
plt.axis("off")
# 绘制背景图片为颜色的图片
plt.figure()
plt.imshow(bg_pic, cmap=plt.cm.gray)
plt.axis("off")
plt.show()
# 保存图片
wc.to_file(path.join(dir_name, img2))