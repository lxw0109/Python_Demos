#!/usr/bin/env python3
# coding: utf-8
# File: word_cloud_demo.py
# Author: lxw
# Date: 1/8/18 11:18 PM

# Reference: http://blog.csdn.net/fontthrone/article/details/72775865
from wordcloud import WordCloud

content = open("./txt/constitution.txt").read()
wordcloud = WordCloud(background_color="white",width=1000, height=860, margin=2).generate(content)

# width, height, margin可以设置图片属性

# generate 可以对全部文本进行自动分词,但是他对中文支持不好,对中文的分词处理请看我的下一篇文章
# wordcloud = WordCloud(font_path = r'D:\Fonts\simkai.ttf').generate(f)
# 你可以通过font_path参数来设置字体集

#b ackground_color参数为设置背景颜色,默认颜色为黑色

import matplotlib.pyplot as plt
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

wordcloud.to_file('./img/test.png')
# 保存图片,但是在第三模块的例子中 图片大小将会按照 mask 保存

# Reference: https://github.com/amueller/word_cloud/blob/master/examples/simple.py
"""
Minimal Example
===============
Generating a square wordcloud from the US constitution using default arguments.
"""

from os import path

dir_name = path.dirname(__file__)

text = open(path.join(dir_name, "./txt/constitution.txt")).read()

# Generate a word cloud image
word_cloud = WordCloud().generate(text)

# Display the generated image:
# 1. the matplotlib way:
import matplotlib.pyplot as plt
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")

# lower max_font_size
word_cloud = WordCloud(max_font_size=40).generate(text)
plt.figure()
plt.imshow(word_cloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# 2. The PIL way (if you don't have matplotlib)
image = word_cloud.to_image()
image.show()
