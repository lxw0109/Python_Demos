#!/usr/bin/env python3
# coding: utf-8
# File: word_cloud_masked_demo.py
# Author: lxw
# Date: 1/8/18 11:36 PM

# Reference: https://github.com/amueller/word_cloud/edit/master/examples/masked.py
"""
Masked wordcloud
================

Using a mask you can generate wordclouds in arbitrary shapes.
"""

from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS

dir_name = path.dirname(__file__)

# Read the whole text.
content = open(path.join(dir_name, "./txt/constitution.txt")).read()

# read the mask image
# taken from
# http://www.stencilry.org/stencils/movies/alice%20in%20wonderland/255fk.jpg
mask = np.array(Image.open(path.join(dir_name, "./img/alice_mask.png")))
# mask = np.array(Image.open(path.join(dir_name, "lxw_logo_original.png")))    # NO

stopwords = set(STOPWORDS)
stopwords.add("said")

wc = WordCloud(background_color="white", max_words=2000, mask=mask, stopwords=stopwords)
# generate word cloud
wc.generate(content)

# store to file
wc.to_file(path.join(dir_name, "./img/alice.png"))

# show
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.figure()
plt.imshow(mask, cmap=plt.cm.gray, interpolation='bilinear')
plt.axis("off")
plt.show()
