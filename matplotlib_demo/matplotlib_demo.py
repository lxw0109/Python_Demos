#!/usr/bin/env python3
# coding: utf-8
# File: plt_demo.py
# Author: lxw
# Date: 11/12/17 9:18 PM

# 利用ipython --pylab，可以进入PyLab模式，已经导入了matplotlib库与相关软件包（例如Numpy和Scipy)，可以直接使用相关库的功能


def basic_demo():
    import matplotlib.pyplot as plt
    x = [1, 2, 3]
    y = [4, 5, 6]

    x1 = [11, 21, 13]
    y1 = [15, 10, 14]

    x2 = [1, 11, 23]
    y2 = [13, 8, 24]

    plt.plot(x, y, label="P")
    plt.plot(x1, y1, label="R")
    plt.plot(x2, y2, label="F")
    plt.legend()    # 显示图例

    plt.show()


def draw_function():
    """
    绘制任何函数的曲线
    Reference:
    1. [利用matplotlib画sigmoid函数](https://www.douban.com/note/630433448/)
    2. [python使用matplotlib:subplot绘制多个子图](http://blog.csdn.net/gatieme/article/details/61416645)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import math

    def sigmoid(x):
        y = []
        for item in x:
            y.append(1 / (1 + math.exp(-item)))
        return y

    def func(x):
        y = []
        for item in x:
            if item <= 0:
                y.append(0)
            else:
                y.append(-item * math.log2(item) - (1 - item) * math.log2(1 - item))
        return y


    x = np.arange(-10, 10, 0.2)    # arange(起点, 终点, 间隔). linspace(起点, 终点, 点个数)
    y = sigmoid(x)
    plt.subplot(121)
    plt.plot(x, y)

    x = np.arange(0, 1, 0.01)    # arange(起点, 终点, 间隔). linspace(起点, 终点, 点个数)
    y = func(x)
    plt.subplot(122)
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    basic_demo()
    # draw_function()
