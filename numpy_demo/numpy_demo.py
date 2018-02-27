#!/usr/bin/env python3
# coding: utf-8
# File: numpy_demo.py
# Author: lxw
# Date: 7/12/17 9:24 AM

# 直接参考 [numpy-快速处理数据](http://old.sebug.net/paper/books/scipydoc/numpy_intro.html)

import numpy as np

def demo1():
    b = np.arange(24).reshape(2, 3, 4)
    print("{0}\n{1}\n{2}\n{3}\n".format(b, b.dtype, type(b), b.shape))
    print(b[1, 1, 1])   # 17
    print(b[1][1][1])   # 17
    print(b[:, 1, 1])   # [5 17]
    print(b[:][1][1])   # [16 17 18 19]  NOTE: 这儿注意, 与b[:, 1, 1]的含义不同

    print("\n----------------")
    print(b)
    print(b.ravel())    # 数组的展平操作
    print(b)    # unchanged

    print("\n----------------")
    print(b)
    print(b.flatten())    # 与ravel功能相同, 这个函数会请求分配内存来保存结果
    print(b)    # unchanged

    print("\n----------------")
    print(b.shape)
    print(b)
    b.shape = (6, 4)
    print(b)    # changed
    b.shape = (4, 6)
    print(b)    # changed
    b.reshape(6, 4)
    print(b)    # unchanged
    b = b.reshape(6, 4)
    print(b)    # changed
    c = b * 2   #对b数组的所有元素乘2
    print(c)
    print(np.hsplit(c, 4))
    print(c)    # unchanged
    print(np.vsplit(c, 6))
    print(c)    # unchanged
    # print(np.hsplit(c, 6))    # error
    print(c.ndim)
    print(c.size)
    print(c.itemsize)
    print(c.nbytes)    # size * itemsize
    print(c.T)  # 转置 transpose
    print(c)
    print(c.tolist())

    print("\n----------------")
    d = np.eye(2)
    print("{0}\n{1}\n{2}\n{3}\n".format(d, d.dtype, type(d), d.shape))
    np.savetxt("d.txt", d)
    np.savetxt("c.txt", c)

    print("\n----------------")
    c_from_file = np.loadtxt("c.txt", delimiter=" ", usecols=(0,), unpack=True)
    c_from_file = np.loadtxt("c.txt", delimiter=" ", usecols=(0,1))
    print("{0}\n{1}\n{2}\n{3}\n".format(c_from_file, c_from_file.dtype, type(c_from_file), c_from_file.shape))
    print(np.mean(c_from_file))
    print(np.median(c_from_file))
    print(np.var(c_from_file))
    print(np.std(c_from_file))
    print(np.where(c_from_file > 10))


def demo2():
    print("\n----------------")
    #
    # a = np.zeros(3, 4)   # NO
    np.set_printoptions(threshold=np.nan) # 当数组较大时，不是只打印数据四周的数据， 而是将整个数组完整的打印出来
    a = np.zeros((10, 10))
    print("{0}\n{1}\n{2}\n{3}\n".format(a, a.dtype, type(a), a.shape))
    b = np.ones((12, 13, 14), dtype=np.int16)
    print("{0}\n{1}\n{2}\n{3}\n".format(b, b.dtype, type(b), b.shape))


def demo3():
    """
    fundamental operations
    """
    a = np.array([80, 10, 20, 30, 40, 60])
    b = np.arange(6)
    c = a - b
    print(c)
    d = a ** 2  # 每个元素平方
    print(d)

    print("\n----------------")
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[4, 3], [2, 1]])
    print(a)
    print(b)
    c = a * b   # numpy中的*指按元素计算
    print(c)
    # 矩阵乘法要使用dot函数
    d = np.dot(a, b)
    print(d)

    print("\n----------------")
    print(d.max())
    print(d.min())
    print(d.sum())
    print(d.mean())
    print(np.median(d))
    # print(d.median())   # NO
    # 这些运算默认应用到数组好像它就是一个数字组成的列表，无关数组的形状。然而，指定axis参数你可以吧运算应用到数组指定的轴上
    print(d.max(axis=0))    # 0: column, 1: row
    print(d.min(axis=1))
    print(d.sum(axis=0))
    print(d.mean(axis=0))
    print(np.median(d, axis=1))
    height, width = d.shape
    for i in range(height):
        for j in range(width):
            print(d[i, j])
    print(d.base)


if __name__ == "__main__":
    demo1()
    # demo2()
    # demo3()


