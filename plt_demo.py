#!/usr/bin/env python3
# coding: utf-8
# File: plt_demo.py
# Author: lxw
# Date: 11/12/17 9:18 PM


import matplotlib.pyplot as plt

def main():
    x = [1, 2, 3]
    y = [4, 5, 6]

    x1 = [11, 21, 13]
    y1 = [15, 10, 14]

    x2 = [1, 11, 23]
    y2 = [13, 8, 24]

    plt.plot(x, y, label="P")
    plt.plot(x1, y1, label="R")
    plt.plot(x2, y2, label="F")
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()