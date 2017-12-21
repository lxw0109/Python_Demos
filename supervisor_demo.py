#!/usr/bin/env python3
# coding: utf-8
# File: supervisor_demo.py
# Author: lxw
# Date: 11/16/17 4:33 PM

"""
假设让supervisor来监控这个Python小程序
"""

import traceback


def main():
    try:
        pass
    except Exception as e:
        # print(traceback.format_exc())
        traceback.print_exc()


if __name__ == '__main__':
    pass
