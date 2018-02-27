#!/usr/bin/env python3
# coding: utf-8
# File: utils.py
# Author: A
# Date: 1/30/18 10:22 PM

import traceback


"""
# 1st Version
def show_nums():
    print([i for i in range(10)])
    # some code that may throw exception


# 2nd Version: 装饰器的思想
def add_try_except(func):
    def _func():
        try:
            print("in decorator.")
            func()
        except Exception as e:
            traceback.print_exc()

    return _func

show_nums = add_try_except(show_nums)
"""


# 3rd Version: 装饰器 decorator
def add_try_except(func):
    def _func():
        try:
            print("in @ decorator.")
            func()
        except Exception as e:
            traceback.print_exc()
    return _func


# 3rd Version: 装饰器 decorator
@add_try_except
def show_nums():
    print([i for i in range(10)])
    # some code that may throw exception
