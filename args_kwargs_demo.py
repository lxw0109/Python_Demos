#!/usr/bin/env python3
# coding: utf-8
# File: args_kwargs_demo.py
# Author: lxw
# Date: 10/9/17 11:16 AM
"""
Demos for "*args" & "**kwargs" in Python.
Reference: [Python中如何使用*args和**kwargs](http://python.jobbole.com/83476/)
"""

def func_args(*args):
    print("type(args):{0}\targs:{1}\n".format(type(args), args))


def func_kwargs(**kwargs):
    print("type(kwargs):{0}\tkwargs:{1}\n".format(type(kwargs), kwargs))


def func_args_kwargs(*args, **kwargs):
    print("type(args):{0}\targs:{1}".format(type(args), args))
    print("type(kwargs):{0}\tkwargs:{1}\n".format(type(kwargs), kwargs))


def func(arg1, arg2, arg3):
    print("arg1:{0}\targ2:{1}\targ3:{2}".format(arg1, arg2, arg3))


def main1():
    # 函数定义
    func_args_kwargs(1, 2, 3)
    func_args_kwargs(1, a = 2, b = 3)
    func_args_kwargs(a = 1, b = 2, c = 3)

    func_args(1, 2, 3)
    # func_args(1, a = 2, b = 3)    # TypeError: func_args() got an unexpected keyword argument 'a'
    # func_args(a = 1, b = 2, c = 3)    # TypeError

    # func_kwargs(1, 2, 3)    # TypeError: func_kwargs() takes 0 positional arguments but 3 were given
    # func_kwargs(1, a = 2, b = 3)    # TypeError
    func_kwargs(a = 1, b = 2, c = 3)

    print("---" * 10, "\n")

    # 函数调用
    args  = (1, "two", 3)
    func(*args)

    args  = [1, "two", 3]
    func(*args)
    del args[0]
    print(args)
    func("One", *args)

    args_dic = {"arg3": 3, "arg1": "1", "arg2": "two"}
    func(**args_dic)
    del args_dic["arg1"]
    print(args_dic)
    func(1, **args_dic)

"""
# Output:
type(args):<class 'tuple'>	args:(1, 2, 3)
type(kwargs):<class 'dict'>	kwargs:{}

type(args):<class 'tuple'>	args:(1,)
type(kwargs):<class 'dict'>	kwargs:{'a': 2, 'b': 3}

type(args):<class 'tuple'>	args:()
type(kwargs):<class 'dict'>	kwargs:{'a': 1, 'b': 2, 'c': 3}

type(args):<class 'tuple'>	args:(1, 2, 3)

type(kwargs):<class 'dict'>	kwargs:{'a': 1, 'b': 2, 'c': 3}

------------------------------ 

arg1:1	arg2:two	arg3:3
arg1:1	arg2:two	arg3:3
['two', 3]
arg1:One	arg2:two	arg3:3
arg1:1	arg2:two	arg3:3
{'arg3': 3, 'arg2': 'two'}
arg1:1	arg2:two	arg3:3
"""

def func_default_value(a=1, b=2, c=3):
    print("a:{0}\tb:{1}\tc:{2}".format(a, b, c))

def main():
    # func_default_value(b=3, c=1, 2)    # SyntaxError: positional argument follows keyword argument
    # func_default_value(a=2, 5, c=8)    # SyntaxError: positional argument follows keyword argument
    func_default_value(2, 5, c=8)    # OK
    func_default_value("a", c=8)    # OK
"""
# Output:
a: 2	b: 5	c:8
a: a	b: 2	c:8
"""


if __name__ == '__main__':
    main1()
else:
    print("Being imported as a module.")

