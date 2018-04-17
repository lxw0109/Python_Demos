#!/usr/bin/env python3
# coding: utf-8
# File: multiprocessing_demo.py
# Author: lxw
# Date: 7/4/17 8:56 AM
"""
References:
1. [Python 进程之间共享数据(全局变量)](https://blog.csdn.net/houyanhua1/article/details/78244288)
"""

import multiprocessing
import os

# 0. 使用全局变量在多个进程间通信(不可行)
global_val = [27017]


def func0(value):
    global global_val
    global_val.append(value)
    print("[func0()]pid:", os.getpid(), ", global_val:", global_val, ", id(global_val):", id(global_val))


def global_var_IPC():
    """
    直接使用全局变量在多个进程间通信是不可行的
    """
    global global_val
    print("[main]pid:", os.getpid(), ", global_val:", global_val, ", id(global_val):", id(global_val))
    processes = []
    for i in range(5):
        process = multiprocessing.Process(target=func0, args=(i, ))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
    print("[main]pid:", os.getpid(), ", global_val:", global_val, ", id(global_val):", id(global_val))


# 1. 进程间共享数值型数据
def func1(obj):
    # 子进程修改数值的值, 主进程跟着改变
    print("[func1]pid:", os.getpid(), ", obj.value:", obj.value, ", id(obj.value):", id(obj.value))
    obj.value = 10.78
    print("[func1]pid:", os.getpid(), ", obj.value:", obj.value, ", id(obj.value):", id(obj.value))


def num_IPC():
    # d表示ctypes.c_double, 主进程与子进程共享这个value.
    # multi_v = multiprocessing.Value("d", 10)
    multi_v = multiprocessing.Value("d", 10)
    """
    typecode_to_type = {
        'c': ctypes.c_char,  'u': ctypes.c_wchar,
        'b': ctypes.c_byte,  'B': ctypes.c_ubyte,
        'h': ctypes.c_short, 'H': ctypes.c_ushort,
        'i': ctypes.c_int,   'I': ctypes.c_uint,
        'l': ctypes.c_long,  'L': ctypes.c_ulong,
        'f': ctypes.c_float, 'd': ctypes.c_double
    }
    """
    print("[main]pid:", os.getpid(), ", multi_v.value:", multi_v.value, ", id(multi_v.value):", id(multi_v.value))

    p = multiprocessing.Process(target=func1, args=(multi_v, ))
    p.start()
    p.join()
    print("[main]pid:", os.getpid(), ", multi_v.value:", multi_v.value, ", id(multi_v.value):", id(multi_v.value))


class MyObj:
    def __init__(self, val):
        self.value = val


def num_IPC_1():
    myobj = MyObj(10)
    print("[main]pid:", os.getpid(), ", myobj.value:", myobj.value, ", id(myobj.value):", id(myobj.value))

    p = multiprocessing.Process(target=func1, args=(myobj, ))
    p.start()
    p.join()
    print("[main]pid:", os.getpid(), ", myobj.value:", myobj.value, ", id(myobj.value):", id(myobj.value))


# 2. 进程间共享数组型数据
def func2(array):
    print("[func2]pid:", os.getpid(), ", array:", array[:], ", id(array):", id(array))
    array[2] = -110    # 子进程改变数组, 主进程跟着改变
    print("[func2]pid:", os.getpid(), ", array:", array[:], ", id(array):", id(array))


def array_IPC():
    array = multiprocessing.Array("i", [1, 2, 3, 4, 5, 6])
    print("[main]pid:", os.getpid(), ", array:", array[:], ", id(array):", id(array))

    p = multiprocessing.Process(target=func2, args=(array, ))
    p.start()
    p.join()
    print("[main]pid:", os.getpid(), ", array:", array[:], ", id(array):", id(array))


# 3. 进程间共享dict,list数据
def func3(mydic, mylist):
    print("[func3]pid:", os.getpid(), ", mydic:", mydic, ", id(mydic):", id(mydic))
    print("[func3]pid:", os.getpid(), ", mylist:", mylist, ", id(mylist):", id(mylist))
    mydic["index1"] = "value1"    # 子进程改变dict, 主进程跟着改变
    mydic["index2"] = "value2"
    mylist.append(111)    # 子进程改变list, 主进程跟着改变
    mylist.append(222)
    mylist.append(333)
    print("[func3]pid:", os.getpid(), ", mydic:", mydic, ", id(mydic):", id(mydic))
    print("[func3]pid:", os.getpid(), ", mylist:", mylist, ", id(mylist):", id(mylist))


def dict_list_IPC():
    with multiprocessing.Manager() as mg:
        mydic = multiprocessing.Manager().dict()
        mylist = multiprocessing.Manager().list(range(5))
        print("[main]pid:", os.getpid(), ", mydic:", mydic, ", id(mydic):", id(mydic))
        print("[main]pid:", os.getpid(), ", mylist:", mylist, ", id(mylist):", id(mylist))

        p = multiprocessing.Process(target=func3, args=(mydic, mylist))
        p.start()
        p.join()

        print("[main]pid:", os.getpid(), ", mydic:", mydic, ", id(mydic):", id(mydic))
        print("[main]pid:", os.getpid(), ", mylist:", mylist, ", id(mylist):", id(mylist))


if __name__ == "__main__":
    # 0. 使用全局变量在多个进程间通信(不可行)
    # global_var_IPC()

    # 1. 进程间共享数值型数
    """
    num_IPC()
    print("--" * 30)
    num_IPC_1()
    """

    # 2. 进程间共享数组型数据
    # array_IPC()

    # 3. 进程间共享dict,list数据
    dict_list_IPC()
