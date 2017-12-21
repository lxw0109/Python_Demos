#!/usr/bin/env python3
# coding: utf-8
# File: multiprocessing_demo.py
# Author: lxw
# Date: 7/4/17 8:56 AM

import multiprocessing
# import sys
import time

def func(num):
    print(num, end=" ", flush=True)    # 增加flush=True才能体现出多进程和进程池
    # sys.stdout.flush()    # 增加这条语句才能体现出多进程和进程池
    time.sleep(1)
    return num

def single_param(value):
    print(value, end=" ")

def multiple_params(value1, value2, value3):
    print("in multiple_params().")
    print(value1, end=" ")
    print(value2, end=" ")
    print(value3, end=" ")

def multi_args():
    with multiprocessing.Pool(5) as pool:
        for i in range(1, 4):
            pool.apply(single_param, (i,))
    print("\n", "--" * 20, "\n")
    with multiprocessing.Pool(5) as pool:
        for v1, v2, v3 in [(1, 2, 3), (4, 5, 6)]:
            pool.apply(multiple_params, (v1, v2, v3))
    print("\n", "--" * 20, "\n\n")

    with multiprocessing.Pool(5) as pool:
        for i in range(1, 4):
            pool.apply_async(single_param, (i,))
        pool.close()
        pool.join()
    print("\n", "--" * 20, "\n")
    with multiprocessing.Pool(5) as pool:
        for i in [(1, 2, 3), (4, 5, 6)]:
            pool.apply_async(multiple_params, i)
        pool.close()
        pool.join()
    print("\n", "--" * 20, "\n\n")

    with multiprocessing.Pool(5) as pool:
        pool.map(single_param, [1, 2, 3])
    print("\n", "--" * 20, "\n")
    """
    # map 无法使用多个参数
    with multiprocessing.Pool(5) as pool:
        # pool.map(multiple_params, [(1, 2, 3), (4, 5, 6)])    # Error
        # pool.map(multiple_params, [[1, 2, 3], [4, 5, 6]])    # Error
        # pool.map(multiple_params, [[1, 2, 3]])    # Error
        # pool.map(multiple_params, (1, 2, 3))    # Error
    print("\n", "--" * 20, "\n\n")
    """

    with multiprocessing.Pool(5) as pool:
        pool.map_async(single_param, [1, 2, 3])
        pool.close()
        pool.join()
    print("\n", "--" * 20, "\n")
    """
    # map_async 无法使用多个参数
    with multiprocessing.Pool(5) as pool:
        # pool.map_async(multiple_params, [(1, 2, 3), (4, 5, 6)])      # multiple_params is never called.
        pool.map_async(multiple_params, ((1, 2, 3), (4, 5, 6)))      # multiple_params is never called.
        pool.close()
        pool.join()
    print("\n", "--" * 20, "\n\n")
    """


def main():
    """
    不带返回值的多进程
    """
    COUNT = 20

    print("\napply_async", "--" * 20)
    t = time.time()
    with multiprocessing.Pool(5) as f:
        for i in range(COUNT):
            f.apply_async(func, (i,))   # 多进程
        f.close()
        f.join()
    print("\napply_async:", time.time() - t, "\n")

    print("\napply", "--" * 20)
    t = time.time()
    with multiprocessing.Pool(5) as f:
        for i in range(COUNT):
            f.apply(func, (i,))    # 不是多进程
    print("\napply:", time.time() - t, "\n")

    print("\nmap", "--" * 20)
    t = time.time()
    with multiprocessing.Pool(5) as f:
        f.map(func, range(COUNT))    # 多进程
    print("\nmap:", time.time() - t, "\n")

    print("\nmap_async", "--" * 20)
    t = time.time()
    with multiprocessing.Pool(5) as f:
        f.map_async(func, range(COUNT))    # 多进程
        f.close()
        f.join()
    print("\nmap_async:", time.time() - t, "\n")

    print('end')
    """
    Output:

    apply_async ------------------------------------------------------------
    0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 
    apply_async: 2.135740041732788 


    apply ------------------------------------------------------------
    0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 
    apply: 10.030931949615479 


    map ------------------------------------------------------------
    0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 16 17 18 19 15 
    map: 2.0116937160491943 


    map_async ------------------------------------------------------------
    0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 
    map_async: 2.016545534133911 

    end
    """

results = []

def collect_result(result):    # 只能带一个参数
    results.append(result)

def authors_code():
    """
    带返回值的多进程
    """
    global results
    COUNT = 20
    print("\napply", "--" * 20)
    t = time.time()
    results = []
    with multiprocessing.Pool(5) as pool:
        for i in range(COUNT, 0, -1):
            results.append(pool.apply(func, (i,)))  # 不是多进程
    print("\n{}".format(results))
    print("apply:", time.time() - t, "\n")

    print("\napply_async", "--" * 20)
    t = time.time()
    results = []
    with multiprocessing.Pool(5) as pool:
        for i in range(COUNT, 0, -1):
            pool.apply_async(func, (i,), callback=collect_result)  # 多进程, callback
        pool.close()
        pool.join()
    print("\n{}".format(results))
    print("apply_async:", time.time() - t, "\n")

    print("\nmap", "--" * 20)
    t = time.time()
    with multiprocessing.Pool(5) as pool:
        results = pool.map(func, range(COUNT, 0, -1))
        pool.close()
    print("\n{}".format(results))
    print("map:", time.time() - t, "\n")
    """
    20 19 17 16 18 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1   # NOTE: print的结果是无序的
    [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1  # NOTE: 但返回的结果是有序的]
    """

    print("\nmap_async", "--" * 20)
    t = time.time()
    results = []
    with multiprocessing.Pool(5) as pool:
        pool.map_async(func, range(COUNT, 0, -1), callback=collect_result)  # 多进程
        pool.close()
        pool.join()
    print("\n{}".format(results))
    print("map_async:", time.time() - t, "\n")

    print('end')
    """
    Output:
    
    apply ----------------------------------------
    20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 
    [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    apply: 20.073657989501953 


    apply_async ----------------------------------------
    20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1    # 有序 
    [19, 17, 20, 18, 16, 15, 14, 13, 11, 12, 9, 10, 7, 8, 6, 3, 5, 4, 2, 1]    # 无序
    apply_async: 4.017202854156494


    map ----------------------------------------
    20 19 17 16 18 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1    # 无序
    [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]    # 有序 
    map: 4.1325297355651855 


    map_async ----------------------------------------
    20 18 17 16 19 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1    # 无序
    [[20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]]    # 有序 
    map_async: 4.0203177928924568

    end
    """

if __name__ == "__main__":
    # main()
    # authors_code()
    multi_args()

