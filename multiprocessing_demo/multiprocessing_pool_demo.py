#!/usr/bin/env python3
# coding: utf-8
# File: multiprocessing_pool_demo.py
# Author: lxw
# Date: 4/17/18 2:21 PM

import multiprocessing
import os
import time


def single_param(value):
    print(value, end=" ")


def multiple_params(value1, value2, value3):
    print("in multiple_params().")
    print(value1, end=" ")
    print(value2, end=" ")
    print(value3, end=" ")


def single_multi_params_demo():
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
        for it in [(1, 2, 3), (4, 5, 6)]:
            pool.apply_async(multiple_params, it)
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
    """


def func(num):
    # print("[func()]pid:", os.getpid())
    print(num, end=" ", flush=True)    # 增加flush=True才能体现出多进程和进程池
    # sys.stdout.flush()    # 增加这条语句才能体现出多进程和进程池
    # time.sleep(2)
    return num


results = []


def collect_result(result):    # NOTE: 只能带一个参数
    global results
    # print("[collect_result()]pid:", os.getpid())
    results.append(result)


def apply_map_async_demo():
    """
    带返回值的多进程
    """
    COUNT = 20

    print("\napply", "--" * 20)
    t = time.time()
    local_results = []
    with multiprocessing.Pool(5) as pool:
        for i in range(COUNT, 0, -1):
            # NOTE: pool.apply不支持并发: 但多个进程串行地执行每个任务(仍然是多进程执行)
            local_results.append(pool.apply(func, (i,)))
        pool.close()
    print("\n{}".format(local_results))
    """
    # NOTE: print的结果是无序的 
    20 19 18 17 16 14 13 15 12 11 10 9 7 6 8 5 4 3 2 1
    # NOTE: 返回的结果也是无序的(apply_async不保证返回值的顺序)
    [20, 18, 19, 16, 17, 14, 13, 12, 11, 10, 9, 7, 6, 5, 4, 3, 15, 2, 8, 1]
    """
    print("apply time cost:", time.time() - t, "\n", "**"*30)

    print("\napply_async", "--" * 20)
    t = time.time()
    with multiprocessing.Pool(5) as pool:
        for i in range(COUNT, 0, -1):
            pool.apply_async(func, (i,), callback=collect_result)    # 并发. callback是可选的, 可以不使用该字段
        pool.close()
        pool.join()
    global results
    print("\n{}".format(results))
    """
    # NOTE: print的结果是无序的 
    20 19 18 17 16 14 13 15 12 11 10 9 7 6 8 5 4 3 2 1
    # NOTE: 返回的结果也是无序的(apply_async不保证返回值的顺序)
    [20, 18, 19, 16, 17, 14, 13, 12, 11, 10, 9, 7, 6, 5, 4, 3, 15, 2, 8, 1]
    """
    print("apply_async time cost:", time.time() - t, "\n", "**"*30)

    print("\nmap", "--" * 20)
    t = time.time()
    with multiprocessing.Pool(5) as pool:
        results = pool.map(func, range(COUNT, 0, -1))    # 并发
        pool.close()
    print("\n{}".format(results))
    """
    # NOTE: print的结果是无序的
    20 19 17 16 18 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1
    # NOTE: 但返回的结果是有序的(map能够保证返回值的顺序按照函数调用的顺序)
    [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    """
    print("map:", time.time() - t, "\n", "**"*30)

    print("\nmap_async", "--" * 20)
    t = time.time()
    results = []
    with multiprocessing.Pool(5) as pool:
        pool.map_async(func, range(COUNT, 0, -1), callback=collect_result)    # 并发. callback是可选的, 可以不使用该字段
        pool.close()
        pool.join()
    print("\n{}".format(results))
    """
    # NOTE: print的结果是无序的
    20 19 17 18 16 15 13 14 10 12 9 8 7 6 11 5 4 3 2 1 
    # NOTE: 但返回的结果是有序的(map_async能够保证返回值的顺序按照函数调用的顺序)
    # map和map_async的结果是以list的形式返回的, 所以只会调用collect_result一次，因此结果是list of list
    [[20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]]
    """
    print("map_async:", time.time() - t, "\n", "**"*30)

    print("end")


if __name__ == "__main__":
    # single_multi_params_demo()

    apply_map_async_demo()

