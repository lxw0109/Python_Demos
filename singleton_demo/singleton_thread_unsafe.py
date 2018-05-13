#!/usr/bin/env python3
# coding: utf-8
# File: singleton_demo.py
# Author: lxw
# Date: 4/20/18 3:20 PM

"""
下面的实现存在以下问题:
1. 类方法get_instance()用于获取单例，但是类本身也可以实例化，这样的方式其实并不符合单例模式的要求
在C#或Java的设计模式中, 通常通过私有化类的构造函数来杀死类本身的繁殖能力, 然而python并没有访问限定强制约束.
2. 这个单例类并不是线程安全的(non-thread-safe)
"""

import threading


class Singleton:
    # 定义在所有方法外面的变量是"静态成员变量"
    __singleton = None

    value = 10
    _value = 10
    __value = 10

    def __init__(self):
        # self.lxw = 100    # 不是静态成员变量
        pass

    @staticmethod
    def get_instance():
        if not Singleton.__singleton:
            Singleton.__singleton = Singleton()

        return Singleton.__singleton


id_set = set()


def _tst_thread_safe():
    global id_set
    id_set.add(id(Singleton.get_instance()))


if __name__ == "__main__":
    # 缺点1: 类本身仍可以实例化
    """
    s1 = Singleton()
    print(id(s1))

    s2 = Singleton()
    print(id(s2))

    s3 = Singleton.get_instance()
    print(id(s3))
    s4 = Singleton.get_instance()
    print(id(s4))

    print(id(Singleton.value))    # 10
    print(id(Singleton._value))    # 10
    # print(id(Singleton.__value))    # AttributeError: type object 'Singleton' has no attribute '__value'
    print(id(Singleton._Singleton__value))    # 10    # NOTE
    """

    # 缺点2: non-thread-safe
    count = 0
    while 1:
        threading.Thread(target=_tst_thread_safe, args=[]).start()
        count += 1
        if count > 1000000000000:
            break

    print(id_set)
