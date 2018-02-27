#!/usr/bin/env python3
# coding: utf-8
# File: property_demo.py
# Author: lxw
# Date: 1/31/18 1:39 PM
"""
Reference: python基础----特性（property）、静态方法（staticmethod）、类方法（classmethod）、__str__的用法
为什么要用property?
将一个类的函数定义成特性以后，对象再去使用的时候obj.name,根本无法察觉自己的name是执行了一个函数然后计算出来的，
这种特性的使用方式**遵循了统一访问的原则**
"""

import math

class Circle:
    def __init__(self, radius):
        self.radius = radius

    @property
    def area(self):
        return math.pi * self.radius ** 2

    @property
    def perimeter(self):
        return 2 * math.pi * self.radius


if __name__ != "__main__":
    c = Circle(10)
    print(c.radius)
    # 可以像访问数据属性一样去访问area,会触发一个函数的执行,动态计算出一个值
    print(c.area)
    # print(c.area())    # NOTE 1: TypeError: "float" object is not callable
    print(c.perimeter)

    # NOTE 2: 此时的特性arear和perimeter不能被赋值
    # c.area = 10    # AttributeError: can"t set attribute


class ACCESS:
    def __init__(self, val):
        """
        :param val: <str>.
        """
        self.__PRIVATE_MEMBER = val + "_PRIVATE"
        self._PROTECTED_MEMBER = val + "_PROTECTED"
        self.PUBLIC_MEMBER = val + "_PUBLIC"


if __name__ == "__main__":
    access = ACCESS("lxw")
    # NOTE 3: python把双下划线开头的变量作为private变量
    # print(access.__PRIVATE_MEMBER)    # AttributeError: 'ACCESS' object has no attribute '__PRIVATE_MEMBER'
    print(access._ACCESS__PRIVATE_MEMBER)    # OK
    print(access._PROTECTED_MEMBER)    # OK
    print(access.PUBLIC_MEMBER)    # OK


class Foo:
    def __init__(self, val):
        self.__NAME = val    # 将所有的数据属性都隐藏起来

    @property
    def name(self):
        return self.__NAME    # obj.name访问的是self.__NAME(这也是真实值的存放位置)

    @name.setter
    def name(self, value):
        if not isinstance(value, str):    # 在设定值之前进行类型检查
            raise TypeError("{} must be str".format(value))
        self.__NAME = value    # 通过类型检查后,将值value存放到真实的位置self.__NAME

    @name.deleter
    def name(self):
        raise TypeError("Can not delete")


if __name__ != "__main__":
    f = Foo("egon")
    # NOTE 3: 虽然Python没有在语法上把public, protected, private三种访问权限内建到类机制中,
    # 但可以通过property实现类似的访问权限控制

    exit(0)
    print(f.__NAME)    # AttributeError: 'Foo' object has no attribute '__NAME'
    # print(f.name)    # egon
    # f.name = 10    # 抛出异常"TypeError: 10 must be str"
    f.name = "hello"    # OK
    print(f.name)    # hello
    # del f.name    # 抛出异常"TypeError: Can not delete"