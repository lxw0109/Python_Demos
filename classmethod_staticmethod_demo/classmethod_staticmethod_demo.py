#!/usr/bin/env python3
# coding: utf-8
# File: classmethod_staticmethod_demo.py
# Author: lxw
# Date: 4/19/18 5:34 PM


def foo(x):
    print("executing foo({})".format(x))


class A:
    def foo(self, x):
        print("executing foo({}, {})".format(self, x))

    @classmethod
    def class_foo(cls, x):
        print("executing class_foo({}, {})".format(cls, x))
        A().foo(x)
        cls().foo(x)

    @staticmethod
    def static_foo(x):
        print("executing static_foo({})".format(x))
        A().foo(x)


if __name__ == "__main__":
    foo(10)
    print("==" * 30)

    a = A()
    a.foo(10)
    print("==" * 30)

    A.class_foo(10)
    a.class_foo(10)
    print("==" * 30)

    A.static_foo(10)
    a.static_foo(10)
    print("==" * 30)
