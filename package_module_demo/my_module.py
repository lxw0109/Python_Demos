#!/usr/bin/env python3
# coding: utf-8
# File: package_module_demo.py
# Author: lxw
# Date: 10/27/17 2:14 PM

count = 1

def func():
    print("my_module.py:func()")
    global count
    count += 1
    return count


class MyClass:
    def my_func(self):
        print("mymoduel.py:MyClass:my_func()")


if __name__ == "__main__":
    print("[my_module.py]Running...")
    func()
else:
    print("[my_module.py]Imported as a module...")
