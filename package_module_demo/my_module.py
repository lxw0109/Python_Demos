#!/usr/bin/env python3
# coding: utf-8
# File: package_module_demo.py
# Author: lxw
# Date: 10/27/17 2:14 PM

count = 1
print("Being imported")

def func():
    global count
    count += 1
    print("my_module.py:func(). count: {}".format(count))
    return count


class MyClass:
    def my_func(self):
        global count
        print("mymoduel.py:MyClass:my_func(). count: {}".format(count))


if __name__ == "__main__":
    print("[my_module.py]Running...")
    mc = MyClass()
    func()   # count: 2
    mc.my_func()    # count: 2
else:
    print("[my_module.py]Imported as a module...")
