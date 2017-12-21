#!/usr/bin/env python3
# coding: utf-8
# File: call_my_module.py
# Author: lxw
# Date: 10/27/17 2:21 PM


if __name__ == "__main__":
    import my_module
    print("[call_my_module.py]Running...")
    print(my_module.func())
    my_class = my_module.MyClass()
    my_class.my_func()
    print(my_module.func())
    import my_module
    print(my_module.func())
else:
    print("[call_my_module.py]Imported as a module...")
