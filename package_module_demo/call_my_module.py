#!/usr/bin/env python3
# coding: utf-8
# File: call_my_module.py
# Author: lxw
# Date: 10/27/17 2:21 PM


if __name__ == "__main__":
    import my_module
    print("[call_my_module.py]Running...")
    print(my_module.func())    # count: 2. 2
    my_class = my_module.MyClass()
    my_class.my_func()    # count: 2
    print(my_module.func())    # count: 3.   3
    import my_module    # NOTE: import一个模块多次，并不会重新初始化该模块中的数据. "Being imported"也不会重复打印出来.
    print(my_module.func())    # count: 4.  4. NOTE here.
    import my_module
    print(my_module.func())    # count: 5.  5.
    import my_module
    print(my_module.func())    # count: 6.  6.
else:
    print("[call_my_module.py]Imported as a module...")
