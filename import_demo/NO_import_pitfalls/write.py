#!/usr/bin/env python3
# coding: utf-8
# File: write.py
# Author: lxw
# Date: 11/27/17 11:03 AM

# import base
from base import demo
from base import foo

def write():
   print("\nOriginal:")
   # print("Original reference data id: " + str(id(base.demo)))
   # base.demo.name = "Updated Demo"    # this will reflect that change
   demo.name = "Updated Demo"    # this will reflect that change
   #base.demo = base.Demo("Updated Demo") # this won't relfect the change
   # print("Original data id: " + str(id(base.foo)))
   # base.foo = 1000
   # global foo
   print("id(foo):{}\nfoo: {}".format(id(foo), foo))
   #foo = 1000
   # print("Original data id after assignment: " + str(id(base.foo)))