#!/usr/bin/env python3
# coding: utf-8
# File: read.py
# Author: lxw
# Date: 11/27/17 11:03 AM

from base import *

def read():
   print('reference data id: ' + str(id(demo)))
   print('reference data value : ' + demo.name)
   print('primitive data id: ' + str(id(foo)))
   print('primitive data value: ' + str(foo))