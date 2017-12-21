#!/usr/bin/env python3
# coding: utf-8
# File: demo.py
# Author: lxw
# Date: 11/27/17 11:43 AM

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
print(sys.path)
from import_demo.demo1.demo11.demo111.module import my_func    # NOTE: 包导入路径中最左边的部分必须是sys.path路径列表中的一个目录。

my_func()