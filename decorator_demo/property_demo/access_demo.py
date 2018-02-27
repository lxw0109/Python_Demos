#!/usr/bin/env python3
# coding: utf-8
# File: access_demo.py
# Author: lxw
# Date: 1/31/18 2:03 PM

from property_demo import ACCESS

if __name__ == '__main__':
    access = ACCESS("lxw")
    # print(access.__PRIVATE_MEMBER)    # AttributeError: 'ACCESS' object has no attribute '__PRIVATE_MEMBER'
    print(access._PROTECTED_MEMBER)    # OK
    print(access.PUBLIC_MEMBER)    # OK
