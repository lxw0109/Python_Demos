#!/usr/bin/env python3
# coding: utf-8
# File: argparse_demo.py
# Author: lxw
# Date: 3/1/18 2:29 PM
"""
Reference:
[argparse](http://wiki.jikexueyuan.com/project/explore-python/Standard-Modules/argparse.html)
"""

import argparse


def echo_demo():
    # positional arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("integer", type=int, help="display an integer")
    args = parser.parse_args()
    print(args.integer)


def square_demo():
    # positional arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("square", type=int, help="display a square of a given integer")
    args = parser.parse_args()
    print(args.square ** 2)


def optinal_args_demo():
    # optional arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--square", type=int, help="display a square of a given integer")
    parser.add_argument("--cubic", type=int, help="display a cubic of a given integer")
    args = parser.parse_args()

    if args.square:
        print(args.square ** 2)
    if args.cubic:
        print(args.cubic ** 3)


def mixture_demo():
    # mixture of positional arguments and optional arguments
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("integers", metavar="N", type=int, nargs="+",
        help="an integer for accumulator")
    parser.add_argument("--sum", dest="accumulator", action="store_const",
        const=sum, default=max, help="sum integers(default: find the max)")
    args = parser.parse_args()
    print(args.accumulator(args.integers))


if __name__ == '__main__':
    # 1. echo the argument(positional arguments)
    # echo_demo()

    # 2. square the argument(positional arguments)
    # square_demo()

    # 3. optional arguments
    optinal_args_demo()

    # 4. mixture of optional arguments
    # mixture_demo()

