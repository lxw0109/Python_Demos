#!/usr/bin/env python3
# coding: utf-8
# File: priorityqueue_demo.py
# Author: lxw
# Date: 7/9/17 11:25 AM

from queue import PriorityQueue
import random

def main():
    q = PriorityQueue()
    for i in range(10):
        q.put((random.randint(1, 10), str(i)))  # (priority, data) must be tuple.
    while not q.empty():
        element = q.get()
        print(element)

if __name__ == "__main__":
    main()