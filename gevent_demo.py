#!/usr/bin/env python3
# coding: utf-8
# File: gevent_demo.py
# Author: lxw
# Date: 12/11/17 2:30 PM

import gevent
import time

from gevent import monkey

# monkey.patch_socket()

def f(n):
    for i in range(n):
        print("gevent.getcurrent(): {0}. i: {1}".format(gevent.getcurrent(), i))
        # time.sleep(1)    # NO
        gevent.sleep(1)    # OK


def main():
    g1 = gevent.spawn(f, 5)
    g2 = gevent.spawn(f, 5)
    g3 = gevent.spawn(f, 5)
    g1.join()
    g2.join()
    g3.join()


if __name__ == '__main__':
    main()
