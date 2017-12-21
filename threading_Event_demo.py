#!/usr/bin/env python3
# coding: utf-8
# File: threading_Event_demo.py
# Author: lxw
# Date: 10/10/17 6:01 PM

import threading
import time
import logging

logging.basicConfig(level=logging.DEBUG, format='(%(threadName)-10s) %(message)s')


def worker(event):
    logging.debug('Waiting for redis ready...')
    event.wait()
    logging.debug('redis ready, and connect to redis server and do some work [%s]', time.ctime())
    time.sleep(1)


redis_ready = threading.Event()
t1 = threading.Thread(target=worker, args=(redis_ready,), name='t1')
t1.start()

t2 = threading.Thread(target=worker, args=(redis_ready,), name='t2')
t2.start()

logging.debug('first of all, check redis server, make sure it is OK, and then trigger the redis ready event')
time.sleep(3)  # simulate the check progress
redis_ready.set()

