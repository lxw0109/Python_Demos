#!/usr/bin/env python3
# coding: utf-8
# File: consumer.py
# Author: lxw
# Date: 11/14/17 3:55 PM

"""
Consumer: Subscriber/Receiver
"""

import stomp
import time


class MyListener(stomp.ConnectionListener):
    def on_error(self, headers, message):
        print("Receive an error \"{}\".".format(message))

    def on_message(self, headers, message):
        print("Receive a message \"{}\".".format(message))


def consume():
    HOST = "127.0.0.1"
    PORT = 61613
    QUEUE_NAME = "/queue/lxw_activemq"

    conn = stomp.Connection([(HOST, PORT)])
    conn.set_listener("", MyListener())
    conn.start()    # essential
    conn.connect(wait=True)    # essential
    conn.subscribe(destination=QUEUE_NAME, id=1, ack="auto")
    time.sleep(3)
    # conn.disconnect()    # 如果要一直处于监听状态， 就不能disconnect()


if __name__ == '__main__':
    consume()
    while 1:
        time.sleep(100)
