#!/usr/bin/env python3
# coding: utf-8
# File: producer.py
# Author: lxw
# Date: 11/14/17 3:55 PM

"""
Producer: Publisher
"""

import stomp


if __name__ == '__main__':
    """
    HOST = "127.0.0.1"
    PORT = 61613
    """
    HOST = "39.106.6.50"
    PORT = 61613
    QUEUE_NAME = "/queue/lxw_activemq"

    conn = stomp.Connection([(HOST, PORT)])
    conn.start()    # essential
    conn.connect(wait=True)    # Not essential, but the program will hold if not existing.
    for i in range(2):
        message = "Hello, lxw{}.".format(i)
        conn.send(body=message, destination=QUEUE_NAME)
        print("Publish message: {}".format(message))

    conn.disconnect()
    print("Complete.")
