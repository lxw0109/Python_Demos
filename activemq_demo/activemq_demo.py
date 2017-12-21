#!/usr/bin/env python3
# coding: utf-8
# File: activemq_demo.py
# Author: lxw
# Date: 9/29/17 4:15 PM

import time
import sys

import stomp


class MyListener(stomp.ConnectionListener):
    def on_error(self, headers, message):
        print("Receive an error \"{}\".".format(message))

    def on_message(self, headers, message):
        print("Receive a message \"{}\".".format(message))


def main():
    # [Simple Example](https://github.com/jasonrbriggs/stomp.py/wiki/Simple-Example)
    conn = stomp.Connection()
    conn.set_listener("", MyListener())
    conn.start()
    conn.connect("admin", "lxw", wait=True)

    conn.subscribe(destination="/queue/test", id=1, ack="auto")

    conn.send(body=" ".join(sys.argv[1:]), destination="/queue/test")
    time.sleep(2)
    conn.disconnect()

    """
    # 推送到队列queue: debug可以立即执行完毕，但run就无法执行完毕
    conn = stomp.Connection10()
    conn.start()
    conn.connect()
    conn.send("SampleQueue", "Simples Assim")
    conn.disconnect()
    time.sleep(2)
    """


if __name__ == "__main__":
    main()