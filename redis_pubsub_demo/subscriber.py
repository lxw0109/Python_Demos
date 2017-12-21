#!/usr/bin/env python3
# coding: utf-8
# File: redis_pubsub_demo.py
# Author: lxw
# Date: 9/28/17 2:58 PM
"""
References:
[使用python来搞定redis的订阅功能](http://www.cnblogs.com/anpengapple/p/7027979.html)
[使用redis快速搭建发布-订阅系统(python3x)](http://blog.csdn.net/drdairen/article/details/51659061)
[A short script exploring Redis pubsub functions in Python](https://gist.github.com/lxw0109/ee6b578a454f5d17e0d1adc3219b8a21)

NOTE: 先运行订阅者再运行发布者: 如果一个消息没有订阅者，那它会从redis服务器中消失（所以要先启动订阅者，再启动发布者）
"""

import redis
import threading

# 要注意检查type，一旦listen就会收到一个消息，但不是发布者的消息，而是系统发来的，内容为{'pattern': None, 'type': 'subscribe', 'channel': 'spub', 'data': 1L}，表示:订阅成功，频道是spub，当前有一个订阅用户。

class RedisSubscriber(threading.Thread):    # Listener
    """
    Redis频道订阅辅助类
    """
    def __init__(self, redis_conn, channels):
        threading.Thread.__init__(self)
        self.redis = redis_conn
        self.pubsub = self.redis.pubsub()
        self.pubsub.psubscribe(channels)    # 同时订阅多个频道，要用psubscribe

    def work(self, item):
        print("Channel:{0}, data:{1}".format(item["channel"], item["data"]))

    def run(self):
        for item in self.pubsub.listen():
            if item["data"] == "KILL":
                self.pubsub.unsubscribe()
                print("{0}, unsubscribed and finished".format(self))
                break
            else:
                self.work(item)


def main():
    pool = redis.ConnectionPool(host="192.168.1.41", port=6379, db=0)
    redis_conn = redis.Redis(connection_pool=pool)
    client = RedisSubscriber(redis_conn, ["channel1", "channel2"])
    client.start()

    redis_conn.publish("channel1", "lxw: this will reach the subscriber.")
    redis_conn.publish("channel123", "lxw: this will NOT reach the subscriber.")
    redis_conn.publish("channel2", "lxw: this will reach the subscriber.")
    redis_conn.publish("channel2", "KILL")
    redis_conn.publish("channel1", "KILL")


if __name__ == "__main__":
    main()