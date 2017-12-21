#!/usr/bin/env python3
# coding: utf-8
# File: publisher.py
# Author: lxw
# Date: 9/28/17 3:05 PM
# Reference: http://www.cnblogs.com/anpengapple/p/7027979.html

import redis

class RedisSubscriber(object):
    """
    Redis频道订阅辅助类
    """
    def __init__(self, channel):
        pool = redis.ConnectionPool(host="192.168.1.41", port=6379, db=0)
        self.conn = redis.Redis(connection_pool=pool)
        self.channel = channel  # 定义频道名称

    def psubscribe(self):
        """
        订阅方法
        """
        subscriber = self.conn.pubsub()
        subscriber.psubscribe(self.channel)  # 同时订阅多个频道，要用psubscribe
        subscriber.listen()
        return subscriber

def main():
    subscriber = RedisSubscriber(["channel1", "channel2"])
    redis_subscriber = subscriber.psubscribe()
    while 1:
        msg = redis_subscriber.parse_response(block=False, timeout=6)
        print("Message:{}".format(msg))

if __name__ == "__main__":
    main()