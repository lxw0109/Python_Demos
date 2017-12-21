#!/usr/bin/env python3
# coding: utf-8
# File: redis_demo.py
# Author: lxw
# Date: 9/30/17 11:04 AM

import redis

if __name__ == "__main__":
    """
    pool = redis.ConnectionPool(host="192.168.1.41", port=6379, db=0)
    conn = redis.Redis(connection_pool=pool)

    conn.zadd("lxw", 1, 10)
    conn.zadd("lxw", 3, 20)
    conn.zadd("lxw", 2, 30)

    print(conn.zrange("lxw", 0, -1, withscores=True))
    print(conn.zrangebyscore("lxw", -100, 100, withscores=True))

    conn.zrem("lxw", 1)
    conn.zrem("lxw", 3)
    conn.zrem("lxw", 2)
    """

    pool = redis.ConnectionPool(host="192.168.1.41", port=6379, db=0)
    redis_uri = redis.Redis(connection_pool=pool)
    REDIS_KEY = "significance_news_public"

    redis_uri.lpush(REDIS_KEY, 1828786)

    # conn_redis

