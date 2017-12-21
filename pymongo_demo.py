#!/usr/bin/env python3
# coding: utf-8
# File: pymongo_demo.py
# Author: lxw
# Date: 12/6/17 2:59 PM

import pymongo
import urllib


def main():
    username = urllib.parse.quote_plus("ic_ro")
    password = urllib.parse.quote_plus("change_this")
    MONGO_HOST = "101.201.102.233"
    conn = pymongo.MongoClient('mongodb://{}:{}@{}:27017'.format(username, password, MONGO_HOST))
    db = conn["topic_askbot"]    # 自建社区的数据
    col = db["data"]    # 话题数据

    item = col.find_one()
    print(item)

if __name__ == '__main__':
    main()