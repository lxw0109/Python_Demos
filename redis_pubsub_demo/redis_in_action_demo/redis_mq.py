#!/usr/bin/env python3
# coding: utf-8
# File: redis_mq.py
# Author: lxw
# Date: 9/29/17 1:47 PM

import json
import pprint
import redis
import time
import uuid


def acquire_lock(conn, lockname, acquire_timeout=10):
    uuid_str = str(uuid.uuid4())    # uuid_str: A 128-bit random uuid_str

    end = time.time() + acquire_timeout
    while time.time() < end:
        ret_val = conn.setnx("lock:" + lockname, uuid_str)    # Get the lock
        if ret_val:
            return uuid_str
        time.sleep(0.001)

    return False


def release_lock(conn, lockname, uuid_str):
    pipe = conn.pipeline(True)
    lockname = 'lock:' + lockname

    while 1:
        try:
            # NOTE: WATCH, MULTI, EXEC组成事务，来消除竞争条件
            pipe.watch(lockname)    # Check and verify that we still have the lock
            if pipe.get(lockname) == uuid_str:    # python3: if pipe.get(lockname).decode("utf-8") == uuid_str:
                pipe.multi()
                pipe.delete(lockname)    # Release the lock
                pipe.execute()
                return True
            pipe.unwatch()
            break
        except redis.exceptions.WatchError:    # Someone else did something with the lock, retry
            pass

    return False


def create_chat(conn, sender, recipients, message, chat_id=None):
    chat_id = chat_id or str(conn.incr("ids:chat:"))    # redis中最开始没有"ids:chat:"，可以直接用incr()

    recipients.append(sender)
    recipients_dic = dict((r, 0) for r in recipients)

    pipeline = conn.pipeline(True)
    pipeline.zadd("chat:" + chat_id, **recipients_dic)
    for rec in recipients:
        pipeline.zadd("seen:" + rec, chat_id, 0)
    pipeline.execute()

    return send_message(conn, chat_id, sender, message)


def send_message(conn, chat_id, sender, message):
    uuid_str = acquire_lock(conn, "chat:" + chat_id)
    if not uuid_str:
        raise Exception("Couldn't get the lock")
    try:
        msg_id = conn.incr("ids:" + chat_id)
        ts = time.time()
        msg_json = json.dumps({
            "id": msg_id,
            "ts": ts,
            "sender": sender,
            "message": message,
        })

        conn.zadd("msgs:" + chat_id, msg_json, msg_id)
    finally:
        release_lock(conn, "chat:" + chat_id, uuid_str)
    return chat_id


def fetch_pending_messages(conn, recipient):
    seen = conn.zrange("seen:" + recipient, 0, -1, withscores=True)

    pipeline = conn.pipeline(True)

    for chat_id, seen_id in seen:
        pipeline.zrangebyscore("msgs:" + chat_id, seen_id+1, "inf")    # 获取所有未读消息
    chat_info = zip(seen, pipeline.execute())
    # chat_info: <type 'list'>: [(('1', 0.0), [ '{"message": "message 1", "sender": "joe", "id": 1, "ts": 1506737486.009723}',
    #                                           '{"message": "message 2", "sender": "joe", "id": 2, "ts": 1506737528.189973}',
    #                                           '{"message": "message 3", "sender": "joe", "id": 3, "ts": 1506737567.379356}',
    #                                           '{"message": "message 4", "sender": "joe", "id": 4, "ts": 1506737579.48115}])]

    for i, ((chat_id, seen_id), messages) in enumerate(chat_info):
        if not messages:    # messages: list of dict
            continue
        messages[:] = map(json.loads, messages)
        seen_id = messages[-1]["id"]
        # NOTE: 这里是conn.zadd(), 而不是pipeline.zadd()，是为了立即生效.
        # 因为此语句在循环中，后面的循环中需要读取"chat:"+chat_id中最小的msg_id，以便于更新"msgs:"+chat_id(删除所有人都读过的消息), 所以必须在当前次循环中就要生效(否则后面读到的都是旧的数据，导致"msgs:"+chat_id没有更新)
        conn.zadd("chat:" + chat_id, recipient, seen_id)    # 如果已经存在了，会更新其值

        # zrange() & zrangebyscore() 都是按照score进行排序(不是按照value进行排序的)，不同的是zrange()的返回结果按照索引的区间进行筛选， zrangebyscore()的返回结果按照score的区间进行筛选
        min_id = conn.zrange("chat:" + chat_id, 0, 0, withscores=True)    # min_id: <type 'list'>: [('jenny', 0.0)]

        pipeline.zadd("seen:" + recipient, chat_id, seen_id)

        if min_id:
            pipeline.zremrangebyscore("msgs:" + chat_id, 0, min_id[0][1])
        chat_info[i] = (chat_id, messages)
    pipeline.execute()

    return chat_info


def join_chat(conn, chat_id, user):
    message_id = int(conn.get("ids:" + chat_id))

    pipeline = conn.pipeline(True)
    pipeline.zadd("chat:" + chat_id, user, message_id)
    pipeline.zadd("seen:" + user, chat_id, message_id)
    pipeline.execute()


def leave_chat(conn, chat_id, user):
    pipeline = conn.pipeline(True)
    pipeline.zrem("chat:" + chat_id, user)
    pipeline.zrem("seen:" + user, chat_id)
    pipeline.zcard("chat:" + chat_id)

    if not pipeline.execute()[-1]:
        pipeline.delete("msgs:" + chat_id)
        pipeline.delete("ids:" + chat_id)
        pipeline.execute()
    else:
        oldest = conn.zrange("chat:" + chat_id, 0, 0, withscores=True)
        conn.zremrangebyscore("msgs:" + chat_id, 0, oldest[0][1])


if __name__ == "__main__":
    pool = redis.ConnectionPool(host="192.168.1.41", port=6379, db=0)
    conn = redis.Redis(connection_pool=pool)

    # TYPE: "ids:chat:": string, "msgs:1": zset, "ids:1": string, "seen:joe": zset, "seen:jeff": zset, "seen:jenny": zset)
    conn.delete("ids:chat:", "msgs:1", "ids:1", "seen:joe", "seen:jeff", "seen:jenny")

    print("\nCreating a new chat session with some recipients.")
    chat_id = create_chat(conn, "joe", ["jeff", "jenny"], "message content 0")    # create_chat(conn, sender, recipients, message, chat_id=None)

    print("\nSending a few messages.")
    for i in xrange(2, 5):
        send_message(conn, chat_id, "joe", "message content %s" % i)    # send_message(conn, chat_id, sender, message)
    print("\nGetting messages that are waiting for jeff and jenny.")
    r1 = fetch_pending_messages(conn, "jeff")
    r2 = fetch_pending_messages(conn, "jenny")
    print("r1 == r2?{}".format(r1 == r2))    # True
    print("Those messages are:")
    pprint.pprint(r1)
    conn.delete("ids:chat:", "msgs:1", "ids:1", "seen:joe", "seen:jeff", "seen:jenny")

