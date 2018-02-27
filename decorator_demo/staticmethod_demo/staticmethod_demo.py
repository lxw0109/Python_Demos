#!/usr/bin/env python3
# coding: utf-8
# File: staticmethod_demo.py
# Author: lxw
# Date: 1/31/18 2:25 PM
import time

class DateObj:
    def __init__(self,year,month,day):
        # print(self)
        self.year=year
        self.month=month
        self.day=day

    def __str__(self):
        return "bixiang"

    @staticmethod
    def now(): #用Date.now()的形式去产生实例,该实例用的是当前时间
        t=time.localtime() #获取结构化的时间格式
        return DateObj(t.tm_year,t.tm_mon,t.tm_mday) #新建实例并且返回

    @staticmethod
    def tomorrow():#用Date.tomorrow()的形式去产生实例,该实例用的是明天的时间
        t=time.localtime(time.time()+86400)
        return DateObj(t.tm_year,t.tm_mon,t.tm_mday)

if __name__ == '__main__':
    # d = DateObj(1, 2, 3)
    DateObj.now()
