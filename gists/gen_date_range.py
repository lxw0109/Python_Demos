#!/usr/bin/env python3
# coding: utf-8
# File: gen_date_range.py
# Author: lxw
# Date: 5/8/18 3:59 PM

import datetime
import pandas as pd


if __name__ == "__main__":
    ##################################################
    # 1. 获取两个日期之间的所有日期的列表
    ##################################################
    # print(pd.date_range(start="20180506", end="20180508"))    # OK    # 闭区间
    # print(pd.date_range(start="2018-05-06", end="2018-05-08"))    # OK
    date_range_list = [datetime.datetime.strftime(date, "%Y%m%d")
                       for date in pd.date_range(start="2018-05-06", end="2018-05-08")]
    print(date_range_list)