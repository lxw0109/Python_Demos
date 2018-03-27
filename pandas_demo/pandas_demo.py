#!/usr/bin/env python3
# coding: utf-8
# File: pandas_demo.py
# Author: lxw
# Date: 9/18/17 11:27 PM
"""
# Reference:
1. [intro_to_pandas.ipynb](https://colab.research.google.com/notebooks/mlcc/intro_to_pandas.ipynb#scrollTo=daQreKXIUslr)
2. [十分钟搞定pandas](http://python.jobbole.com/84416/) /
[10 Minutes to pandas](http://pandas.pydata.org/pandas-docs/stable/10min.html)
"""

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


def basic_demo():
    # Series是单一**列**。DataFrame 中包含一个或多个 Series，每个 Series 均有一个名称
    # <class "pandas.core.series.Series">
    city_names = pd.Series(["San Francisco", "San Jose", "Sacramento"])
    population = pd.Series([852469, 1015785, 485199])
    # print(population / 1000)    # 可以向 Series 应用 Python 的基本运算指令
    # print(np.log(population))
    # print(population.apply(np.log))    # 像Python映射函数一样，Series.apply以参数形式接受lambda函数，并应用于每个值
    # print(population.apply(lambda x:x > 1000000))    # 像Python映射函数一样，Series.apply以参数形式接受lambda函数，并应用于每个值

    # DataFrame，可以将它理解成一个关系型数据表格，其中包含多个行和已命名的列
    # <class "pandas.core.frame.DataFrame">
    cities = pd.DataFrame({"City name": city_names, "Population": population})
    # print(cities["City name"])    # <class "pandas.core.series.Series">
    # print(cities["City name"][1])    # <class "str">
    # print(cities["City name"][0:2])    # <class "pandas.core.series.Series">
    cities["Area square miles"] = pd.Series([46.87, 176.53, 97.92])
    cities["Population density"] = cities["Population"] / cities["Area square miles"]
    # print(cities)

    # 通过添加一个新的布尔值列(当且仅当以下两项均为 True 时为 True): 1) 城市以圣人命名  2) 城市面积大于 50 平方英里
    # YES:
    # area_gt_50 = cities["Area square miles"].apply(lambda x: x > 50)    # OK
    area_gt_50 = cities["Area square miles"] > 50    # OK
    name_startswith_San = cities["City name"].apply(lambda x: x.startswith("San"))    # OK
    # 布尔值Series是使用"按位"而非传统布尔值"运算符"组合的。例如，执行逻辑与时，应使用 &，而不是 and。
    cities["City name and area"] = area_gt_50 & name_startswith_San
    print(cities)
    # NO:
    # cities["City name and area"] = cities.apply(lambda city: city["Area square miles"] > 50 and city["City name"].startswith("San"))
    # NOTE: item 是每一列的**名字**，而不是每一列的数据，也不是每一行的数据
    # for item in cities:
    #     print(item)


def pandas_file():
    california_housing_df = pd.read_csv("./california_housing_train.csv", sep=",")
    # print(california_housing_df.describe())    # 显示关于 DataFrame 的统计信息(count/mean/std/min/max/...)
    # print(california_housing_df.head())    # 显示 DataFrame 的前几个记录
    housing_median_age = california_housing_df.hist("housing_median_age")    # 通过DataFrame.hist可以大体了解数据的分布
    plt.show(housing_median_age)


def pandas_index():
    city_names = pd.Series(["San Francisco", "San Jose", "Sacramento"])
    population = pd.Series([852469, 1015785, 485199])
    cities = pd.DataFrame({"City name": city_names, "Population": population})
    # print(cities)
    # print()
    # print(cities.index)
    # print()
    city_names = city_names.reindex([2, 0, 1])
    # print(city_names)
    # print()
    # print(cities)
    # print()
    # print(cities.index)
    # print()

    print(np.random.permutation(cities.index))
    print(cities.reindex(np.random.permutation(cities.index)))
    print(cities)


def ten_minutes_to_pandas():
    # 创建对象
    dates = pd.date_range("20170130", periods=6)
    # np.random.randn(d0, d1, ...): Return a sample (or samples) from the “standard normal” distribution
    df = pd.DataFrame(data=np.random.randn(6, 4), index=dates, columns=list("ABCD"))
    # print(df)

    df = pd.DataFrame({"A": 1.,
                       "B": pd.Timestamp("20180316"),
                       # "C": pd.Series(1, index=list(range(4)), dtype="float32"),
                       # "C": pd.Series(1, index=list("ABCD"), dtype="float32"),
                       "C": pd.Series(1, index=pd.date_range("19900130", periods=4), dtype="float32"),
                       "D": np.array([3] * 4, dtype="int32"),
                       "E": pd.Categorical(["test", "train", "test", "train"]),
                       "F": "foo"})
    # 查看数据
    """
    print(df, "\n")
    print(df.dtypes, "\n")
    print(df.index, "\n")
    print(df.columns, "\n")
    print(df.values, "\n")
    print(df.describe(), "\n")
    print(df.T, "\n")    # 转置
    print(df.sort_index(axis=1, ascending=False), "\n")    # 按轴排序
    # print(df.sort(columns="B"))    # 按值排序【不好使】
    """

    # 选择器
    """
    标准Python / Numpy表达式可以完成这些互动工作, 但在生产代码中, 我们推荐使用优化的pandas数据访问方法:
    .at, .iat, .loc, .iloc, .ix
    """
    """
    print(df.A, "\n")
    print(df["A"], "\n")
    print(df[0:3], "\n")
    print(df["19900131":"19900202"], "\n")
    """




if __name__ == "__main__":
    # basic_demo()

    # pandas_file()

    # pandas_index()

    ten_minutes_to_pandas()
