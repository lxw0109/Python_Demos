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
    print(cities)
    print("--"*20)
    print(cities.index)
    print("--"*20)
    city_names = city_names.reindex([2, 0, 1])
    print(city_names)
    print("--"*20)
    print(cities)
    print("--"*20)
    print(cities.index)
    print("--"*20)

    print(np.random.permutation(cities.index))
    print("--"*20)
    print(cities.reindex(np.random.permutation(cities.index)))
    print("--"*20)
    print(cities)


def ten_minutes_to_pandas():
    # 1. 创建对象
    dates = pd.date_range("20170224", periods=6)
    df = pd.DataFrame(data=np.random.randn(6, 4), index=dates, columns=list("ABCD"))
    # np.random.randn(d0, d1, ...): Return a sample (or samples) from the “standard normal” distribution
    # print(dates, "\n", "--" * 20)
    print(df, "\n", "--" * 20)

    """
    df = pd.DataFrame({"A": 1.0,
                       "B": pd.Timestamp("20180327"),
                       # "C": pd.Series(1, index=list(range(4)), dtype="float32"),
                       # "C": pd.Series(1, index=list("ABCD"), dtype="float32"),
                       "C": pd.Series(1, index=pd.date_range("20180226", periods=4), dtype="float32"),
                       "D": np.array([3] * 4, dtype="int32"),
                       "E": pd.Categorical(["test", "train", "test", "train"]),
                       "F": "foo"})
    """
    # 2. 查看数据
    """
    print(df, "\n", "--" * 40)
    print(df.dtypes, "\n", "--" * 40)
    print(df.index, "\n", "--" * 40)
    print(df.columns, "\n", "--" * 40)
    print(df.values, "\n", "--" * 40)
    print(df.describe(), "\n", "--" * 40)
    print(df.T, "\n", "--" * 40)    # 转置
    print(df.sort_index(axis=0, ascending=False), "\n", "--" * 40)    # 按轴排序
    # print(df.sort(columns="B"))    # 按值排序【不好使】

    df = pd.DataFrame({"A": pd.Series([1, 2, 3, 4], index=list("ABCD"))})
    print(df.describe())
    """

    # 3. 选择器
    # 标准Python/Numpy表达式可以完成这些互动工作, 但在生产代码中, 我们推荐使用优化的pandas数据访问方法:
    # .at(label-location based), .iat(integer-location based), .loc(label-location based), .iloc(integer-location based)
    """
    print(df.A, "\n", "--" * 20)
    print(df["A"], "\n", "--" * 20)
    print(df[0:3], "\n", "--" * 20)
    print(df["20180226":"20180228"], "\n", "--" * 20)
    """

    """
    # DataFrame.loc: Purely label-location based indexer for selection by label.
    print(df.loc[:, ["A", "E"]], "\n", "--" * 20)
    print(df.loc["20180226":"20180228", "B":"C"], "\n", "--" * 20)
    print(df.loc["20180226", "B"], "\n", "--" * 20)
    # print(df.at[dates[0], "B"], "\n", "--" * 20)    # NO

    # DataFrame.iloc: Purely integer-location based indexing for selection by position.
    # print(df.iloc(3), "\n", "--" * 20)    # <class "pandas.core.indexing._iLocIndexer">
    print(df.iloc[3], "\n", "--" * 20)    # <class "pandas.core.series.Series">
    print(df.iloc[2:5, 0:2], "\n", "--" * 20)
    print(df.iloc[[2, 3], [0, 1]], "\n", "--" * 20)
    print(df.iloc[0, 0], "\n", "--" * 20)
    print(df.iat[0, 0], "\n", "--" * 20)
    """

    # 4. 布尔索引
    """
    print(df, "\n", "--" * 20)
    print(df[df.A > 0], "\n", "--" * 20)
    print(df[df > 0], "\n", "--" * 20)
    """

    # 5. 使用 isin() 筛选
    """
    print(df2["E"].isin(["Two", "Four"]))
    print(df2[df2["E"].isin(["Two", "Four"])])
    print(df2[[False, True, False, True, False, False]])    # OK
    print(df2.iloc[[1, 3]])    # OK
    print(df2.iloc[:, 3])    # 第三列 <class "pandas.core.series.Series">
    print(df2.iloc[3])    # 第三行 <class "pandas.core.series.Series">
    """

    # 6. 赋值
    """
    print(df, "\n", "--" * 20)
    df.at[dates[0], "A"] = 100
    df.at[dates[0], "B"] = -100
    print(df, "\n", "--" * 20)
    df.iat[0, 0] = 110
    print(df, "\n", "--" * 20)
    df.loc[:, "D"] = [-6, -5, -4, -3, -2, -1]
    print(df, "\n", "--" * 20)
    """

    """
    df2 = df.copy()
    # NOTE: 赋值一个新列，通过索引自动对齐数据(如下面的例子，如果没有index参数，那么直接使用pd.Series()赋值是无效的)
    df2["E"] = pd.Series(["One", "Two", "Three", "Four", "Five", "Six"])    # NO
    df2["E"] = pd.Series(["One", "Two", "Three", "Four", "Five", "Six"], index=dates)    # OK
    # df2["E"] = ["One", "Two", "Three", "Four", "Five", "Six"]    # OK
    print(df2, "\n", "--" * 20)
    print(df, "\n", "--" * 20)

    # df2[df2 < 0] = -df2    # NO: TypeError: bad operand type for unary -: "str"
    # df2[df2 < 0] = 111    # NO: TypeError: Cannot do inplace boolean setting on mixed-types with a non np.nan value
    df2[df2 < 0] = np.nan    # OK
    print(df2, "\n", "--" * 20)
    """

    # 7. 丢失的数据(Kaggle中非常常用)
    """
    print(df, "\n", "--" * 20)

    df1 = df.reindex(index=dates[0:3], columns=list(df.columns) + ["E"])
    print(df1, "\n", "--" * 20)
    df1.loc[dates[0]:dates[1], "E"] = 1
    print(df1, "\n", "--" * 20)
    df1.iloc[1, -2] = np.nan
    print(df1, "\n", "--" * 20)

    # 删除任何有丢失数据的行
    print(df1.dropna(how="any"), "\n", "--" * 20)
    print(df1.dropna(), "\n", "--" * 20)

    # 填充丢失数据
    print(df1.fillna(value="lxw"), "\n", "--" * 20)

    print(pd.isnull(df1), "\n", "--" * 20)
    """

    # 8. 运算
    # 8.1 统计：计算时一般不包括丢失的数据
    """
    df1 = df.reindex(index=dates[0:3], columns=list(df.columns) + ["E"])
    print(df1, "\n", "--" * 20)
    df1.loc[dates[0]:dates[1], "E"] = 1
    print(df1, "\n", "--" * 20)
    print(df1.mean(), "\n", "--" * 20)
    # print(df1.mean(0), "\n", "--" * 20)    # default
    print(df1.mean(1), "\n", "--" * 20)
    """

    """
    s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates)
    print(s, "\n", "--" * 20)
    s = s.shift(2)
    print(s, "\n", "--" * 20)
    print(df1.sub(s, axis="index"), "\n", "--" * 20)
    """

    # 8.2 Apply: 在数据上使用函数
    """
    df.apply(np.cumsum)    # NOT in-place
    print(df.apply(np.cumsum), "\n", "--" * 20)
    print(df.apply(np.cumsum, axis=0), "\n", "--" * 20)
    print(df.apply(np.cumsum, axis=1), "\n", "--" * 20)

    df.apply(lambda x: x.max() - x.min())    # NOT in-place
    print(df.apply(lambda x: x.max() - x.min()), "\n", "--" * 20)
    """

    # 8.3. 合并
    # df = pd.DataFrame(np.random.randn(10, 4))
    # print(df, "\n", "--" * 20)
    # pieces = [df[:2], df[2:3], df[3:]]
    # print(pd.concat(pieces), "\n", "--" * 20)

    # 8.4 连接
    """
    left = pd.DataFrame({"key": ["foo", "foo"], "lval": [1, 2]})
    right = pd.DataFrame({"key": ["foo", "foo"], "rval": [4, 5]})
    print(pd.merge(left, right, on="key"), "\n", "--" * 20)
    """

    # 8.5 添加: 添加行到数据中
    """
    df = pd.DataFrame(np.random.randn(8, 4), columns=list("ABCD"))
    s = df.iloc[3]
    df.append(s, ignore_index=True)    # NOT in-place
    print(df.append(s, ignore_index=True), "\n", "--" * 20)
    """

    # 8.6 分组
    """
    df = pd.DataFrame({"A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
                       "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
                       "C": np.random.randn(8),
                       "D": np.random.randn(8)})
    print(df, "\n", "--" * 20)
    print(df.groupby("A").sum(), "\n", "--" * 20)
    print(df.groupby(["A", "B"]).sum(), "\n", "--" * 20)
    """

    # 8.7 分类: 转换原始类别为分类数据类型
    """
    df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "raw_grade": ["a", "b", "b", "a", "a", "e"]})
    df["grade"] = df["raw_grade"].astype("category")
    print(df, "\n", "--" * 20)
    print(df["raw_grade"], "\n", "--" * 20)
    print(df["grade"], "\n", "--" * 20)
    # 重命令分类为更有意义的名称 (分配到Series.cat.categories对应位置)
    # df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])
    # print(df["grade"], "\n", "--" * 20)
    # print(df, "\n", "--" * 20)
    # print(df.sort("grade"), "\n", "--" * 20)    # NO
    print(df.groupby("grade").size(), "\n", "--" * 20)
    """

    # 9. 直方图和离散化
    """
    s = pd.Series(np.random.randint(0, 8, size=10))    # andint(low[, high, size, dtype]) 	Return random integers from low (inclusive) to high (exclusive).
    print(s, "\n", "--" * 20)
    print(s.value_counts())
    """

    # 10. 字符串
    """
    s = pd.Series(["A", "B", "C", "F", np.nan, "dCabA"])
    print(s.str.lower(), "\n", "--" * 20)
    """


    # 11. 画图
    """
    ts = pd.Series(np.random.randn(800), index=pd.date_range("1/1/2000", periods=800))
    ts = ts.cumsum()
    print(ts, "\n", "--" * 20)
    # ts.plot()
    # plt.show()

    df = pd.DataFrame(np.random.randn(800, 4), index=ts.index, columns=list("ABCD"))
    df = df.cumsum()
    print(df, "\n", "--" * 20)

    plt.figure()
    df.plot()
    plt.legend(loc="best")
    plt.show()
    """

    # 12. 写入文件
    """
    # DataFrame.to_hdf(path_or_buf, key, **kwargs):
    # path_or_buf: the path(string) or HDFStore object. key : <string> identifier for the group in the store
    df.to_hdf("foo.h5", "df_data")    # Write the contained data to an HDF5 file using HDFStore.
    # print(pd.read_hdf("foo.h5", "dflxw"))    # KeyError: 'No object named dflxw in the file'
    print(pd.read_hdf("foo.h5", "df_data"))    # KeyError: 'No object named dflxw in the file'
    """

    df.to_excel("foo.xlsx", sheet_name="Sheet1")
    # df.to_excel("foo.xlsx", sheet_name="Sheet2")    # 同一个文件的不同sheet不会追加，会把整个excel文件覆盖
    print(pd.read_excel("foo.xlsx", sheet_name="Sheet1", index_col=None, na_values=["NA"]))




if __name__ == "__main__":
    # basic_demo()
    # pandas_file()
    # pandas_index()
    ten_minutes_to_pandas()
