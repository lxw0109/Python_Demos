#!/usr/bin/env python3
# coding: utf-8
# File: iris_classification_demo.py
# Author: lxw
# Date: 5/26/18 5:30 PM
"""
Reference:
[手把手教你使用sklearn快速入门机器学习](https://cloud.tencent.com/developer/article/1091908)
"""


import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn import linear_model


def iris_demo():
    # 1. 加载数据
    iris = datasets.load_iris()
    """
    # 特征名称: ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
    print("feature_names: {0}".format(iris.feature_names))
    # 标签名称: ["setosa" "versicolor" "virginica"]
    print("target_names: {0}".format(iris.target_names))

    print("data shape: {0}".format(iris.data.shape))  # (150, 4)
    print("data top 5:\n {0}".format(iris.data[:5]))
    print("target unique: {0}".format(np.unique(iris.target)))  # [0 1 2]
    print("targe shape: {0}".format(iris.target.shape))  # (150,)
    print("target top 5:\n {0}".format(iris.target[:5]))
    """

    # 2. 数据可视化: 简单分析下不同花萼长度和宽度的情况下，对应的不同的鸢尾花的情况
    sepal_length_ndarray = iris.data[:, 0]  # 花萼长度
    sepal_width_ndarray = iris.data[:, 1]  # 花萼宽度

    # 构建 setosa, versicolor, virginica 索引数组
    setosa_index_ndarray = iris.target == 0  # setosa 索引数组
    # NOTE: setosa_index_ndarray: [True  True  True ... False  False  False]
    versicolor_index_ndarray = iris.target == 1  # versicolor 索引数组
    virginica_index_ndarray = iris.target == 2  # virginica 索引数组

    plt.scatter(sepal_length_ndarray[setosa_index_ndarray], sepal_width_ndarray[setosa_index_ndarray],
                color="red", marker="o", label="setosa")
    plt.scatter(sepal_length_ndarray[versicolor_index_ndarray], sepal_width_ndarray[versicolor_index_ndarray],
                color="blue", marker="x", label="versicolor")
    plt.scatter(sepal_length_ndarray[virginica_index_ndarray], sepal_width_ndarray[virginica_index_ndarray],
                color="green", marker="+", label="virginica")
    plt.legend(loc="best", title="iris type")  # NOTE
    plt.xlabel("sepal_length(cm)")
    plt.ylabel("sepal_width(cm)")
    plt.show()

    # 3. 使用逻辑回归分类器识别
    X = iris.data
    y = iris.target

    clf = linear_model.LogisticRegression()
    clf.fit(X, y)

    predict_sample = X[np.newaxis, 0]  # 待预测样本
    """
    X[0]: [5.1 3.5 1.4 0.2]
    predict_sample: [[5.1 3.5 1.4 0.2]]
    """
    print("predict_sample: {0}".format(predict_sample))
    # 预测所属目标类别
    print("predict: {0}".format(clf.predict(predict_sample)))    # [0]
    # 预测所属不同目标类别的概率
    print("predict_proba: {0}".format(clf.predict_proba(predict_sample)))
    # [[8.79681649e-01 1.20307538e-01 1.08131372e-05]]


def visualize_model():
    """
    可视化模型结果
    """
    iris = datasets.load_iris()

    # 只考虑前两个特征，即花萼长度(sepal length), 花萼宽度(sepal width)
    X = iris.data[:, 0:2]
    y = iris.target

    h = 0.02    # 网格大小
    # 将 X 的第一列(花萼长度)作为 x 轴，并求出 x 轴的最大值与最小值
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    # 将 X 的第二列(花萼宽度)作为 y 轴，并求出 y 轴的最大值与最小值
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # 使用 x 轴的最小值、最大值、步长生成数组，y 轴的最小值、最大值、步长生成数组
    # 然后使用 meshgrid 函数生成一个网格矩阵 xx 和 yy(xx 和 yy 的形状都一样)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # 调用 ravel() 函数将 xx 和 yy 平铺，然后使用 np.c_ 将平铺后的列表拼接
    # 生成需要预测的特征矩阵，每一行表示一个样本，每一列表示每个特征的取值
    pre_data = np.c_[xx.ravel(), yy.ravel()]

    Z = classify(X, y, pre_data)

    # Put the result into a color plot
    # 将预测结果 Z 的形状转为与 xx(或 yy)一样
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(8, 6))

    # 使用 pcolormesh 函数来填充颜色，对 xx，yy的位置来填充颜色，填充方案为 Z
    # cmap 表示使用的主题
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # 将训练数据所表示的样本点填充上颜色
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=plt.cm.Paired)

    plt.xlabel("sepal length")
    plt.ylabel("sepal width")

    # 设置坐标轴范围
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # 设置坐标轴刻度
    plt.xticks(np.arange(x_min, x_max, h * 10))
    plt.yticks(np.arange(y_min, y_max, h * 10))

    plt.show()


def classify(X, y, pre_data):
    """
    :param X: training features.
    :param y: training labels.
    :param pre_data: prediction features.
    :return: model predict result.
    """
    '''
    # 1. LR: LR算法的优点是可以给出数据所在类别的概率
    model = linear_model.LogisticRegression(C=1e5)
    """
    C: default: 1.0
    Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller
    values specify stronger regularization.
    """

    # 2. NB: 也是著名的机器学习算法, 该方法的任务是还原训练样本数据的分布密度, 其在多分类中有很好的效果
    from sklearn import naive_bayes
    model = naive_bayes.GaussianNB()    # 高斯贝叶斯

    # 3. KNN:
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()

    # 4. 决策树: 分类与回归树(Classification and Regression Trees, CART)算法常用于特征含有类别信息
    # 的分类或者回归问题，这种方法非常适用于多分类情况
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier()
    
    # 5. SVM: SVM是非常流行的机器学习算法，主要用于分类问题，
    # 如同逻辑回归问题，它可以使用一对多的方法进行多类别的分类
    from sklearn.svm import SVC
    model = SVC()
    '''

    # 6. MLP: 多层感知器(神经网络)
    from sklearn.neural_network import MLPClassifier
    # model = MLPClassifier(activation="relu", solver="adam", alpha=0.0001)
    # model = MLPClassifier(activation="identity", solver="adam", alpha=0.0001)
    # model = MLPClassifier(activation="logistic", solver="adam", alpha=0.0001)
    model = MLPClassifier(activation="tanh", solver="adam", alpha=0.0001)

    model.fit(X, y)
    Z = model.predict(pre_data)
    return Z


if __name__ == "__main__":
    # iris加载数据、模型训练、模型预测结果绘制的demo，有一些值得学习的地方
    # iris_demo()

    # 2. 可视化模型结果
    visualize_model()
