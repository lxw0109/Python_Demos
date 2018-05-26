#!/usr/bin/env python3
# coding: utf-8
# File: sklearn_demo.py
# Author: lxw
# Date: 5/9/18 11:16 PM
"""
References:
1. [ML神器：sklearn的快速使用](https://www.cnblogs.com/lianyingteng/p/7811126.html)
2. [[译]使用scikit-learn进行机器学习(scikit-learn教程1)](http://www.cnblogs.com/taceywong/p/4568806.html)
3. [【机器学习实验】scikit-learn的主要模块和基本使用](https://www.jianshu.com/p/1c6efdbce226)
4. [Scikit-learn：数据预处理Preprocessing data](https://blog.csdn.net/pipisorry/article/details/52247679)
"""

# 1. 获取数据
from sklearn import datasets
from sklearn.datasets.samples_generator import make_classification


def prepare_data():
    # 1.1 导入sklearn数据集
    iris = datasets.load_iris()    # type(iris): <sklearn.datasets.base.Bunch>
    x = iris.data    # x.shape: (150, 4)
    y = iris.target    # y.shape: (150,)
    print(x)
    print(y)

    # 1.2 创建数据集
    x, y = make_classification(n_samples=6, n_features=5, n_informative=2, n_redundant=2,
                               n_classes=2, n_clusters_per_class=2, scale=1.0, random_state=20)
    # n_samples：指定样本数, n_features：指定特征数, n_classes：指定几分类, random_state：随机种子，使得随机性可重现
    for x_, y_ in zip(x, y):
        print("{0}: {1}".format(x_, y_))


# 2. 数据预处理
from sklearn import preprocessing
import numpy as np

def preprocess_data():
    '''
    # 2.1 数据归一化: 针对每个axis进行归一化，不是所有axis的数据一起归一化
    data = [[0, 1], [1, 4], [3, 2], [1, 1]]

    # 1) 基于mean和std的标准化
    scaler = preprocessing.StandardScaler().fit(data)
    # print(scaler.std_, scaler.mean_)
    data_scaled = scaler.transform(data)
    print(data_scaled)

    # 2) 将每个特征值归一化到一个固定范围
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 4)).fit(data)    # feature_range: 归一化范围, 用()括起来.
    """
    MinMaxScaler: The transformation is given by:
    ```
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = X_std * (max - min) + min
    ```
    where min, max = feature_range.
    """
    # print(scaler.data_max_)    # [ 3.  4.]
    data_scaled = scaler.transform(data)
    print(data_scaled)

    # 2.2 标准化(normalize)
    # NOTE: 当要计算两个样本的相似度时必不可少的一个操作，就是标准化
    # 其思想是：首先求出样本的p-范数，然后该样本的所有元素都除以该范数，这样使得每个样本的范数都为1
    data = [[0, 1], [1, 4], [3, 2], [1, 1]]
    """
    # NOTE: 每个样本normalize, 不是全体样本一起normalize
    例如: [0, 1]自己normalize, [1, 4]自己normalize, ...
    """
    x_normalized = preprocessing.normalize(data, norm="l2")
    print(x_normalized)
    '''

    # 2.3 one-hot编码
    # one-hot编码是一种对离散特征值的编码方式，在LR模型中常用到
    # data = np.array([0, 5, 2, 1, 4, 3])
    data = np.array([0, 5, 12, 1, 14, 3])
    print(data)    # [0 5 2 1 4 3]
    data = data.reshape(-1, 1)
    print(data)
    """
    [[0]
     [5]
     [2]
     [1]
     [4]
     [3]
    """
    # data = [[0, 5], [2, 3], [3, 4], [1, 6]]
    encoder = preprocessing.OneHotEncoder().fit(data)
    print(encoder.transform(data).toarray())
    """
    # data = np.array([0, 5, 2, 1, 4, 3])
    [[ 1.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  1.]
     [ 0.  0.  1.  0.  0.  0.]
     [ 0.  1.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  1.  0.]
     [ 0.  0.  0.  1.  0.  0.]
     
    # data = [[0, 5], [2, 3], [3, 4], [1, 6]]
    [[ 1.  0.  0.  0.  0.  0.  1.  0.]
     [ 0.  0.  1.  0.  1.  0.  0.  0.]
     [ 0.  0.  0.  1.  0.  1.  0.  0.]
     [ 0.  1.  0.  0.  0.  0.  0.  1.]]
    """


# 3. 数据集拆分: 将数据集划分为训练集和测试集/把训练数据集进一步拆分成训练集和验证集
from sklearn.model_selection import train_test_split

def data_split():
    x = np.array([[0, 1], [1, 4], [3, 2], [1, 1], [-1, -2]])
    y = np.array([-1, -2, -3, -4, 5])
    y = y.reshape(-1, 1)

    """
    iris = datasets.load_iris()    # type(iris): <sklearn.datasets.base.Bunch>
    x = iris.data    # x.shape: (150, 4)
    y = iris.target    # y.shape: (150,)
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    print(x_train)
    print(x_train.shape)    # (105, 4)
    print(x_test)
    print(x_test.shape)    # (45, 4)
    print(y_train)
    print(y_train.shape)    # (105,)
    print(y_test)
    print(y_test.shape)    # (45,)


# 4. 定义模型
def models():
    from sklearn.linear_model import LinearRegression
    # 4.1 线性回归
    model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
    """
    fit_intercept:是否计算截距, False-模型没有截距
    normalize: 当fit_intercept设置为False时, 该参数将被忽略. 若fit_intercept为True, 则回归前的回归系数X将通过
    减去平均值并除以l2-范数而归一化
    n_jobs：指定线程数

    y = a * x + b       a: model.coef_, b: model.intercept_
    """

    # 4.2 逻辑回归LR
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty="l2", dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
                               class_weight=None, random_state=None, solver="liblinear", max_iter=100, multi_class="ovr",
                               verbose=0, warm_start=False, n_jobs=1)
    """
    penalty: 使用指定正则化项(默认：l2). dual: Prefer dual=False when n_samples > n_features.
    tol: Tolerance for stopping criteria. C：正则化强度的倒数，值越小正则化强度越大
    n_jobs: 指定线程数. If given a value of -1, all cores are used.
    random_state:随机数生成器
    """

    # 4.3 朴素贝叶斯算法NB
    from sklearn import naive_bayes
    model = naive_bayes.GaussianNB()    # 高斯贝叶斯
    model = naive_bayes.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
    model = naive_bayes.BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
    """
    文本分类问题常用MultinomialNB
    alpha：平滑参数. Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    fit_prior：是否要学习类的先验概率；false-使用统一的先验概率
    class_prior: 是否指定类的先验概率；若指定则不能根据参数调整
    binarize: 二值化的阈值，若为None，则假设输入由二进制向量组成
    """

    # 4.4 决策树DT
    from sklearn import tree
    model = tree.DecisionTreeClassifier(criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                        min_weight_fraction_leaf=0.0, max_features=None, random_state=None,
                                        max_leaf_nodes=None, min_impurity_split=None,
                                        class_weight=None, presort=False)
    """参数
    criterion: 特征选择准则gini/entropy
    max_depth:树的最大深度，None-尽量下分
    min_samples_split: 分裂内部节点，所需要的最小样本树
    min_samples_leaf: 叶子节点所需要的最小样本数
    max_features: 寻找最优分割点时的最大特征数
    max_leaf_nodes: 优先增长到最大叶子节点数
    min_impurity_decrease: 如果这种分离导致杂质(impurity)的减少大于或等于这个值，则节点将被拆分
    """

    # 4.5 支持向量机SVM
    from sklearn.svm import SVC
    model = SVC(C=1.0, kernel="rbf", gamma="auto")
    """
    C: 误差项的惩罚参数C
    gamma: 核相关系数. 浮点数, If gamma is "auto" then 1/n_features will be used instead.
    """


    # 4.6 K近邻算法KNN
    from sklearn import neighbors
    #定义kNN分类模型
    model = neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=1)    # 分类
    model = neighbors.KNeighborsRegressor(n_neighbors=5, n_jobs=1)    # 回归

    # 4.7 多层感知机(神经网络)
    from sklearn.neural_network import MLPClassifier
    # 定义多层感知机分类算法
    model = MLPClassifier(activation="relu", solver="adam", alpha=0.0001)
    """
    hidden_layer_sizes: tuple. The ith element represents the number of neurons in the ith hidden layer.
    activation: 激活函数
    solver: The solver for weight optimization. 权重优化算法{"lbfgs", "sgd", "adam"}
    alpha: L2惩罚(正则化项)参数
    """
    return model


# 5. 模型评估与选择
def model_selection(model):
    # 5.1 交叉验证
    from sklearn.model_selection import cross_val_score
    X = np.array([1, 2, 3, 4, 5])
    cross_val_score(model, X, y=None, scoring=None, cv=None, n_jobs=1)
    """参数
    model: 拟合数据的模型. The object to use to fit the data.
    X: The data to fit. Can be for example a list, or an array.
    cv: k-fold. None, to use the default 3-fold cross validation.
    scoring: 打分参数-"accuracy", "f1", "precision", "recall", "roc_auc", "neg_log_loss"等
    """

    # 5.2 检验曲线: 使用检验曲线可以更加方便的改变模型参数, 获取模型表现
    from sklearn.model_selection import validation_curve
    train_score, test_score = validation_curve(model, X, y, param_name, param_range, cv=None, scoring=None, n_jobs=1)
    """
    model: object type that implements the “fit” and “predict” methods. An object of that type which is cloned for each validation.
    X, y: 训练集的特征和标签
    param_name: 将被改变的参数的名字
    param_range: 参数的改变范围
    cv: k-fold
    
    返回值
    train_score: 训练集得分(array)
    test_score: 验证集得分(array)
    """

# 6. 保存模型
def save_model(model, X_test):
    import pickle

    # 保存模型
    # 6.1 保存为pickle文件
    with open("model.pickle", "wb") as f:
        pickle.dump(model, f)
    # 读取模型
    with open("model.pickle", "rb") as f:
        model = pickle.load(f)
    model.predict(X_test)

    # 6.2 sklearn自带方法joblib
    from sklearn.externals import joblib

    # 保存模型
    joblib.dump(model, "model.pickle")

    #载入模型
    model = joblib.load("model.pickle")

# 7. 使用实例
def demo():
    """
    References: [[译]使用scikit-learn进行机器学习(scikit-learn教程1)](http://www.cnblogs.com/taceywong/p/4568806.html)
    :return: 
    """
    from sklearn import svm
    from sklearn import datasets
    import matplotlib.pyplot as plt

    digits = datasets.load_digits()

    clf = svm.SVC(gamma=0.001, C=100)
    clf.fit(digits.data[:-1], digits.target[:-1])
    result = clf.predict(digits.data[-1])
    print(result)
    print(digits.images[-1])

    import pickle
    obj = pickle.dumps(clf)
    clf2 = pickle.loads(obj)
    clf2.predict(digits.data[0])
    print(digits.images[0])
    print(digits.target[0])

    plt.figure(1, figsize=(3, 3))
    plt.imshow(digits.images[0], cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.show()

# 2'. 特征选择(感觉应该放到数据预处理之后进行)
def feature_selection():
    from sklearn import datasets
    from sklearn import metrics
    from sklearn.ensemble import ExtraTreesClassifier

    import random
    random.seed(0)    # NO USE

    model = ExtraTreesClassifier()
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    model.fit(X, y)

    # display the relative importance of each attribute
    print(model.feature_importances_)
    # [0.08144046 0.07187995 0.3553828  0.49129679]


# 4'. 定义模型
def models_1():
    from sklearn import datasets
    from sklearn import metrics

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    # print(X.shape)    # (150, 4)
    # print(y.shape)    # (150,)

    """
    # 4.1 LR: 大多数问题都可以归结为二元分类问题, LR算法的优点是可以给出数据所在类别的概率
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()

    # 4.2 朴素贝叶斯: 也是著名的机器学习算法, 该方法的任务是还原训练样本数据的分布密度, 其在多分类中有很好的效果
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()

    # 4.3 K近邻
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()

    # 4.4 决策树: 分类与回归树(Classification and Regression Trees, CART)算法常用于特征含有类别信息
    # 的分类或者回归问题，这种方法非常适用于多分类情况
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier()
    """

    # 4.5 支持向量机: SVM是非常流行的机器学习算法，主要用于分类问题，
    # 如同逻辑回归问题，它可以使用一对多的方法进行多类别的分类
    from sklearn.svm import SVC
    model = SVC()

    model.fit(X, y)
    print(model)

    # make predictions
    expected = y
    predicted = model.predict(X)

    # summarize the fit of the model.
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))

# 8. 优化算法参数
def optimize_params():
    """
    一项更加困难的任务是构建一个有效的方法用于选择正确的参数，我们需要用搜索的方法来确定参数
    """
    import numpy as np

    from sklearn import datasets
    from sklearn.linear_model import Ridge
    from sklearn.grid_search import GridSearchCV

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # prepare a range of alpha values to test
    alphas = np.array([1, 0.1, 0.01, 0.001, 0.0001, 0])
    # create and fit a ridge regression model, testing each alpha
    model = Ridge()
    grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
    # print(dict(alpha=alphas))
    grid.fit(X, y)
    print(grid)

    # summarize the results of the grid search
    print(grid.best_score_)
    print(grid.best_estimator_.alpha)

    # 有时随机从给定区间中选择参数是很有效的方法，然后根据这些参数来评估算法的效果进而选择最佳的那个
    from scipy.stats import uniform as sp_rand
    from sklearn.grid_search import RandomizedSearchCV

    # prepare a uniform distribution to sample for the alpha parameter
    param_grid = {"alpha": sp_rand()}

    # create and fit a ridge regression model, testing random alpha values
    model = Ridge()
    rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100)
    rsearch.fit(X, y)
    print(rsearch)

    # summarize the results of the random parameter search
    print(rsearch.best_score_)
    print(rsearch.best_estimator_.alpha)

# 2". 数据预处理
from sklearn import preprocessing

def preprocess_data_1():
    """
    References: [Scikit-learn：数据预处理Preprocessing data](https://blog.csdn.net/pipisorry/article/details/52247679)
    """
    pass


if __name__ == "__main__":
    # 1. 获取数据
    # prepare_data()

    # 2. 数据预处理
    # preprocess_data()

    # 2'. 特征选择(感觉应该放到数据预处理之后进行)
    # feature_selection()

    # 2". 数据预处理
    # preprocess_data_1()

    # 3. 数据集拆分: 将数据集划分为训练集和测试集/把训练数据集进一步拆分成训练集和验证集
    # data_split()

    # 4. 定义模型
    # models()

    # 4'. 定义模型
    models_1()

    # 7. 使用实例
    # demo()

    # 8. 优化算法参数
    # optimize_params()
