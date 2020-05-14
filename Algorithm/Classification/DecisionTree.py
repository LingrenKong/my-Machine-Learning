# _*_coding:utf-8_*_
"""
决策树算法实现
Author: Lingren Kong
Created Time: 2020/5/14 19:50
"""


def createDataSetAll():
    """
    导入数据，数据集有八个特征 '色泽', '根蒂', '敲声', '纹理'，'脐部'，'触感'，'密度'，'含糖率'
    """
    import pandas as pd
    dataSet = pd.read_csv('西瓜3.0完整版.csv', encoding='gbk', index_col=0)
    train = [0, 1, 2, 5, 6, 9, 13, 14, 15, 16]
    test = [3, 4, 7, 8, 10, 11, 12]
    return dataSet.iloc[train, :], dataSet.iloc[test, :], dataSet.columns


def get_Values(dataSet, labels):
    """
    获得特征的取值，为分支划分做准备
    Parameters
    ----------
    dataSet:一个数据集
    labels:数据集各项对应的标签

    Returns：labelsCounts
    数据集中每个特征的所有取值，表现为字典形式【对于数值给出二类标签】；键是特征名，值是对应特征的所有取值

    -------

    """
    labelsCounts = {}  # 初始化字典
    for label in labels:  # 遍历特征集
        if dataSet[label].dtype == float:
            labelsCounts[label] = set(['<=数值', '>数值'])  # 对于数值给出二类标签】
        else:
            featValues = dataSet[label]  # 所有取值
            labelsCounts[label] = set(featValues)  # 将去重后的数据集合放入字典中，键名为特征名字
    return labelsCounts


def calcShannonEnt(dataSet):
    """
    计算香农熵，用于判断划分
    Parameters
    ----------
    dataSet:数据集（可能是原始数据集，有可能是划分子集）

    Returns:数据集的信息熵，根据分类标签信息确定
    -------

    """
    from math import log
    numEntries = len(dataSet)  # 数据总量
    labelCounts = {}  # 创建一个数据字典，用来计数各个类别
    for featVec in dataSet.values:  # 每次取一行
        currentLabel = featVec[-1]  # 默认每行最后一列的元素是样本类标签
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0.0  # 不存在的类别要进行添加，初始为0
        labelCounts[currentLabel] += 1  # 计数累进
    # print(labelCounts)
    shannonEnt = 0.0  # 从0累加信息熵
    for key in labelCounts:  # 遍历数据字典的键
        prob = float(labelCounts[key]) / numEntries  # 计算数据集i类样本所占比例P_i
        shannonEnt -= prob * log(prob, 2)  # 以二为底
    return shannonEnt


# 计算样本集中类别数最多的类别
def calmaxCnt(dataSet):
    """
    计算样本集中类别数最多的类别
    【用于叶节点的类别确定】
    Parameters
    ----------
    dataSet:数据集

    Returns:在输入数据集中类别数最多的类别名称
    -------

    """
    classCount = {}  # 创建字典
    for featVec in dataSet.values:  # 对数据集中每一行遍历
        currentLabel = featVec[-1]  # 默认每行最后一列的元素是样本类标签
        if currentLabel not in classCount.keys():  # 思路和之前相同
            classCount[currentLabel] = 0
        classCount[currentLabel] += 1
    items = list(classCount.items())  # 转为列表
    items.sort(key=lambda x: x[1], reverse=True)  # 列表以值来排序（从大到小）
    return items[0][0]  # 输出类别数最多的类别名称


def majorityCnt(classList):
    """
    返回该数据集中类别数最多的类名
    该函数使用分类名称的列表(某个数据集或者其子集的)，然后创建键值为classList中唯一值的数据字典。
    字典对象的存储了classList中每个类标签出现的频率。最后利用operator操作键值排序字典，
    并返回出现次数最多的分类名称

    Parameters
    ----------
    classList:分类类别列表

    Returns:子节点的分类
    -------
    数据集已经处理了所有属性，但是类标签依然不是唯一的，则采用多数判决的方法决定该子节点的分类
    """
    import operator
    classCount = {}  # 创建字典
    for vote in classList:  # 对类名列表遍历
        if vote not in classCount.keys():  # 键已存在字典中+1,不存在字典中创建后初始为0后+1
            classCount[vote] = 0
        classCount[vote] += 1
    # print(classCount)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),
                              reverse=True)  # 将字典转换成列表并按照值（[i][1]）进行从大到小排序
    return sortedClassCount[0][0]


def splitDataSet(dataSet, bestFeat, value, method, save=False):
    """
    划分数据集，为下一层计算准备，对于连续和离散等情况下提供了不同的方法
    Parameters
    ----------
    dataSet：数据集
    bestFeat：划分的目标列
    value：选择的值
    method：划分方法
    save：是否保存作为划分依据的列

    Returns
    -------

    """
    if method == '=':
        t = dataSet[dataSet[bestFeat] == value]
        if save:
            return t
        else:
            return t.drop(bestFeat, axis=1)  # 返回删去该列的结果
    elif method == '<=':
        t = dataSet[dataSet[bestFeat] <= value]
        if save:
            return t
        else:
            return t.drop(bestFeat, axis=1)  # 返回删去该列的结果
    elif method == '>':
        t = dataSet[dataSet[bestFeat] > value]
        if save:
            return t
        else:
            return t.drop(bestFeat, axis=1)  # 返回删去该列的结果


def BestDividing(dataSet, bestFeat):
    """
    选出连续性特征最优划分点+返回其信息熵
    Parameters
    ----------
    dataSet:数据集
    bestFeat：对哪个连续特征进行判断

    Returns：最优划分点+其信息熵
    -------

    """
    import numpy as np
    bestEntropy = 1000.0  # 初始设一个不可能的超大熵
    bestValue = 0.0
    features = np.sort(dataSet[bestFeat].values)
    uni = set()  # 利用集合去重
    for i in range(len(features) - 1):
        uni.add((features[i] + features[i + 1]) / 2)
    div = list(uni)
    for value in div:  # 遍历每一个分点
        subDataSet1 = splitDataSet(dataSet, bestFeat, value, "<=")  # 分别计算大于该分点与小于该分点的
        subDataSet2 = splitDataSet(dataSet, bestFeat, value, ">")
        prob1 = len(subDataSet1) / float(len(dataSet))
        prob2 = len(subDataSet2) / float(len(dataSet))
        tempEntropy = prob1 * calcShannonEnt(subDataSet1) + prob2 * calcShannonEnt(subDataSet2)
        if tempEntropy < bestEntropy:
            bestEntropy = tempEntropy
            bestValue = value
    return bestEntropy, bestValue


def chooseBestFeatureToSplit(dataSet):
    """
    选出最优划分特征bestFeat
    Parameters
    ----------
    dataSet:数据集

    Returns:最好的划分维度（采用列名）
    -------

    """
    numFeatures = len(dataSet.columns) - 1  # 特征个数（不含最后的标签）
    baseEntropy = calcShannonEnt(dataSet)  # 计算父样本集的信息熵
    bestInfoGain = 0.0  # 初始化信息增益为0.0
    bestFeature = -1  # 初始化最佳特征索引维度
    for i in range(numFeatures):  # 遍历每个特征
        label = dataSet.columns[i]
        if dataSet[label].dtype == float:  # 连续特征操作
            tempEntropy, value = BestDividing(dataSet, label)
        else:
            featList = dataSet[label].values  # 所有可选取值
            uni = set(featList)  # 去重
            tempEntropy = 0.0
            for value in uni:  # 遍历该特征每一种取值结果
                subDataSet = splitDataSet(dataSet, label, value, "=")  # 得到子集（由于是数据只用一次，所以子集不含这一列save默认false）
                prob = len(subDataSet) / float(len(dataSet))  # 计算各个频率
                tempEntropy += prob * calcShannonEnt(subDataSet)  # 累计信息熵
        gain = baseEntropy - tempEntropy  # 这个feature的infoGain
        if (gain > bestInfoGain):  # 选择最大的信息增益gain对应的特征
            bestInfoGain = gain
            bestFeature = label
    print("最佳信息增益是：", bestInfoGain)
    return bestFeature


def createTree(dataSet, labelscounts):
    """
    采取多重字典来构建树
    Parameters
    ----------
    dataSet：数据集
    labelscounts：当前数据情况下每个标签剩余的可能类别

    Returns：一颗决策树
    -------

    """
    classList = dataSet.values[:, -1]  # 分类标签
    if (classList == classList[0]).all():
        return classList[0]  # 只有一个类别了
    if len(dataSet.columns) == 2:
        # 划分直到只有一个特质后
        return majorityCnt(classList)  # 选多的那个
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 否则可分，进行划分并取得最大的分法列名
    # 这里直接使用字典变量来存储树信息，这对于绘制树形图很重要。
    if dataSet[bestFeat].dtype == float:  # 处理连续性变量
        entropy, value = BestDividing(dataSet, bestFeat)
        bestFeatLabel = bestFeat + "<=" + str(value)
        DecisionTree = {bestFeatLabel: {}}
        # --------
        # 处理小于等于的划分
        subdataSet = splitDataSet(dataSet, bestFeat, value, "<=", save=True)  # 划分且不去除该列，因为连续
        if len(subdataSet) == 0:  # 若划分出的数据子集为空集
            DecisionTree[bestFeatLabel]['Y'] = calmaxCnt(dataSet)  # 子集为叶节点，用父集合最多的那个
        else:
            DecisionTree[bestFeatLabel]['Y'] = createTree(subdataSet, labelscounts)  # 递归继续
        # --------
        # 处理大于划分值
        subdataSet = splitDataSet(dataSet, bestFeat, value, ">", save=True)
        if len(subdataSet) == 0:
            DecisionTree[bestFeatLabel]['N'] = calmaxCnt(dataSet)
        else:
            DecisionTree[bestFeatLabel]['N'] = createTree(subdataSet, labelscounts)
    else:  # 离散型
        bestFeatLabel = bestFeat
        DecisionTree = {bestFeatLabel: {}}  # 树的嵌套
        uni = labelscounts[bestFeatLabel]  # 获得最佳特征对应的所有特征值取值
        for value in uni:  # 对所有特征取值遍历
            subdataSet = splitDataSet(dataSet, bestFeat, value, "=")  # 划分出数据子集
            if len(subdataSet) == 0:  # 若划分出的数据子集为空集
                DecisionTree[bestFeatLabel][value] = calmaxCnt(dataSet)  # 子集为叶节点，用父集合最多的那个
            else:
                DecisionTree[bestFeatLabel][value] = createTree(subdataSet, labelscounts)  # 按照最优的进行递归
    return DecisionTree  # 返回字典形式树结构信息





def createTreePre(dataSet, test, labelscounts):
    """
    预剪枝模式
    Parameters
    ----------
    dataSet：数据集
    labelscounts：当前数据情况下每个标签剩余的可能类别

    Returns：一颗决策树
    -------

    """
    classList = dataSet.values[:, -1]  # 分类标签
    if (classList == classList[0]).all():
        return classList[0]  # 只有一个类别了
    if len(dataSet.columns) == 2:
        # 划分直到只有一个特质后
        return majorityCnt(classList)  # 选多的那个
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 否则可分，进行划分并取得最大的分法列名
    # 这里直接使用字典变量来存储树信息，这对于绘制树形图很重要。
    if dataSet[bestFeat].dtype == float:  # 处理连续性变量
        entropy, value = BestDividing(dataSet, bestFeat)
        bestFeatLabel = bestFeat + "<=" + str(value)
        DecisionTree = {bestFeatLabel: {}}
        # --------
        # 处理小于等于的划分
        subdataSet = splitDataSet(dataSet, bestFeat, value, "<=", save=True)  # 划分且不去除该列，因为连续
        if len(subdataSet) == 0:  # 若划分出的数据子集为空集
            DecisionTree[bestFeatLabel]['Y'] = calmaxCnt(dataSet)  # 子集为叶节点，用父集合最多的那个
        else:
            DecisionTree[bestFeatLabel]['Y'] = createTreePre(subdataSet, test, labelscounts)  # 递归继续
        # --------
        # 处理大于划分值
        subdataSet = splitDataSet(dataSet, bestFeat, value, ">", save=True)
        if len(subdataSet) == 0:
            DecisionTree[bestFeatLabel]['N'] = calmaxCnt(dataSet)
        else:
            DecisionTree[bestFeatLabel]['N'] = createTreePre(subdataSet, test, labelscounts)
    else:  # 离散型
        bestFeatLabel = bestFeat
        DecisionTree = {bestFeatLabel: {}}  # 树的嵌套
        uni = labelscounts[bestFeatLabel]  # 获得最佳特征对应的所有特征值取值
        for value in uni:  # 对所有特征取值遍历
            subdataSet = splitDataSet(dataSet, bestFeat, value, "=")  # 划分出数据子集
            if len(subdataSet) == 0:  # 若划分出的数据子集为空集
                DecisionTree[bestFeatLabel][value] = calmaxCnt(dataSet)  # 子集为叶节点，用父集合最多的那个
            else:
                DecisionTree[bestFeatLabel][value] = createTreePre(subdataSet, test, labelscounts)  # 按照最优的进行递归
    return DecisionTree  # 返回字典形式树结构信息


if __name__ == '__main__':
    """
    导入一份西瓜书的数据，用于模型的检验，采用西瓜数据3.0内容，按照西瓜2.0进行划分
    """
    train, test, labels = createDataSetAll()
    print(labels)
    labelsCounts = get_Values(train, labels)
    print(labelsCounts)
