# -*- coding: utf-8 -*-
'''
决策树
(1) 基于最好的属性值划分数据集
(2) 构造决策树
(3) 使用决策树进行分类
(4) 存储决策树
'''

from math import log
import operator

'''
创建样本集和对应标签

Returns: 
    dataSet: 数据集
    labels: 标签
'''
def createDataSet():
    dataSet = [[1,1,'yes'],
             [1,1,'yes'],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet,labels

'''
信息增益：在划分数据集前后信息发生的变化。
计算熵。

Args:
    dataSet: 数据集

Returns:
    shannonEnt: 数据集的熵
'''
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for feat in dataSet:
        current = feat[-1]
        labelCounts[current] = labelCounts.get(current,0) + 1
    shannonEnt = 0
    for key in labelCounts:
        prob = labelCounts[key]/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

'''
划分数据集。
根据axis划分特征。

Args:
    dataSet: 数据集
    axis: 选取的特征项
    value: 该特征的值

Returns:
    retDataSet: 符合选取特征的数据集
'''
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for feat in dataSet:
        if feat[axis] == value:
            #去掉feat[axis]
            reducedFeat = feat[:axis]
            reducedFeat.extend(feat[axis+1:])
            retDataSet.append(reducedFeat)
    return retDataSet

'''
选择最好的数据集划分方式。

Args:
    dataSet: 数据集

Returns:
    bestFeature: 按此特征划分可以最多程度地降低熵。
'''
def chooseBestFeatureToSplit(dataSet):
    numOfFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0
    bestFeature = -1
    for i in range(numOfFeatures):
        #选择一项特征
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0
        # 计算每种划分方式的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/len(dataSet)
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        # 计算最好的信息增益
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

'''
Args:
    classList: 一项特征的列表。

Returns:
    sortedClass[0][0]: 返回出现次数最多的key。
'''
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClass = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)
    return sortedClass[0][0]

'''
创建树

Args:
    dataSet: 数据集
    labels: 标签

Returns:
    myTree:决策树
'''
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    #当类别标签完全一样时停止递归
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #当所有特征都遍历过后停止递归
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    copyLabels = labels[:]
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = copyLabels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(copyLabels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        #为了不改变labels
        subLabels = copyLabels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

'''
使用决策树进行分类

Args:
    inputTree: 输入树
    featLabels: 标签
    test: 测试向量

Returns:
    classLabel: 叶子节点值
'''
def classify(inputTree, featLabels, test):
    first = list(inputTree.keys())[0]
    second = inputTree[first]
    featIndex = featLabels.index(first)
    for key in second:
        if key == test[featIndex]:
            if type(second[key]).__name__=='dict':
                classLabel = classify(second[key], featLabels, test)
            else:
                classLabel = second[key]
    return classLabel

'''
使用pickle模块序列化对象以存储。

Args:
    inputTree: 树
    fileName: 文件名
'''
def storeTree(inputTree, fileName):
    import pickle
    #fw = open(fileName, 'wb')
    with open(fileName, 'wb') as fw:
        pickle.dump(inputTree, fw)
    #fw.close()

'''
读取文件
    
Args:
    fileName: 文件名
'''
def grabTree(fileName):
    import pickle
    fr=open(fileName, 'rb')
    return pickle.load(fr)

'''
预测隐形眼镜类型
隐形眼镜类型包括硬材质、软材质和不适合佩戴隐形眼镜。

Returns:
    lensesTree: 预测隐形眼镜决策树
'''
def contactLensesType():
    #fr = open('lenses.txt')
    with open('lenses.txt') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
        lensesLabels = ['age', 'prescript', 'astigmatic', 'teatRate']
        lensesTree = createTree(lenses, lensesLabels)
    return lensesTree

if __name__ == '__main__':
    myDat, labels = createDataSet()
    print (calcShannonEnt(myDat))
    print (splitDataSet(myDat, 0, 1))
    print (chooseBestFeatureToSplit(myDat))
    print (createTree(myDat, labels))

    import treeplotter
    myTree = treeplotter.retrieveTree(0)
    print (classify(myTree, labels, [1,1]))

    storeTree(myTree, 'classifierStorge.txt')
    print (grabTree('classifierStorge.txt'))

    print (contactLensesType())
    treeplotter.createPlot(contactLensesType())











