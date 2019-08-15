# -*- coding: utf-8 -*-
'''
树回归
(1) CART算法
(2) 回归与模型树
(3) 树剪枝
'''

from numpy import *

'''
(1) CART算法：使用二元切分来处理连续型变量
找到最佳的待切分特征：
    如果该节点不能再分，将该节点存为叶节点
    执行二元切分
    在右子树调用createTree()方法
    在左子树调用createTree()方法
'''
def loadDataSet(fileName):
    fr = open(fileName)
    dataMat = []
    for line in fr.readlines():
        line = line.strip().split('\t')
        # 讲每行映射成浮点数
        fltLine = list(map(float,line))
        dataMat.append(fltLine)
    return dataMat

'''
通过数组过滤将数据集合切分得到两个子集并返回

Args:
    dataMat: 数据集合
    feature: 待切分的特征
    value: 该特征的某个值

Returns:
    mat0: 特征值大于目标值的子集
    mat1: 特征值小于目标值的子集
'''
def binSplitDataSet(dataMat, feature, value):
    mat0 = dataMat[nonzero(dataMat[:,feature] > value)[0],:]
    mat1 = dataMat[nonzero(dataMat[:,feature] <= value)[0],:]
    return mat0, mat1

'''
生成叶节点，在回归树中，就是目标变量的均值

Args:
    dataSet: 样本点

Returns:
    mean(dataSet[:,-1]): 均值
'''
def regLeaf(dataSet):
    return mean(dataSet[:,-1])

'''
返回总方差，由均方差 * 样本点个数得到

Args:
    dataSet: 样本点

Returns:
    var(dataSet[:,-1]) * shape(dataset)[0]: 总方差
'''
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]

'''
树构建，该树包含待切分的特征，待切分的特征值，右子树，左子树

Args:
    dataSet: 数据集
    leafType: 建立叶节点的函数
    errType: 误差计算函数
    ops: 包含树构建所需其它参数的元组

Returns:
    retTree: 树
'''
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

'''
找到数据集切分的最佳位置

Args:
    dataSet: 数据集
    leafType: 建立叶节点的函数
    errType: 误差计算函数
    ops: 包含树构建所需其它参数的元组

Returns:
    bestIndex: 最佳切分特征
    bestValue: 最近切分特征值
''' 
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    # 容许的误差下降值
    tolS = ops[0]
    # 切分的最小样本数
    tolN = ops[1]
    # 如果只剩一个特征未被切分，则返回
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0
    # 对每个特征
    for featIndex in range(n-1):
        # 对每个特征值
        for splitVal in set(dataSet[:,featIndex].T.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 如果切分数据集后效果提升不够大，那么就不应该进行切分操作而直接创建叶节点
    if (S-bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 如果子集大小小于用户定义的tolN，那么也不应切割
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue

'''
(3) 树剪枝---后剪枝
一棵树如果节点太多，表明该模型可能对数据进行了过拟合。
后剪枝：利用测试集对树进行剪枝

测试输入变量是否是一棵树

Args:
    obj:测试对象

Returns:
    ture/flase: 是否为树
'''
def isTree(obj):
    return (type(obj).__name__ == 'dict')

'''
递归函数，从上往下遍历树直到叶节点为止。
对树进行塌陷处理（即返回树平均值）

Args:
    tree: 树

Returns:
    (tree['left']+tree['right'])/2: 叶节点的平均值
'''
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2

'''
基于已有的树切分测试数据：
    如果存在任一子集是一棵树，则在该子集递归剪枝过程
    计算将当前两个叶节点合并后的误差
    计算不合并的误差
    如果合并会降低误差的话，则将叶节点合并

Args:
    tree: 待剪枝的树
    testData: 剪枝所需的测试数据

Returns:
    tree/treeMean: 不合并/合并的树
'''
def prune(tree, testData):
    if shape(testData)[0] == 0:
        return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) + sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2
        errorMerge = sum(power(testData[:,-1]-treeMean,2))
        if errorMerge < errorNoMerge:
            print ("mergming")
            return treeMean
        else:
            return tree
    else:
        return tree

'''
模型树，把叶节点设定为分段线性函数

将数据集格式化

Args:
    dataSet: 数据集

Returns:
    ws: 回归系数
    X: 自变量
    Y: 目标变量
'''
def linearSolve(dataSet):
    m,n = shape(dataSet)
    X = mat(ones((m,n)))
    Y = mat(ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1]
    Y = dataSet[:,-1]
    xTx = X.T*X
    if linalg.det(xTx) == 0:
        raise NameError('this matrix is singular')
    ws = xTx.I * X.T * Y
    return ws, X, Y

'''
当数据不再需要切分的时候负责生成叶节点模型

Args:
    dataSet: 数据集

Returns:
    ws: 回归系数
'''
def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws

'''
在给定的数据集上计算误差

Args:
    dataSet: 数据集

Returns:
    sum(power(Y-yHat,2)): 误差
'''
def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y-yHat,2))

# '''
# 比较模型树，回归树，和一般回归方法的性能，通过相关系数(越接近1越好)

# 对回归树叶节点进行预测
# '''
# def regTreeEval(model, inDat):
#     return float(model)

# '''
# 对模型树节点进行预测
# '''
# def modelTreeEval(model, inDat):
#     n = shape(inDat)[1]
#     X = mat(ones((1,n+1)))
#     X[:,1:n+1] = inDat
#     return float(X * model)

# '''
# Args:
#     tree: 树
#     inData: 测试数据
#     modelEval: 树的模式
# '''
# def treeForeCast(tree, inData, modelEval=regTreeEval):
#     if not isTree(tree):
#         return modelEval(tree, inData)
#     if inData[tree['spInd']] > tree['spVal']:
#         if isTree(tree['left']):
#             return treeForeCast(tree['left'], inData, modelEval)
#         else:
#             return modelEval(tree['left'], inData)
#     else:
#         if isTree(tree['right']):
#             return treeForeCast(tree['right'], inData, modelEval)
#         else:
#             return modelEval(tree['right'], inData)

# '''
# 多次调用treeForeCast()

# Args:
#     tree: 树
#     testData: 测试数据
#     modelEval: 树的模式
# '''
# def createForeCast(tree, testData, modelEval=regTreeEval):
#     m = len(testData)
#     yHat = mat(zeros((m,1)))
#     for i in range(m):
#         yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
#     return yHat

if __name__ == '__main__':
    myDat = loadDataSet('ex0.txt')
    myMat = mat(myDat)
    #print (createTree(myMat))
    myDat2 = loadDataSet('ex2.txt')
    myMat2 = mat(myDat2)
    myTree = createTree(myMat2, ops=(0,1))
    myDatTest = loadDataSet('ex2test.txt')
    myMat2Test = mat(myDatTest)
    #print (prune(myTree, myMat2Test))
    myMat2 = mat(loadDataSet('exp2.txt'))
    # print (createTree(myMat2, modelLeaf, modelErr, (1,10)))
    # trainMat = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    # testMat = mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    # # 回归树
    # myTree = createTree(trainMat, ops=(1,20))
    # yHat = createForeCast(myTree, testMat[:,0])
    # print (corrcoef(yHat, testMat[:,1], rowvar=0)[0,1])
    # # 模型树
    # myTree = createTree(trainMat, modelLeaf, modelErr, ops=(1,20))
    # yHat = createForeCast(myTree, testMat[:,0])
    # print (corrcoef(yHat, testMat[:,1], rowvar=0)[0,1])
































