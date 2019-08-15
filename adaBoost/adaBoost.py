# -*- coding: utf-8 -*-
'''
利用AdaBoost元算法提高分类性能

元算法是对其他算法进行组合的一种方式。
使用集成方法的多种形式：
    不同算法的集成
    同一算法在不同设置下的集成
    数据集不同部分分配给不同分类器的集成
'''

from numpy import *

'''
集成方法有两种：bagging和boosting
bagging：从原始数据集选择s次后得到s个新数据集，每个数据集都是通过在
        原始数据集中随机选择一个样本得到。数据集可重复(抽样之后放回)
boosting：基于所有分类器的加权求和，权重不相等，上一轮错分的数据权重高。
        AdaBoost属于boosting

AdaBoost：训练每一个样本并赋予一个权重，这些权重构成向量D(初始化相等值)。
        对样本进行第二次分类，此时第一次分错的样本权重会提高。
        为了从弱分类器中得到最终的分类结果，每个分类器都有一个权重值alpha
        alpha = 0.5 * ln((1 - e)/e)    e为错误率
        计算出alpha后，对D进行更新
        如果该样本被分类正确
            D[i] = (D[i] * e的-alpha次方)/sum(D)
        如果该样本被分类错误
            D[i] = (D[i] * e的alpha次方)/sum(D)
        不断进行迭代直到错误率为0或弱分类器的数目达到用户指定值为止。
'''

'''
Returns:
    dataMat: 样本集
    classLabels: 标签
'''
def loadSimpData():
    dataMat=matrix([[1, 2.1],
                   [2, 1.1],
                   [1.3, 1],
                   [1, 1],
                   [2, 1]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels

'''
基于单层决策树(基于单个特征来做决策)构建弱分类器
通过阈值比较对数据进行分类

Args:
    dataMatrix: 数据集
    dimen: 某个特征
    threshVal: 阈值
    threshIneq: 阈值类型

Returns:
    retArray: 数据集对应标签
'''
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

'''
构建单层决策树

Args:
    dataArr: 样本集
    classLabels: 标签
    D: 权重比

Returns:
    bestStump: 存储给定权重向量D时所得到的最佳单层决策树相关信息的字典
    minError: 最小权重
    bestClasEst: 最优划分列表
'''
def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst=mat(zeros((m,1)))
    minError = inf
    # 在数据集所有特征上遍历
    for i in range(n):
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1, int(numSteps)+1):
            for inequal in ['lt', 'gt']:
                threshVal = rangeMin + j * float(stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m,1)))
                # 若不相等为1
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst

'''
基于单层决策树的AdaBoost训练过程
对每次迭代：
    buildStump()找到最佳单层决策树
    将最佳单层决策树加入到单层决策树数组
    计算alpha
    更新D
    更新累计类别估计值
    错误率为0则退出

Args:
    dataArr: 样本集
    classLabels: 标签
    numIt: 迭代次数

Returns:
    weakClassArr: 单层决策树数组
'''
def adaBoostTrainsDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)
    # 类别估计累计值
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print ("D:", D.T)
        # max()防止没有错误时发生除零溢出
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print ("classEst:", classEst.T)
        #  对应相乘，若是预测值正确为-alpha，错误为alpha
        expon = multiply(-1*alpha*mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D=D/D.sum()
        aggClassEst += alpha * classEst
        # sign()取正负，若不相等为true，相等为false，与ones()对应相乘，即错误处为1
        aggError = multiply(sign(aggClassEst)!=mat(classLabels).T, ones((m,1)))
        errorRate = aggError.sum()/m
        print ("total error:", errorRate)
        if errorRate == 0:
            break
    return weakClassArr

'''
测试算法：基于AdaBoost的分类

Args:
    dataToClass: 待分类数据
    classifierArr: 单层决策树数组

Returns:
    sign(aggClassEst): 分类结果
'''
def adaClassify(dataToClass, classifierArr):
    dataMatrix = mat(dataToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classEst*classifierArr[i]['alpha']
    #   print aggclassest
    return sign(aggClassEst)

'''
在一个难数据集上应用AdaBoost
在前面马疝病数据集上应用AdaBoost

加载数据函数

Args:
    fileName: 文件名

Returns:
    dataMat: 数据集
    labelMat: 标签
'''
def loadDataSet(fileName):
    # 读取列数
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

if __name__ == '__main__':
    dataMat, classLabels = loadSimpData()
    D = mat(ones((5,1))/5)
    # print (buildStump(dataMat, classLabels, D))
    # classifierArr = adaBoostTrainsDS(dataMat, classLabels, 30)
    # print (adaClassify([[5,5], [0,0]], classifierArr))
    dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArr = adaBoostTrainsDS(dataArr, labelArr, 10)
    testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
    prediction = adaClassify(testArr, classifierArr)
    errorArr = mat(ones((67,1)))
    print (errorArr[prediction != mat(testLabelArr).T].sum())




















