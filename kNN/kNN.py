# -*- coding: utf-8 -*-
'''
k-近邻算法
存在一个样本集和对应标签，
查找与待分类数据距离最近的样本点，（取前k个最相似的数据）
选择k个中出现次数最多的分类作为新数据的分类
'''

from numpy import *
import operator
from os import listdir

'''
创建样本集和对应标签

Returns: 
    group: 数据集
    labels: 标签
'''
def createDataSet():
    group = array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

'''
(1) 计算已知类别数据集中的点与当前点之间的距离
(2) 按照距离升序排序
(3) 选取与当前点距离最小的k个点
(4) 确定前k个点所在类别的出现频率
(5) 返回出现频率最高的类别作为当前点的预测分类

Args: 
    inX: 待分类数据
    dataSet: 样本集
    labels: 样本集对应标签
    k: 选取前k个点

Returns:
    sortedClassCount[0][0]: 前k个点出现频率最高的类别
'''
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    # 索引值升序排序
    sortedDisIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        votelabel=labels[sortedDisIndicies[i]]
        classCount[votelabel] = classCount.get(votelabel, 0) + 1
    # 字典降序排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

'''
在约会网站上使用k-近邻算法
三种特征：每年获得的飞行里程数，玩游戏所耗时间百分比，每周消费冰激凌公升数
(1) 准备数据：将文本记录转换为Numpy的解析程序

Args:
    fileName: 文件名

Returns:
    returnMat: numberOfLines*3的特征数据
    classLabelVector: 对应标签
'''
def file2matrix(fileName):
    fr = open(fileName)
    arrayLines = fr.readlines()
    numberOfLines = len(arrayLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

'''
(2) 准备数据：归一化数值
newValue = (oldValue - min)/(max - min)

Args:
    dataSet: 特征数组

Returns:
    normDataSet: 归一化后的特征数组
    ranges: 特征数据变化范围
    minVals: 每个特征的最小值
'''
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals

'''
(3) 测试算法：作为完整程序验证分类器
使用已有数据的90%作为训练样本，10%用于测试分类器，检测分类器的正确性。
'''
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minvals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTest = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTest):
        classifierResult = classify0(normMat[i,:], normMat[numTest:m,:], datingLabels[numTest:m], 3)
        print ("the classifier came back with: %d, the real answer is: %d." % (classifierResult,datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print ("the total error rate is : %f%%" % (errorCount/numTest*100))

'''
(4) 使用算法：构建完整可用系统
'''
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage time playing video games?"))
    ffMiles = float(input("frequent flier miles?"))
    iceCream = float(input("ice cream consumed?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minvals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print ("you will probably like this person %s" % (resultList[classifierResult-1]))

'''
手写识别系统
(1) 准备数据：将图像转换为测试向量 
32*32的二进制图像转换为1*1024的向量 

Args:
    fileName: 文件名

Returns:
    returnVect: 1*1024的Numpy数组
'''
def img2vector(fileName):
    returnVect = zeros((1,1024))
    with open(fileName) as fr:
        for i in range(32):
            lineStr=fr.readline()
            for j in range(32):
                returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

'''
(2) 测试算法：使用k-近邻算法识别手写数字
'''
def handwritingClassTest():
    hwLabels = []
    # 获取文件夹内的所有文件名
    traingFileList = listdir('trainingDigits')
    m = len(traingFileList)
    traingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = traingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        traingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileTest = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileTest)
    for i in range(mTest):
        fileNameStr = testFileTest[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, traingMat, hwLabels, 3)
        print ("the classifier came back with: %d, the real answer is %d" % (classifierResult, classNumStr))
        if(classifierResult != classNumStr):
            errorCount += 1.0
    print ("\nthe total number of errors is:%d" % errorCount)
    print ("\nthe total error rate is:%f" % (errorCount/mTest))

if __name__ == '__main__':
    # group, labels = createDataSet()
    # datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    # normMat, ranges, minVals = autoNorm(datingDataMat)
    # datingClassTest()
    # classifyPerson()
    handwritingClassTest()
















