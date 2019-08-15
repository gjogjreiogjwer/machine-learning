# -*- coding: utf-8 -*-
'''
Logistic回归

用一条直线（最佳拟合直线）对数据点进行拟合，该过程称为回归。
把每个特征乘以一个回归系数，所有的结果值相加，总和带入sigmoid函数，得到一个范围在
0-1之间的数值。大于0.5的数据被分入1类，小于0.5的数据被分入0类。

'''

from numpy import *

'''
便利函数，打开文件并逐行读取

Returns:
    dataMat: 样本集(x0(0), x1, x2)
    labelMat: 标签
'''
def loadDataSet():
    dataMat = []
    labelMat = []
    for line in open('testSet.txt'):
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

'''
g(z)=1/(1+exp(-z))
z=w0x0+w1x1+...+wnxn
z=w.T * x

Returns:
    1.0/(1+exp(-inx)): 用于分类，大于0.5为1类；小于0.5为0类。
'''
def sigmoid(inx):
    return 1.0/(1 + exp(-inx))

'''
梯度上升法：
要找到某函数的最大值，沿着该函数的梯度方向探寻。
迭代公式：
w = w + alpha * (i从1到m)((label[i] - sigmoid(data[i])) * data[i]]
alpha是步长

z=w0x0 + w1x1 + w2x2
x0为全1向量，令z = 0, 即sigmoid(0) = 0.5（分割线方程）
0=w0 + w1x1 + w2x2
x1为横坐标， x2为纵坐标
这个方程未知的参数为w0，w1，w2，也就是我们需要求的回归系数(最优参数)。

Args:
    dataMatIn: 样本
    classLabels: 标签

Returns:
    weights: 回归系数
'''
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    # 迭代次数
    cycle = 500
    weights = ones((n, 1))
    for k in range(cycle):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

'''
画出数据集和Logistic回归最佳拟合直线

Args:
    weights: 回归系数
'''
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = dataArr.shape[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if labelMat[i] == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

'''
随机梯度上升：在线算法
一次仅用一个样本点来更新回归系数，用于新样本到来时的增量式更新
相比较梯度上升，没有矩阵转换过程

Args:
    dataMatrix: 样本点
    classLabels: 标签

Returns:
    weights: 回归系数
'''
def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    # array([1, 1, ..]),注意和ones((1,n))的区别，后者array([[1, 1, ..]])
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

'''
改进的随机梯度上升算法

Args:
    dataMatrix: 样本点
    classLabels: 标签
    num: 迭代次数

Returns:
    weights: 回归系数
'''
def stocGradAscent1(dataMatrix, classLabels, num=150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(num):
        dataIndex = list(range(m))
        for i in range(m):
            # alpha每次迭代时更新
            alpha=4/(1.0+j+i) + 0.01
            # 随机选取样本来更新回归系数
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

'''
从疝气病症预测病马的死亡率
处理数据中的缺失值：
1. 采用0替换所有缺失值
    weights = weights + alpha * error * dataMatrix[randIndex]
    dataMatrix[randIndex] = 0, 特征系数不会更新
2. 类别标签丢失时，丢弃该条数据
'''

'''
现已有预处理后的数据集

Args:
    inX: 特征向量
    weights: 回归系数

Returns:
    1 or 0
'''
def classifyVector(inX, weights):
    p = sigmoid(sum(inX * weights))
    if p > 0.5:
        return 1.0
    else:
        return 0.0

'''
打开测试集和训练集，并对数据进行格式化处理

Returns:
    errorRate: 错误率
'''
def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    error = 0
    num = 0.0
    for line in frTest.readlines():
        num += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            error += 1
    errorRate = error/num
    print ("the error rate of this test is :%f" % errorRate)
    return errorRate

'''
调用colicTest()10次并求平均值
'''
def multiTest():
    num = 10
    errorSum = 0.0
    for k in range(num):
        errorSum += colicTest()
    print ("after %d iterations the average error rate is: %f" % (num,errorSum/num))

if __name__ == '__main__':
    dataArr, labelMat = loadDataSet()
    # print(gradAscent(dataArr, labelMat)) 
    # weights = gradAscent(dataArr, labelMat)
    # plotBestFit(weights.getA())
    # weights = stocGradAscent1(array(dataArr), labelMat)
    # plotBestFit(weights)
    multiTest()










