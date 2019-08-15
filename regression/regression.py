# -*- coding: utf-8 -*-
'''
预测数值型数据：回归
(1) 线性回归
(2) 局部平滑技术
(3) “欠拟合”下的缩减
'''

from numpy import *

'''
用线性回归找到最佳拟合直线
输入数据存放在矩阵X（每一个输入数据为一列向量）
回归系数存放在列向量w
预测结果Y1=X1.T * w
已知X和Y，求w
找出使误差最小的w，该误差为预测y值和真实y值之间的差值，采用平方误差表示：
(y[i] - x[i].T * w)的平方，即(Y - X * w).T * (Y - X * w)
对w求导，得到 2X.T * (Y -Xw),令其为零
w = (X.T * X)的-1次方 * X.T * Y
(X.T * X)的-1次方,也就是对矩阵求逆，该方程只在逆矩阵存在时适用
'''

'''
数据导入，打开一个用tab键分割的文本

Args:
    fileName: 文件名

Returns:
    dataMat: 输入X
    labelMat: 目标值Y
'''
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    numFeat = len(open(fileName).readline().split('\t')) - 1
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        currLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(currLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(currLine[-1]))
    return dataMat, labelMat

'''
计算最佳拟合直线

Args:
    xArr: 输入X
    yArr: 目标值Y

Returns:
    ws: 回归系数
'''
def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    # 判断行列式是否为0，为0则没有逆矩阵
    if linalg.det(xTx) == 0:
        print ("this matrix is singular")
        return
    ws = xTx.I * xMat.T * yMat
    return ws

'''
局部加权线性回归 LWLR
为防止出现欠拟合现象，引入一些偏差
给待测点附近的每个点赋予一定的权重S
w = (X.T * S * X).I * X.T * S * Y

Args:
    testPoint: 一个数据点
    xArr: 输入X
    yArr: 目标Y
    k: 控制衰减的速度

Returns:
    testPoint * ws: 估计值
'''
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye(m))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat * diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights*xMat)
    if linalg.det(xTx) == 0:
        print ("this matrix is sigular")
        return
    ws = xTx.I * (xMat.T * (weights*yMat))
    return testPoint * ws

'''
为每一个点调用lwlr()
'''
def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

'''
如果数据的特征比样本点还多(n>m)，就是说输入数据的矩阵X不是满秩矩阵，非满秩矩阵不能求逆
引入“岭回归”，缩减系数
w = (X.T * X + q * I).I * X.T * Y
I是单位矩阵
q是用户定义的数值
'''

'''
计算回归系数

Args:
    xMat: 输入X
    yMat: 目标Y
    lam: q

Returns:
    ws: 回归系数
'''
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + lam*eye(shape(xMat)[1])
    if linalg.det(denom) == 0:
        print ("this matrix is singular")
        return
    ws = denom.I * xMat.T * yMat
    return ws

'''
通过选取不同的q来重复测试过程，最终得到一个使预测误差最小的q

Args:
    xArr: 输入X
    yArr: 目标Y

Returns:
    wMat: 一组回归系数
'''
def ridgeTest(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    # 数据标准化
    yMean = mean(yMat)
    yMat = yMat - yMean
    xMean = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat-xMean)/xVar
    numTest = 30
    wMat = zeros((numTest,shape(xMat)[1]))
    for i in range(numTest):
        # exp(i-10)以指数级变化，可以看出i在取非常小的值和取非常大的值时分别对结果造成的影响
        ws = ridgeRegres(xMat, yMat, exp(i-10))
        wMat[i,:] = ws.T
    return wMat

if __name__ == '__main__':
    xArr, yArr = loadDataSet('ex0.txt')
    ws = standRegres(xArr, yArr)
    xMat = mat(xArr)
    yMat = mat(yArr)
    # 预测y值
    yHat = xMat * ws

    # # 绘制原始x，y点
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # # flatten(): 矩阵降到一维， .A: 转为数组
    # ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
    # # plt.show()

    # # 绘制yHat拟合直线
    # xCopy = xMat.copy()
    # xCopy.sort(0)
    # yHat = xCopy * ws
    # ax.plot(xCopy[:,1], yHat)
    # plt.show()

    # 判断模型的好坏，计算预测值yHat和真实值y的匹配程度，就是计算两个序列的相关系数
    yHat = xMat * ws
    # 相关系数为0.98
    #print (corrcoef(yHat.T, yMat))
    #print (lwlrTest(xArr, xArr, yArr, 0.003))
    abX, abY = loadDataSet('abalone.txt')
    ridgeWeights = ridgeTest(abX, abY)
    print (ridgeWeights)










