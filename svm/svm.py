# -*- coding: utf-8 -*-
'''
支持向量机-序列最小化SMO

线性可分数据：画出一条直线可将两组数据分开
分割超平面：将数据集分开的直线，针对于二维
超平面：针对于多维
间隔：支持向量对应的样本点到决策面距离的两倍   W=2d，求解W的最大化，即d的最大化
d:点到线的距离
d=|w的转置*A+b|/||w||

找到离分割超平面最近的点，确保它们离分割面距离尽可能远
支持向量：离分割超平面最近的那些点，最大化支持向量到分割面的距离

SMO:求出一系列的alpha和b，计算出权重向量w并得到分割超平面。
每次循环中选择两个alpha进行优化处理，并固定其它参数。一旦找到一对合适的alpha，增大其中一个同时减小另一个。
这两个alpha必须在间隔边界之外，并且还没有进行过区间化或者不在边界上。
KKT条件违背程度越大，则变量更新后目标函数数值增幅越大。因此，第一个alpha选择违背KKT条件程度最大的，
第二个alpha采用启发式：使选取的两变量所对应样本之间的间隔最大
'''

from numpy import *

'''
便利函数，打开文件并逐行读取

Args:
    fileName: 文件名

Returns:
    dataMat: 样本集
    labelMat: 标签
'''
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    for line in open(fileName):
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

'''
取第二个alpha下标

Args:
    i: 第一个alpha的下标
    m: 所有alpha的数目

Returns:
    j: 第二个alpha的下标
'''
def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j

'''
调整大于H或小于L的alpha值

Args:
    aj:alpha值
    H:上限
    L:下限

Returns:
    aj:alpha值
'''
def clipAlpha(aj, H, L):
    if aj > H:
        aj=H
    if aj < L:
        aj = L
    return aj

'''
简化版SMO算法

Args:
    dataMatIn: 数据集
    classLabels: 标签
    C: 松弛变量
    toler: 容错率
    maxIter: 退出前最大的循环次数
'''
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m, n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))
    # 没有任何alpha改变的情况下遍历数据集的次数
    iter = 0
    while(iter < maxIter):
        # 记录alpha是否已经优化
        alphaChange = 0
        for i in range(m):
            # 预测的类别
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix*dataMatrix[i,:].T)) + b
            # 误差
            Ei = fXi-float(labelMat[i])
            if((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphasIold = alphas[i].copy()
                alphasJold = alphas[j].copy()
                # 计算上下边界
                if(labelMat[i] != labelMat[j]):
                    l = max(0, alphas[j]-alphas[i])
                    h = min(C, C+alphas[j]-alphas[i])
                else:
                    l = max(0, alphas[j]+alphas[i]-C)
                    h = min(C, alphas[j]+alphas[i])
                if l == h:
                    print ("l==h")
                    continue
                # 最优修改量
                eta = 2.0*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0:
                    print ("eta>=0")
                    continue
                # 更新aj
                alphas[j] -= labelMat[j]*(Ei-Ej)/eta
                alphas[j] = clipAlpha(alphas[j], h, l)
                if(abs(alphas[j]-alphasJold) < 0.00001):
                    print ("j not moving enough")
                    continue
                # 更新ai
                alphas[i] += labelMat[j]*labelMat[i]*(alphasJold-alphas[j])
                b1 = b-Ei-labelMat[i]*(alphas[i]-alphasIold)*dataMatrix[i,:]*dataMatrix[i,:].T-labelMat[j]*(alphas[j]-alphasJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b-Ej-labelMat[i]*(alphas[i]-alphasIold)*dataMatrix[i,:]*dataMatrix[j,:].T-labelMat[j]*(alphas[j]-alphasJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                # 更新b
                if(0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif(0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1+b2)/2
                alphaChange += 1
                print ("iter: %d, i: %d, pairs changed %d" % (iter,i,alphaChange))
        if(alphaChange == 0):
            iter += 1
        else:
            iter = 0
        print ("iteration number: %d" % iter)
    return b, alphas

'''
完整Platt SMO算法
通过外循环选择第一个alpha，内循环选择第二个alpha
建立全局缓存eCache保存误差值，并从中选择是的步长或者说Ei-Ej最大的alpha值(启发式)
'''

'''
作为一个数据结构，除了增加一个m*2的矩阵成员变量eCache之外，其他和简化版SMO一模一样

Args:
    dataMatIn: 数据集
    classLabels: 标签
    C: 松弛变量
    toler: 容错率
    kTup: 包含核函数信息的元组
'''
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.x = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.b = 0
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        # 第一列给出的是eCache是否有效的标志位（意味着它已经计算好了），第二列给出的是实际的E值
        self.eCache = mat(zeros((self.m, 2)))
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.x, self.x[i,:], kTup)

'''
计算E值

Args:
    oS: optStruct的对象
    K: 第几个样本

Returns:
    Ek: 误差值
'''
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:,k] + oS.b)
    Ek = fXk - oS.labelMat[k]
    return Ek

'''
用于选择第二个alpha,保证在每次优化中采用最大步长
即max(|Ei-Ej|)

Args:
    oS: optStruct的对象
    i: 第一个alpha所对应索引值
    Ei: 第一个alpha的误差

Returns:
    j: 第二个alpha所对应索引值
    Ej: 第二个alpha的误差
'''
def selectJ(oS, i, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    # eCache标识有效并存储Ei
    oS.eCache[i] = [1,Ei]
    # 返回有效的eCache的索引值，.A为mat转数组
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if(len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i: 
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if(deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    # 如果这是第一次循环，随机选择一个alpha值
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

'''
计算误差并存入缓存中

Args:
    oS: optStruct的对象
    k: 第几个样本
'''
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

'''
与smoSimple几乎一样，但使用了自己的数据结构optSttruct，并且用selectJ()来选择第二个alpha和更新Ecache

Args:
    i: 第i个样本
    oS: optStruct的对象
'''
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if((oS.labelMat[i]*Ei<-oS.tol) and (oS.alphas[i]<oS.C)) or ((oS.labelMat[i]*Ei>oS.tol) and (oS.alphas[i]>0)):
        j, Ej = selectJ(oS, i, Ei)
        alphasIold = oS.alphas[i].copy()
        alphasJold = oS.alphas[j].copy()
        if(oS.labelMat[i] != oS.labelMat[j]):
            l = max(0, oS.alphas[j]-oS.alphas[i])
            h = min(oS.C, oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            l = max(0, oS.alphas[j]+oS.alphas[i]-oS.C)
            h = min(oS.C, oS.alphas[j]+oS.alphas[i])
        if l == h:
            print("l==h");return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
        if eta >= 0:
            print("eta>=0");return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], h, l)
        updateEk(oS, j)
        if(abs(oS.alphas[j]-alphasJold) < 0.00001):
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphasJold-oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i]-alphasIold) * oS.K[i,i] - oS.labelMat[j] * (oS.alphas[j] - alphasJold) * oS.K[i,j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i]-alphasIold) * oS.K[i,j] - oS.labelMat[j] * (oS.alphas[j] - alphasJold) * oS.K[j,j]
        if(0 < oS.alphas[i]) and (oS.alphas[i] < oS.C):
            oS.b = b1
        elif(0 < oS.alphas[j]) and (oS.alphas[j]<oS.C):
            oS.b = b2
        else:
            oS.b = (b1+b2)/2
        return 1
    else:
        return 0

'''
完整版的Platt SMO算法

Args:
    dataMatIn: 数据集
    classLabels: 标签
    C: 松弛变量
    toler: 容错率
    maxIter: 退出前最大的循环次数
'''
def smoP(dataMatIn, classLabels, C, toler, maxIter, ktup=('lin', 0)):
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, ktup)
    iter = 0
    entireSet = True
    alphaChange = 0
    while (iter < maxIter) and ((alphaChange > 0) or (entireSet)):
        alphaChange = 0
        if entireSet:
            for i in range(oS.m):
                alphaChange += innerL(i, oS)
                print("fullset,iter:%d,i:%d,pairs changed %d" % (iter, i, alphaChange))
            iter += 1
        else:
            nonBound = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBound:
                alphaChange += innerL(i, oS)
                print("non-bound,iter:%d,i:%d,pairs changed %d" % (iter, i, alphaChange))
            iter += 1
        if entireSet:
            entireSet = False
        elif(alphaChange == 0):
            entireSet = True
        print("iteration number:%d" % iter)
    return oS.b, oS.alphas

'''
基于alpha计算w
'''
def calcWs(alphas, dataArr, classLabels):
    x = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m,n = shape(x)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],x[i,:].T)
    return w

'''
非线性SVM
利用核函数将数据映射到高维空间
本节采用径向基核函数

核转换函数
'''
def kernelTrans(X, A, kTup):
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K/(-1*kTup[1]**2))
    else:
        raise NameError('houston we have a problem that kernal is not recognized')
    return K

'''
利用核函数进行分类的径向基测试函数

Args: 
    k1: 用户定义变量
'''
def testRbf(k1=1.3):
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr,200, 0.0001, 10000, ('rbf',k1))
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A>0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print ("there are %d support vectors" % shape(sVs)[0])
    m,n = shape(dataMat)
    error = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i,:], ('rbf',k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            error += 1
    print ("the training error rate is:%f" % (error/m))
    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    error = 0
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m,n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i,:], ('rbf',k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            error += 1
    print ("the test error rate is:%f" % (error/m))

'''
手写识别系统回顾
使用svm保存的样本更少（只保存支持向量）

Args:
    fileName: 文件名

Returns:
    returnVect: 1*1024的Numpy数组
'''
def img2vector(fileName):
    returnVect = zeros((1,1024))
    fr = open(fileName)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

'''
数字9标签为-1
其他为1

Args:
    dirName: 文件夹名

Returns:
    trainingMat: 训练集
    hwLabels: 对应标签
'''
def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName,fileNameStr))
    return trainingMat, hwLabels

'''
基本与testRbf()相同，唯一区别调用loadImages()来获取数据

Args:
    k1: 用户定义参数
'''
def testDigits(k1=1.3):
    dataArr, labelArr = loadImages('trainingDigits')
    b, alphas = smoP(dataArr, labelArr,200, 0.0001, 10000, ('rbf',k1))
    print ('frefjref')
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A>0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print ("there are %d support vectors" % shape(sVs)[0])
    m,n = shape(dataMat)
    error = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i,:], ('rbf',k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            error += 1
    print ("the training error rate is:%f" % (error/m))
    dataArr, labelArr = loadImages('testDigits')
    error = 0
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m,n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i,:], ('rbf',k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            error += 1
    print ("the test error rate is:%f" % (error/m))



def testDigits1(kTup=('rbf', 10)):
    """
    测试函数
    Parameters:
        kTup - 包含核函数信息的元组
    Returns:
        无
    """
    dataArr,labelArr = loadImages('trainingDigits')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10, kTup)
    datMat = mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A>0)[0]
    sVs=datMat[svInd]
    labelSV = labelMat[svInd];
    print("支持向量个数:%d" % shape(sVs)[0])
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print("训练集错误率: %.2f%%" % (float(errorCount)/m))
    dataArr,labelArr = loadImages('testDigits')
    errorCount = 0
    datMat = mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1   
    print("测试集错误率: %.2f%%" % (float(errorCount)/m))




if __name__ == '__main__':
    dataArr, labelArr = loadDataSet('testSet.txt')
    #print (labelArr)
    #b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    #print (alphas[alphas>0])
    #b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
    #ws = calcWs(alphas, dataArr, labelArr)
    # 对数据进行分类
    datMat = mat(dataArr)
    # 对第一个数据点分类，值大于0属于1类，小于0属于-1类
    #print (datMat[0]*mat(ws)+b)
    #testRbf()
    testDigits()

















