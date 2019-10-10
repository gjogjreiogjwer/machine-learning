# -*- coding: utf-8 -*-
'''
利用K-均值聚类算法对未标注数据分组

聚类是一种无监督的学习，它将相似的对象归到同一个簇中。
K-均值：发现k个不同的簇，且每个簇的中心采用簇中所含值的均值计算而成。

创建k个点作为起始质心（随机选择）
当任意一个点的簇分配结果发生改变时
    对数据集中的每个数据点
        对每个质心
            计算质心到数据点的距离
        将数据点分配搭配距其最近的簇
    对每一个簇，计算簇中所有点的均值并将均值作为质心
'''

from numpy import *

'''
将文本文件导入到一个列表中

Args:
    fileName: 文件名

Returns:
    dataMat: 数据集
'''
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat

'''
计算两个向量的欧式距离

Args:
    vecA: 向量A
    vecB: 向量B

Returns:
    sqrt(sum(power(vecA-vecB, 2))): 距离
'''
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA-vecB, 2)))

'''
创建一个包含k个随机质心的集合

Args:
    dataSet: 数据集
    k: 质心个数

Returns:
    centroids: 质心
'''
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ*random.rand(k,1)
    return centroids

'''
K-均值算法

Args:
    datSet: 数据集
    k: 簇的数目
    distMeas: 计算距离
    createCent: 创建初始质心

Returns:
    centroids: 簇
    clusterAssment: 每个点的簇分配结果
'''
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    # 第一列簇索引值，第二列点到质心距离
    clusterAssment = mat(zeros((m,2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                dist = distMeas(centroids[j,:], dataSet[i,:])
                if dist < minDist:
                    minDist = dist
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex, minDist**2
        print (centroids)
        for cent in range(k):
            # 提取出归于cent簇的数据
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            if len(ptsInClust) != 0:
                # 更新质心
                centroids[cent,:] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment

'''
使用后处理来提高聚类性能
度量聚类效果的指标：SSE，对应于clusterAssment第二列之和
SSE越小表示数据点越接近质心，聚类效果也越好
一种降低SSE的方法：
    将具有最大SSE值的簇划分为两个簇：将最大簇包含的点过滤出来并在这些点上运行K-均值算法，k为2
    为了保持簇总数不变，可以讲某两个簇合并：
        1.合并最近的质心
        2.合并两个使得SSE增幅最小的质心

二分K-均值算法：
将所有点看成一个簇
当簇数目小于k时
    对于每一个簇
        计算总误差
        在给定的簇上进行K-均值聚类(k=2)
        计算将该簇一分为二后的总误差
    选择使得误差最小的簇进行划分操作

Args:
    datSet: 数据集
    k: 簇的数目
    distMeas: 计算距离

Returns:
    centList: 簇
    clusterAssment: 每个点的簇分配结果
'''
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    # 把所有点看成一个簇
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    # 存储所有质心
    centList = [centroid0]
    # 更新所有点到初始质心的距离
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:]) ** 2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            # i簇中的所有点
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])
            sseNoSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print ("sseSplit, sseNoSplit ", sseSplit, sseNoSplit)
            # sseSplit这些误差和剩余数据集的误差之和作为本次划分误差
            if(sseSplit + sseNoSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseNoSplit + sseSplit
        # k为2的聚类划分产生两个簇0和1，0归为原来的簇，1归为新的簇
        bestClustAss[nonzero(bestClustAss[:,0].A==1)[0],0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A==0)[0],0] = bestCentToSplit
        print ('the bestCentToSplit is ', bestCentToSplit)
        print ('the len of bestClustAss is ', len(bestClustAss))
        # bestNewCents[0,:]归为原来的簇，bestNewCents[1,:]归为新的簇
        centList[bestCentToSplit] = bestNewCents[0,:]
        centList.append(bestNewCents[1,:])
        # 对本次划分的簇进行更新
        clusterAssment[nonzero(clusterAssment[:,0].A==bestCentToSplit)[0],:]=bestClustAss
    return centList, clusterAssment
                                                       
if __name__ == '__main__':
    datMat = mat(loadDataSet('testSet.txt'))
    # print (min(datMat[:,0]))
    # print (randCent(datMat, 2))
    # print (distEclud(datMat[0], datMat[1]))
    #myCentroids, clustAssing = kMeans(datMat, 4)
    datMat3 = mat(loadDataSet('testSet2.txt'))
    centList, myNewAssments = biKmeans(datMat3, 3)
    print (centList)






















































