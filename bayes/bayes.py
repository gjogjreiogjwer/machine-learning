# -*- coding: utf-8 -*-
'''
基于概率论的分类方法：朴素贝叶斯

条件概率：
p(A|B) = p(A and B)/p(B)
p(B|A) = p(A and B)/p(A)
so p(A|B) * p(B) = p(B|A) * p(A)
   p(A|B) = p(B|A) * p(A) / p(B)

p(ci|x,y) = p(x,y|ci)*p(ci)/p(x,y)
给定某个由x，y表示的坐标点，那么该数据点来自类别ci的概率是多少？
如果 p(c1|x,y) > p(c2|x,y) , 那么属于类别c1

朴素的意义：假设每个特征相互独立。
p(a|X) = p(X|a)*p(a)/p(X) = p(x1,x2,x3,x4...,xn|a)*p(a)/p(x1,x2,x3,x4...,xn)
       = (p(x1|a)*p(x2|a)*p(x3|a)*..*p(xn|a))*p(a) / (p(x1)*p(x2)*..*p(xn))

有些情况下贝叶斯方法求出来的结果不好，考虑是不是条件独立假设的原因。
'''

from numpy import *

'''
创建样本集。

Returns:
    postingList: 进行词条切分后的文档集合
    classVec: 标签
'''
def loadDataSet():
    postingList = [['my','dog','has','flea','problems','help','please'],
                  ['maybe','not','take','him','to','dog','park','stupid'],
                  ['my','dalmation','is','so','cute','I','love','him'],
                  ['stop','posting','stupid','worthless','garbage'],
                  ['mr','licks','ate','my','steak','how','to','stop','him'],
                  ['quit','buying','worthless','dog','food','stupid']
                  ]
    classVec = [0,1,0,1,0,1]
    return postingList, classVec

'''
创建一个包含所有文档不重复词的列表。

Args:
    dataSet: 进行词条切分后的文档集合
Returns:
    list(returnVec): 不重复词条的列表
'''
def createVocabList(dataSet):
    returnVec = set([])
    for document in dataSet:
        returnVec = returnVec | set(document)
    return list(returnVec)

'''
得到一个文档向量，内容判断为是否出现在词汇表中。
词集模型，每个单词只能出现一次

Args:
	vocabList: 不重复词条的列表
	inputSet: 某个文档
Returns:
	returnVec: 文档向量，表示词汇表中单词在输入文档中是否出现
'''
def setWordVec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print ("the word:%s is not in my vocabulary" % word)
    return returnVec

'''
得到一个文档向量，内容判断为是否出现在词汇表中。
词袋模型，每个单词可以出现多次

Args:
	vocabList: 不重复词条的列表
	inputSet: 某个文档
Returns:
	returnVec: 文档向量，表示词汇表中单词在输入文档中是否出现
'''
def bagWordVec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
        	# 与词集模型的区别
            returnVec[vocablist.index(word)] += 1
    return returnVec

'''
训练算法：从词向量计算概率
判断一个留言板内的留言是否为侮辱性文档。
p(ci|W) = p(W|ci)*p(ci)/p(W)   W是一个词向量
在已知w词条的情况下，求为侮辱性文档c1的概率。

Args:
	trainMatrix: 文档矩阵
	trainCategory: 文档矩阵对应标签
Returns:
	p0Vect: 在已知为非侮辱性文档的情况下，各单词出现概率  p(W|c0)
	p1Vect: 在已知为侮辱性文档的情况下，各单词出现概率  p(W|c1)
	pAbusive: 侮辱性文档出现概率  p(c1)
'''
def train(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/numTrainDocs
    # 朴素贝叶斯需计算多个概率的乘积以获得文档属于某个类别的概率 
    # 将所有词出现数初始化为1，分母初始化为2，避免其中一个概率值为0，最后乘积也为0
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2
    p1Denom = 2
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
        	#向量
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 由于太多很小的数相乘，造成下溢出
    # 通过求对数避免下溢出或者浮点数舍入导致的错误
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive

'''
朴素贝叶斯分类函数

Args:
	vec: 测试词条
	p0Vec: 在已知为非侮辱性文档的情况下，各单词出现概率  p(W|c0)
	p1Vec: 在已知为侮辱性文档的情况下，各单词出现概率  p(W|c1)
	pClass1: 侮辱性文档出现概率  p(c1)

Returns:
	1 or 0: 是否为侮辱性文档
'''
def classify(vec, p0Vec, p1Vec, pClass1):
    p1 = sum(vec * p1Vec) + log(pClass1)
    p0 = sum(vec * p0Vec) + log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

'''
便利函数，用于测试
'''
def testing():
    listPost, listClass = loadDataSet()
    myVocabList = createVocabList(listPost)
    trainMat = []
    for post in listPost:
        trainMat.append(setWordVec(myVocabList, post))
    p0, p1, pa = train(trainMat,listClass)
    testEntry = ['love','my','dalmation']
    thisDoc = setWordVec(myVocabList, testEntry)
    print (testEntry, "classified as", classify(thisDoc,p0 , p1, pa))
    testEntry = ['stupid','garbage']
    thisDoc = setWordVec(myVocabList, testEntry)
    print (testEntry, 'classified as', classify(thisDoc, p0, p1, pa))

'''
使用朴素贝叶斯过滤垃圾邮件
准备数据：切分文本

Args:
	bigString: 表示一个文档所有内容的字符串

Returns:
	[tok.lower() for tok in listOfToken if len(tok)>2]: 按要求划分为字符串列表
'''
def testParse(bigString):
    import re
    listOfToken = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfToken if len(tok)>2]

'''
垃圾邮件测试
'''
def spamTest():
    docList = []
    classList = []
    # 导入并解析文本文件
    for i in range(1, 26):
        wordList = testParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        classList.append(1)
        wordList = testParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        classList.append(0)
    # 创建词汇表
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    # 挑取十封作为测试文件
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClass = []
    # 训练样本
    for doc in trainingSet:
        trainMat.append(setWordVec(vocabList, docList[doc]))
        trainClass.append(classList[doc])
    p0, p1, pa = train(trainMat, trainClass)
    errorCount = 0
    for doc in testSet:
        word = setWordVec(vocabList, docList[doc])
        if classify(word, p0, p1, pa) != classList[doc]:
            errorCount += 1
    print ('the error rate is:', errorCount/len(testSet))
    
if __name__ == '__main__':
    postingList, classVec = loadDataSet()
    myVocabList = createVocabList(postingList)
    print (setWordVec(myVocabList, postingList[0]))

    # trainMat = []
    # for post in postingList:
    # 	trainMat.append(setWordVec(myVocabList, post))
    # p0, p1, pAbusive = train(trainMat, classVec)
    # print (p0)

    testing()
    spamTest()




























