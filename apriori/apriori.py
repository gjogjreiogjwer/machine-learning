# -*- coding: utf-8 -*-
'''
使用Apriori算法进行关联分析

频繁项集：经常出现在一起的物品集合
支持度：数据集中包含该项集的记录所占比例
一条规则 P->H 的可信度定义为 support(P | H) /support(P)
尿布 --> 酒 的 可信度 = 支持度({尿布，酒}) / 支持度({尿布})

Apriori原理：如果某个项集是频繁的，那么它的所有子集也是频繁的。
    反过来：如果一个项集是非频繁集，那么它的所有超集也是非频繁的。
如果某条规则并不满足最小可信度要求，那么该规则的所有子集也不会满足最小可信度要求。
'''

'''
创建用于测试的数据集
'''
def loadDataSet():
    return [[1,3,4], [2,3,5], [1,2,3,5], [2,5]]

'''
创建一个集合，用于存储所有不重复的项值。

Args:
    dataSet: 数据集

Returns:
    map(frozenset, C1): 不重复的项值
'''
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                # 注意C1是一个集合的集合，方便后续操作
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))

'''
生成候选项集。

Args:
    D: 数据集                [set([1,3,4]), set([2,3,5]), set([1,2,3,5])]
    Ck: 候选项集列表          [frozenset([1]), frozenset([2]), frozenset([3]), frozenset([4]), frozenset([5])]
    minSupport: 最小支持度

Returns:
    retList: 候选项集
    supportData: 包含支持度的字典
'''
def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            # 如果can是tid的子集
            if can.issubset(tid):
                ssCnt[can] = ssCnt.get(can,0) + 1
    num = len(D)
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/num
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList, supportData

'''
创建更多候选项集Ck
输入 {0} {1} {2}
输出 {0,1} {0,2} {1,2}

Args:
    Lk: 频繁项集列表
    k: 项集元素个数

Returns:
    retList: 候选项集
'''
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            # k-2:减少重复次数。如果用{1,2},{0,1},{0,2}来创建三元素项集，先比较第一个元素是否相等，
            # 只对相等的做并集
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

'''
Apriori算法

Args:
    dataSet: 数据集
    minSupport: 支持度

Returns:
    L:候选项集的列表
    supportData: 每个项集的支持度字典
'''
def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while(len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

'''
从频繁项集中挖掘关联规则

Args:
    L: 频繁项集列表
    supportData: 包含频繁项集支持数据的字典
    minConf: 最小可信度

Returns:
    bigRuleList: 包含可信度的规则列表
'''
def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    # 因为无法从单元素项集中构建关联规则，所以要从包含两个或更多元素的项集开始规则构建过程
    for i in range(1,len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            # 如果频繁项集元素数目超过2，做进一步合并
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            # 如果只有两个元素，计算可信度
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

'''
计算可信度

Args:
    freqSet: {1,3}
    H: {1},{3}
    supportData: 包含频繁项集支持数据的字典
    br1: 包含可信度的规则列表
    minConf: 最小可信度

Returns:
    prunedH: 满足最小可信度要求的规则列表
'''
def calcConf(freqSet, H, supportData, br1, minConf=0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet-conseq]
        if conf >= minConf:
            print (freqSet-conseq, '--->', conseq, ' conf:', conf)
            br1.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

'''
为最初的项集生成更多的关联规则

Args:
    freqSet: {1,2,3}
    H: {1},{2},{3}
    supportData: 包含频繁项集支持数据的字典
    br1: 包含可信度的规则列表
    minConf: 最小可信度
'''
def rulesFromConseq(freqSet, H, supportData, br1, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m+1)):
        Hmp1 = aprioriGen(H, m+1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, br1, minConf)
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, br1, minConf)

if __name__ == '__main__':
    dataSet = loadDataSet()
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, suppData0 = scanD(D, C1, 0.5)
    print (L1[0])
    print (list(L1[0])[:0])
    L, supportData = apriori(dataSet)
    rules= generateRules(L, supportData)
    print (rules)





































