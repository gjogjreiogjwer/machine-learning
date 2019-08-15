# -*- coding: utf-8 -*-
'''
使用FP-growth算法来高效发现频繁项集

发现频繁项集过程：
    （1）构建FP树（扫描数据集两次，第一遍对所有元素项的出现次数计数，第二遍只考虑频繁元素）
    （2）从FP树中挖掘频繁项集
'''

class treeNode:
    '''
    构建FP树的数据结构

    Args:
        nameValue: 节点名字
        numOccur: 计算值
        parentNode: 父节点
    '''
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    '''
    Args:
        numOccur: 对count增加给定值
    '''
    def inc(self,numOccur):
        self.count += numOccur

    '''
    将树以文本形式显示
    '''
    def disp(self, ind=1):
        print ('  '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)

'''
构建FP树

Args:
    dataSet: 包含项集和出现频率的字典
    minSup: 最小支持度

Returns:
    retTree: FP树
    headerTable: 头指针表
'''
def createTree(dataSet, minSup=1):
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item,0) + dataSet[trans]
    for k in list(headerTable.keys()):
        # 移除不满足最小支持度的元素项
        if headerTable[k] < minSup:
            del(headerTable[k])
    freqItemSet = set(list(headerTable.keys()))
    # 如果没有元素项满足要求，则退出
    if len(freqItemSet) == 0:
        return None, None
    for k in headerTable:
        # 头指针表扩展，以便存储指向每种类型第一个元素项的指针
        headerTable[k] = [headerTable[k],None]
    retTree = treeNode('null set', 1, None)
    for tranSet, count in list(dataSet.items()):
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            # 按照值的大小，对键做降序
            orderedItems = [v[0] for v in sorted(list(localD.items()), key=lambda x:x[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable

'''
FP树的成长

Args:
    items: orderedItems
    inTree: FP树
    headerTable: 头指针表
    count: 计算值
'''
def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

'''
确保节点链接指向书中该元素项的每一个实例

Args:
    nodeToTest: headerTable[sth][1]
    targetNode: 节点
'''
def updateHeader(nodeToTest, targetNode):
    # 从头指针表的nodeLink开始，一直沿着nodeLink直到到达链表末尾
    while(nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

def loadsimpDat():
    simpDat = [['r','z','h','j','p'],
             ['z','y','x','w','v','u','t','s'],
             ['z'],
             ['r','x','n','o','s'],
             ['y','r','x','z','q','t','p'],
             ['y','z','x','e','q','s','t','m']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict
           
'''
从一颗FP树中挖掘频繁项集
（1）抽取条件模式基（以所查找元素项为结尾的路径集合）

指向该类型的第一个元素项

Args:
    leafNode: 节点
    prefixPath: 路径名
'''  
def ascendTree(leafNode,prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

'''
遍历链表，每遇到一个元素项都调用ascendTree上溯FP树，收集路径名称（条件模式基）
'''
def findPrefixPath(basePat, treeNode):
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

if __name__ == '__main__':
    rootNode = treeNode('pyramid', 9, None)
    rootNode.children['eye'] = treeNode('eye', 13, None)
    rootNode.children['phoenix'] = treeNode('phoenix', 3, None)
    rootNode.disp()
    simpDat = loadsimpDat()
    initSet = createInitSet(simpDat)
    myFPtree, myHeaderTab = createTree(initSet, 3)
    myFPtree.disp()
    print (findPrefixPath('x', myHeaderTab['x'][1]))










































































