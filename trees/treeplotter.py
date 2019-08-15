# -*- coding: utf-8 -*-
'''
使用Matplotlib绘制决策树
'''

import matplotlib.pyplot as plt

# 节点格式
decisionNode = dict(boxstyle = "sawtooth", fc = "0.8")
# 叶节点格式
leafNode = dict(boxstyle = "round4", fc="0.8")
# 箭头格式
arrow_args = dict(arrowstyle = "<-")

'''
绘制带箭头的注解.

Args:
    nodeTxt: 注解txt
    centerPt: 箭头尾部坐标
    parentPt: 箭头起始坐标
    nodeType: 节点类型
'''
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', 
        xytext=centerPt, textcoords='axes fraction', va="center", ha="center", 
        bbox=nodeType, arrowprops=arrow_args)

'''
绘制树节点.
'''
def createPlot0():
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    createPlot0.ax1 = plt.subplot(111,frameon=False)
    plotNode('decisionNode', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('leafNode', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

'''
获取叶节点个数。

Args:
    myTree: 树

Returns:
    numLeafs: 叶节点个数
'''
def getNumLeafs(myTree):
    numLeafs = 0
    first = list(myTree.keys())[0]
    second = myTree[first]
    for key in second:
        if type(second[key]).__name__=='dict':
            numLeafs += getNumLeafs(second[key])
        else:
            numLeafs += 1
    return numLeafs

'''
获取树的层数。

Args:
    myTree: 树

Returns:
    maxDepth: 层数
'''
def getTreeDepth(myTree):
    maxDepth = 0
    first = list(myTree.keys())[0]
    second = myTree[first]
    for key in second:
        if type(second[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(second[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

'''
树信息。

Args:
    i: index 

Returns:
    list[i]: 树
'''
def retrieveTree(i):
    list=[{'no surfacing':{0:'no',1:{'flippers':{0:'no',1:'yes'}}}},
          {'no surfacing':{0:'no',1:{'flippers':{0:{'head':{0:'no',1:'yes'}},1:'no'}}}}
          ]
    return list[i]

'''
在父子节点间填充信息。

Args:
    cntrPt: 子节点坐标
    parentPt: 父节点坐标
    txtString: 节点信息
'''
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0+cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0+cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

'''
Args:
    myTree: 树
    parentPt: 父节点坐标
    nodeTxt: 节点信息
'''
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    first = list(myTree.keys())[0]
    cntrPt = (plotTree.xoff + (1.0 + numLeafs)/2/plotTree.totalW, plotTree.yoff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(first, cntrPt, parentPt, decisionNode)
    second = myTree[first]
    plotTree.yoff = plotTree.yoff - 1/plotTree.totalD
    for key in second:
        if type(second[key]).__name__=='dict':
            plotTree(second[key], cntrPt, str(key))
        else:
            plotTree.xoff = plotTree.xoff + 1/plotTree.totalW
            plotNode(second[key], (plotTree.xoff, plotTree.yoff), cntrPt, leafNode)
            plotMidText((plotTree.xoff, plotTree.yoff), cntrPt, str(key))
    plotTree.yoff = plotTree.yoff + 1/plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xoff = -0.5/plotTree.totalW
    plotTree.yoff = 1.0
    plotTree(inTree, (0.5,1.0), '')
    plt.show()

if __name__ == '__main__':
    #createPlot0()
    myTree = retrieveTree(1)
    print (getNumLeafs(myTree))
    print (getTreeDepth(myTree))
    createPlot(myTree)









































