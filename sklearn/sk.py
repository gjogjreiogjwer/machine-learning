# 1. 获取数据*******************************************************************
# 1.1 导入sklearn数据集
# 导入datasets模块以使用sklearn的数据集
from sklearn import datasets
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
a = [['波士顿房价', 'load_boston()','回归','506*13'],
    ['鸢尾花', 'load_iris()','分类','150*4'],
    ['糖尿病', 'load_diabetes()','回归','442*10'],
    ['手写数字', 'load_digits()','分类','5620*64'],   ]
a = np.array(a)
df = pd.DataFrame(a, columns=['数据集名称', '调用方式', '适用算法', '数据规模'])
print (df)

# 导入数据集
iris = datasets.load_iris()
# 获得特征向量
X = iris.data
# 获得样本标签
Y = iris.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


# 1.1.1 手写数字数据集
digits = datasets.load_digits()
print (digits.data.shape)
print (digits.target.shape)
print (digits.images.shape)

# import matplotlib.pyplot as plt 
# plt.matshow(digits.images[0])
# plt.show()


# 2. 数据预处理*******************************************************************
from sklearn import preprocessing
# 2.1 z-score标准化 StandardScaler
# 标准化后，各特征0均值，单位方差

# scaler内存有均值和方差
scaler = preprocessing.StandardScaler().fit(X)

# 用scaler中的均值和方差来转换X，使X标准化
scaler.transform(X)
# 也要对测试集做同样的标准化


# 2.2 min-max标准化化
# 对原始数据进行线性变化，使结果落到[0,1]区间
scaler = preprocessing.MinMaxScaler(feature_range=(0,1)).fit(X_train)
scaler.transform(X_train)
scaler.transform(X_test)

# 2.3 正则化
'''
当你想要计算两个样本的相似度时必不可少的一个操作，就是正则化。
其思想是：首先求出样本的p范数，然后该样本的所有元素都要除以该范数，这样最终使得每个样本的范数都是1。
规范化（Normalization）是将不同变化范围的值映射到相同的固定范围，常见的是[0,1]，也成为归一化。

我们可以发现对于每一个样本都有0.4^2+0.4^2+0.81^2=1
这就是L2 norm，变换后每个样本的各维特征的平方和为1
类似的，L1 norm则是变换后每个样本的各维特征的绝对值之和为1
'''
x = [[1,-1,2],
    [2,0,0],
    [0,1,-1]]
x_normalized = preprocessing.normalize(x, norm='l2')
print (x_normalized)



# 3. 数据集拆分*******************************************************************
'''
Args:
    test_size: 样本占比
    random_state: 随机数的种子

Returns:
    X_train: 划分出的训练集数据
    X_test: 划分出的测试集数据
    Y_train: 划分出的训练集标签
    Y_test: 划分出的测试集标签
'''
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)



# 4. 模型定义*******************************************************************
# 4.1 线性回归
from sklearn.linear_model import LinearRegression
'''
Args:
    fit_intercept: 是否计算截距
    normalize: 当fit_intercept为True时，该参数才存在。normalize为True时，回归系数通过减去平均值并除以l2范数归一化
    n_jobs: 指定线程数
'''
model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)

# 4.2 逻辑回归
from sklearn.linear_model import LogisticRegression
'''
Args:
    penalty='l2' : 字符串‘l1’或‘l2’,默认‘l2’。
        用来指定惩罚的基准（正则化参数）。只有‘l2’支持‘newton-cg’、‘sag’和‘lbfgs’这三种算法。
        如果选择‘l2’，solver参数可以选择‘liblinear’、‘newton-cg’、‘sag’和‘lbfgs’这四种算法；
        如果选择‘l1’的话就只能用‘liblinear’算法。
    dual=False : 对偶或者原始方法。Dual只适用于正则化相为l2的‘liblinear’的情况，通常样本数大于特征数的情况下，默认为False。
    C=1.0 : C为正则化系数λ的倒数，必须为正数，默认为1。和SVM中的C一样，值越小，代表正则化越强。
    fit_intercept=True : 是否存在截距，默认存在。
    intercept_scaling=1 : 仅在正则化项为‘liblinear’，且fit_intercept设置为True时有用。
    solver='liblinear' : solver参数决定了我们对逻辑回归损失函数的优化方法，有四种算法可以选择。
        a) liblinear：使用了开源的liblinear库实现，内部使用了坐标轴下降法来迭代优化损失函数。
        b) lbfgs：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
        c) newton-cg：也是牛顿法家族的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
        d) sag：即随机平均梯度下降，是梯度下降法的变种，和普通梯度下降法的区别是每次迭代仅仅用一部分的样本来计算梯度，适合于样本数据多的时候。
        从上面的描述可以看出，newton-cg、lbfgs和sag这三种优化算法时都需要损失函数的一阶或者二阶连续导数，
        因此不能用于没有连续导数的L1正则化，只能用于L2正则化。而liblinear通吃L1正则化和L2正则化。
        同时，sag每次仅仅使用了部分样本进行梯度迭代，所以当样本量少的时候不要选择它，而如果样本量非常大，
        比如大于10万，sag是第一选择。但是sag不能用于L1正则化，所以当你有大量的样本，又需要L1正则化的话就要自己做取舍了。
        要么通过对样本采样来降低样本量，要么回到L2正则化。但是liblinear也有自己的弱点！
        我们知道，逻辑回归有二元逻辑回归和多元逻辑回归。对于多元逻辑回归常见的有one-vs-rest(OvR)和many-vs-many(MvM)两种。
        而MvM一般比OvR分类相对准确一些。而liblinear只支持OvR，不支持MvM，这样如果我们需要相对精确的多元逻辑回归时，
        就不能选择liblinear了。也意味着如果我们需要相对精确的多元逻辑回归不能使用L1正则化了。
        维度<10000时，选择"lbfgs"法，维度>10000时，选择"cs"法比较好，显卡计算的时候，lbfgs"和"cs"都比"seg"快
        正则化  算法               适用场景 
        L1  liblinear   liblinear适用于小数据集；如果选择L2正则化发现还是过拟合，即预测效果差的时候，
                        就可以考虑L1正则化；如果模型的特征非常多，希望一些不重要的特征系数归零，从而让模型系数稀疏化的话，
                        也可以使用L1正则化。
        L2  liblinear   libniear只支持多元逻辑回归的OvR，不支持MvM，但MVM相对精确。
        L2  lbfgs/newton-cg/sag 较大数据集，支持one-vs-rest(OvR)和many-vs-many(MvM)两种多元逻辑回归。
        L2  sag 如果样本量非常大，比如大于10万，sag是第一选择；但不能用于L1正则化。
    multi_class='ovr' : 分类方式
        ovr即one-vs-rest(OvR)，multinomial是many-vs-many(MvM)。如果是二元逻辑回归，ovr和multinomial并没有任何区别，
        区别主要在多元逻辑回归上。ovr不论是几元回归，都当成二元回归来处理。mvm从从多个类中每次选两个类进行二元回归。
        如果总共有T类，需要T(T-1)/2次分类。OvR相对简单，但分类效果相对略差（大多数样本分布情况）。而MvM分类相对精确，
        但是分类速度没有OvR快。如果选择了ovr，则4种损失函数的优化方法liblinear，newton-cg,lbfgs和sag都可以选择。
        但是如果选择了multinomial,则只能选择newton-cg, lbfgs和sag了。
    class_weight=None : 类型权重参数。用于标示分类模型中各种类型的权重。默认不输入，即所有的分类的权重一样。
        选择‘balanced’自动根据y值计算类型权重。
        自己设置权重，格式：{class_label: weight}。例如0,1分类的er'yuan二元模型，设置class_weight={0:0.9, 1:0.1}，
        这样类型0的权重为90%，而类型1的权重为10%。
    random_state=None : 随机数种子，默认为无。仅在正则化优化算法为sag,liblinear时有用。
    max_iter=100 : 算法收敛的最大迭代次数。
    tol=0.0001 : 迭代终止判据的误差范围。
    verbose=0 : 日志冗长度int：冗长度；0：不输出训练过程；1：偶尔输出； >1：对每个子模型都输出
    warm_start=False : 是否热启动，如果是，则下一次训练是以追加树的形式进行（重新使用上一次的调用作为初始化）。布尔型，默认False。
    n_jobs=1 : 并行数，int：个数；-1：跟CPU核数一致；1:默认值。
'''
model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0,
    fit_intercept=True, intercept_scaling=1, class_weight=None,
    random_state=None, solver='liblinear', max_iter=100, multi_class='ovr',
    verbose=0, warm_start=False, n_jobs=1)


# 4.3 朴素贝叶斯算法
from sklearn import naive_bayes
'''
文本分类问题常用MultinomialNB
Args:
    alpha：平滑参数
    fit_prior：是否要学习类的先验概率；false-使用统一的先验概率
    class_prior: 是否指定类的先验概率；若指定则不能根据参数调整
    binarize: 二值化的阈值，若为None，则假设输入由二进制向量组成
'''
model = naive_bayes.GaussianNB() # 高斯贝叶斯
model = naive_bayes.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
model = naive_bayes.BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)

# 4.4 决策树
from sklearn import tree
'''
Args:
    criterion ：特征选择准则gini/entropy
    max_depth：树的最大深度，None-尽量下分
    min_samples_split：分裂内部节点，所需要的最小样本树
    min_samples_leaf：叶子节点所需要的最小样本数
    max_features: 寻找最优分割点时的最大特征数
    max_leaf_nodes：优先增长到最大叶子节点数
    min_impurity_decrease：如果这种分离导致杂质的减少大于或等于这个值，则节点将被拆分。
'''
model = tree.DecisionTreeClassifier(criterion='gini', max_depth=None,
    min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
    max_features=None, random_state=None, max_leaf_nodes=None,
    min_impurity_decrease=0.0, min_impurity_split=None,
     class_weight=None, presort=False)


# 4.5 支持向量机
from sklearn.svm import SVC
'''
Args:
    C：误差项的惩罚参数C
    gamma: 核相关系数。浮点数，If gamma is ‘auto’ then 1/n_features will be used instead.
'''
model = SVC(C=1.0, kernel='rbf', gamma='auto')


# 4.6 kNN
from sklearn import neighbors
'''
Args:
    n_neighbors： 使用邻居的数目
    n_jobs：并行任务数
'''
model = neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=1) # 分类
model = neighbors.KNeighborsRegressor(n_neighbors=5, n_jobs=1) # 回归

# 4.7 多层感知器（神经网络）
from sklearn.neural_network import MLPClassifier
'''
Args:
    hidden_layer_sizes: 元祖
    activation：激活函数
    solver ：优化算法{‘lbfgs’, ‘sgd’, ‘adam’}
    alpha：L2惩罚(正则化项)参数。
'''
model = MLPClassifier(activation='relu', solver='adam', alpha=0.0001)





















