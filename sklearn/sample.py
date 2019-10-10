'''
官方文档：https://scikit-learn.org/stable/
'''

'''
鸢尾花识别是一个经典的机器学习分类问题，它的数据样本中包括了4个特征变量，1个类别变量，样本总数为150。

它的目标是为了根据花萼长度（sepal length）、花萼宽度（sepal width）、花瓣长度（petal length）、
花瓣宽度（petal width）这四个特征来识别出鸢尾花属于山鸢尾（iris-setosa）、
变色鸢尾（iris-versicolor）和维吉尼亚鸢尾（iris-virginica）中的哪一种。
'''
# 引入数据集
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# 将数据分为测试机和训练集
from sklearn.model_selection import train_test_split

# 利用kNN训练数据
from sklearn.neighbors import KNeighborsClassifier

# 导入鸢尾花数据
iris = datasets.load_iris()

# 特征变量,有四个
iris_X = iris.data
# print (iris_X)

# 目标值，分为3种
iris_Y = iris.target
# print (iris_Y)

# 利用train_test_split进行训练集和测试机区分，测试机占30%
X_train, X_test, Y_train, Y_test = train_test_split(iris_X, iris_Y, test_size=0.3)

# 标准化，先拟合数据再标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)

# 训练数据
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)

# params = knn.get_params()
# print (params)

score = knn.score(X_test, Y_test)
# print ('预测得分为', score)

# 预测数据特征值
predictions = knn.predict(X_test)
# print (predictions)






































