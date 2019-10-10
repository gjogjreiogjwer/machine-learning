from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# 投票分类器
# 导入数据
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 三个基学习器
log_clf = LogisticRegression()
rf_clf = RandomForestClassifier()
svm_clf = SVC()
# 投票分类器
voting_clf = VotingClassifier( estimators=[("lr", log_clf), ("rf", rf_clf), ("svc", svm_clf)], voting="hard" )
# voting_clf.fit( X_train, y_train )

for clf in ( log_clf, rf_clf, svm_clf, voting_clf ):
    clf.fit( X_train, y_train )
    y_pred = clf.predict( X_test )
    print( clf.__class__.__name__, accuracy_score(y_test, y_pred) )

'*****************************************************************************'
'''
bagging

n_estimators：基学习器的数量
max_samples：每个基学习器中的样本数，如果是整形，则就是样本个数；如果是float，则是样本个数占所有训练集样本个数的比例
bootstrap ：是否采用有放回抽样(bagging)，为True表示采用，否则为pasting。默认为True
n_jobs：并行运行的作业数量。-1时，个数为处理器核的个数
oob_socre：为True时，对模型进行out-of-bag的验证，即在一个基学习器中，没有用于训练的数据用于验证
'''
bag_clf = BaggingClassifier( DecisionTreeClassifier(), n_estimators=500, max_samples=100, bootstrap=True, n_jobs=-1 )

bag_clf.fit( X_train, y_train )
y_pred = bag_clf.predict( X_test )
pred_score = accuracy_score( y_pred, y_test )
print( pred_score )
