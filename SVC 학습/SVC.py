import numpy as np
from sklearn.svm import SVC

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])

# 1. model 만들기
clf = SVC()

# 2. model fit으로 학습
clf.fit(X, y)

# 3. model predict로 검증

clf.predict(X)


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

x, y = iris.data, iris.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 1. model 만들기
clf = SVC()

# 2. model fit으로 학습
clf.fit(x_train,y_train)

# 3. model predict로 검증
y_pred = clf.predict(x_test)

from sklearn.metrics import accuracy_score

# 정확도 체크
accuracy_score(y_pred, y_test)

