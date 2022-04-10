X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

clf.fit(X,y)

y_pred = clf.predict(X)

print(y, y_pred)

"""## Iris data를 불러와서
### 1) train_test 데이터 분리
### 2) Decision tree 모델 만들고
### 3) train 데이터로 학습
### 4) test 데이터로 성능 검증
"""

import numpy as np
from sklearn.model_selection import train_test_split

# 데이터를 불러오기 : sklearn은 유방암 데이터를 dict 타입으로 제안합니다.
from sklearn.datasets import load_iris
iris = load_iris()

X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

print(y_train, y_test)

clf = DecisionTreeClassifier()

clf.fit(X_train,y_train)

from sklearn.metrics import accuracy_score

y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
