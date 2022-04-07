from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
data


data.DESCR


data.data[0]


# 1. breast cancer data 셋을 load 하기
# 2. train/test 데이터 나누기
# 3. LR/KND/DT/SVM 4개의 모델을 만들기
# 4. test 데이터로 검증해서 성능 확인


# 1. breast cancer data 셋을 load 하기

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np


data = load_breast_cancer()

x = data.data
y = data.target

# 2. train/test 데이터 나누기

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# LR

from sklearn.linear_model import LinearRegression


line_fitter = LinearRegression()
line_fitter.fit(x_train, y_train)
line_fitter
y_pred = np.round(line_fitter.predict(x_test))
accuracy_score(y_pred, y_test)


# KN
from sklearn.neighbors import KNeighborsRegressor

regressor = KNeighborsRegressor()
regressor.fit(x_train, y_train)
y_pred = np.round(regressor.predict(x_test)) 
accuracy_score(y_pred, y_test)

# DT

from sklearn.tree import DecisionTreeClassifier


clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy_score(y_pred, y_test)

# SVM

from sklearn.svm import SVC

# 1. model 만들기
clf = SVC()

# 2. model fit으로 학습
clf.fit(x_train,y_train)

# 3. model predict로 검증
y_pred = clf.predict(x_test)


# 정확도 체크
accuracy_score(y_pred, y_test)

