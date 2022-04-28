# -*- coding: utf-8 -*-
"""지도/비지도 학습 정리.ipynb
# **지도학습과 비지도 학습**

머신러닝 알고리즘은 크게 지도학습과 비지도 학습으로 나눌 수 있다. 강화 학습도 있음.
---

## **지도 학습**
**지도 학습 알고리즘**은 훈련하기 위한 **데이터**와 **정답**이 필요하다.

지도학습에서는 데이터와 정답을 **입력**과 **타겟**이라고 하고, 이 둘을 합쳐 **훈련 데이터**라고 부른다.

입력으로 사용된 무언가는 **특성**이라고 한다. ex) 도미의 길이와 무게

### **지도 학습 역할**
- 데이터 분류
- 회귀

### **지도 학습의 알고리즘은 다음과 같다.**

- LinearRegression : 선형 회귀 (유일, 1차 방정식)
- KNeighborsClassifier 
- DecisionTreeClassifier : 결정 트리
- SVM



---
## **비지도 학습**
타겟이 없을 때 사용하는 머신러닝 알고리즘은 **비지도 학습**이다.


사람이 가르쳐 주지 않아도 데이터에 있는 무언가를 학습 하는 것이라고 생각하면 된다. 

=> 정답이 없고 데이터만 있는 것, 타겟이 없고 입력만이 있는 것

### **비지도 학습 역할**
- 데이터의 특징 => 입력 데이터만 있을때 데이터 특징들을 뽑아낸다.
- 군집화 => 비슷한것 끼리 묶기
- 모델 생성 => 데이터를 많이보며 모델 형성

### **비지도 학습의 알고리즘은 다음과 같다.**

- KMeans : 클러스트 
- PCA : 주성분 분석

# 지도 학습 코드
"""

# 어디서 했는지, 왜 했는지 기억이 나질 않는다..
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
data
data.DESCR
data.data[0]

"""### 1. breast cancer data 셋을 load 하기
### 2. train/test 데이터 나누기
### 3. LR/KND/DT/SVM 4개의 모델을 만들기
### 4. test 데이터로 검증해서 성능 확인
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. breast cancer data 셋을 load 하기
data = load_breast_cancer()
x = data.data # 입력
y = data.target # 타겟

# 2. train/test 데이터 나누기
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# LR
from sklearn.linear_model import LinearRegression

# LR 모델 만들기
line_fitter = LinearRegression()

# 학습
line_fitter.fit(x_train, y_train)

# 검증
y_pred = np.round(line_fitter.predict(x_test)) # LR => 1차식이기 때문에 실수가 나와야 한다. => np.round()를 통해 반올림.

# 성능 확인
accuracy_score(y_pred, y_test)

# K- Nearest Neighbor 
# - 서로 가까운 점들은 유사하다는 가정
# - 데이터셋에 내재된 패턴을 찾기 위해 데이터셋 전체를 봐야하지만, K-NN은 내 주변에 점만 확인하면 되기때문에 전체의 점을 볼 필요가 없음
# - K-NN은 특정 현상의 원인을 파악하는데 큰 도움을 주지는 않음 
# ex : 어떤 모델들은 나의 투표결과가 소득 수준이나 혼인 상태에 따라 판가름 된다고 알 수 있지만, K-NN은 왜 그렇게 결정되는지 알 수 없음 
# - 단순하고 빠르며, 효율적임 - K값이 클때는 overfitting의 우려가 있으며, K값이 작을 때는 구조 파악이 어려운 단점이 있음

# KN
from sklearn.neighbors import KNeighborsClassifier

# 이미 위에서 만든 데이터와 정답을 통해 수행

# 1. breast cancer data 셋을 load 하기
data = load_breast_cancer()
x = data.data # 입력
y = data.target # 타겟

# 2. train/test 데이터 나누기
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33)

# KN 모델 만들기
regressor = KNeighborsClassifier(n_neighbors=3)

# 학습
regressor.fit(x_train, y_train)

# 검증
y_pred = regressor.predict(x_test)

# 성능 확인
accuracy_score(y_test, y_pred)

# DT
from sklearn.tree import DecisionTreeClassifier

# 이미 위에서 만든 데이터와 정답을 통해 수행

# DT 모델 만들기
clf = DecisionTreeClassifier()

# 학습
clf.fit(x_train, y_train)

# 검증
y_pred = clf.predict(x_test)

# 성능 확인
accuracy_score(y_pred, y_test)

# SVC
from sklearn.svm import SVC

# 1. SVC model 만들기
clf = SVC()

# 2. model fit으로 학습
clf.fit(x_train,y_train)

# 3. model predict로 검증
y_pred = clf.predict(x_test)


# 4. 정확도 체크
accuracy_score(y_pred, y_test)

# 4가지 모델를 한번에 수행

mdl=[LinearRegression(),KNeighborsClassifier(),DecisionTreeClassifier(),SVC()]
res=[]
clf=[]
for model in mdl:
  # 각 모델 학습
  model.fit(x_train,y_train)
  
  # 각 모델 검증와 정확도 체크
  res.append(accuracy_score(np.round(model.predict(x_test)),y_test))  
  
  # 모델
  clf.append(model)


print(res)

"""# 비지도 학습 코드"""

# 클러스터링 -> 거리가 작을 수 록 비슷한 데이터다.
# 데이터의 거리 = 데이터의 유사도
# 클러스트의 중심 = 데이터의 평균
# 클러스트 = 가장 가까운 거리의 데이터
# 백터양자화

# k-means
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as pit


X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])


# pit.scatter(X[:, 0], X[:, 1])

kmeans = KMeans(n_clusters=2) #2개의 클러스터 찾기

# 학습
kmeans.fit(X) # y가 필요 없다

# 검증
kmeans.predict(X)


# 각 데이터의 대한 클러스터 결과물
kmeans.labels_   

kmeans.predict([[0, 0], [12, 3]])

# 3개의 클러스터로 클러스터링을 하고
# predict를 해서 실제 타겟과 비교

# 데이터 로드
X = load_iris().data

# 모델 만들기 (3개의 클러스터)
kmeans = KMeans(n_clusters=3)

# 학습
kmeans.fit(X) 

# 검증
k_pred=kmeans.predict(X)

# print(k_pred,"---",load_iris().target)  # 수가 중요한것이 아니다 비슷한 것 끼리 모인것
# print(abs(k_pred-load_iris().target))

# 클러스터 중심을 찾는 코드 => 데이터의 평균
kmeans.cluster_centers_

# pca의 축을 찾기 위한 이유 => 데이터가 가장 잘보이는 축으로 현재 축으로 바꿔주기 위해 == data가 잘보이는 축으로 바꿔주기 위해
# 데이터가 잘보인다 => 데이터가 흩어져있는 정도가 큰 축
# x가 현재 축에서 안보일떄 축을 변환시켜 잘보이도록한다.(행렬을 곱해서)
# 현재축을 다른축으로 바꾸는 이유 데이터를 가장 잘보이는 축으로 바꿔주는 것이 목적이다
# 데이터가 잘 보인다 데이터가 가장 넓게 흩어져 있어야 함 공분산으로 부터

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

a = np.array([[1,2],[3,4]])
np.linalg.eig(a) # => 고유 벡터 찾기

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
plt.scatter(X[:,0],X[:,1])

pca = PCA(n_components=2)   #n com-> 몇차원으로 볼것인가? 최대 2차원

# 학습
pca.fit(X)  # 비지도학습으로 y없음 

# 축 변환
X_pca=pca.transform(X)

# 그래프
plt.scatter(X_pca[:,0],X_pca[:,1])

print(X_pca)

pca = PCA(n_components=2)   #n com-> 몇차원으로 볼것인가? 최대 2차원
# 학습
pca.fit(X)  # 비지도학습으로 y없음 
print(X.shape)
print(pca.n_components_)

# 축 변환
X_pca=pca.transform(X)

print(X_pca)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)

kmeans = KMeans(n_clusters=2) #2개의 클러스터 찾기

# 학습
kmeans.fit(X_pca) # y가 필요 없다

# 검증
kmeans.predict(X_pca)

