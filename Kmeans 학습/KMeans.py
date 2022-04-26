
"""
예측 모델 (입력 -> 출력)

지도 학습

### 비지도 학습 역할
- 데이터의 특징
- 군집화 
- 모델 생성 

데이터의 거리 = 데이터의 유사도

클러스트의 중심 = 데이터의 평균

클러스트 = 가장 가까운 거리의 데이터
"""

from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])

import matplotlib.pyplot as pit

# pit.scater(X[:, 0], X[:, 1])

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
kmeans.predict(X)

kmeans.predict([[0, 0], [12, 3]])

"""3개의 클러스터로 클러스터링을 하고

predict를 해서 실제 타겟과 비교
"""

from sklearn.datasets import load_iris

X = load_iris().data

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

y_pred = kmeans.predict(X)

kmeans.cluster_centers_

