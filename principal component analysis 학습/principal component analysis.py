
# pca의 축을 찾기 위한 이유 => 데이터가 가장 잘보이는 축으로 현재 축으로 바꿔주기 위해 == data가 잘보이는 축으로 바꿔주기 위해
# 데이터가 잘보인다 => 데이터가 흩어져있는 정도가 큰 축


import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

import matplotlib.pyplot as plt

plt.scatter(X[:,0], X[:,1])


pca = PCA(n_components=2)
pca.fit(X) # 비지도 학습이라 y 없음.



X_pca = pca.transform(X) # 축 변경
plt.scatter(X_pca[:,0], X_pca[:,1])


print(X_pca)


