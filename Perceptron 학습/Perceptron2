import numpy as np

alpha = 0.1
training_endnum = 10000
W = np.random.random((1,2))
dataset_num = 4

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

T = np.array([0, 0, 0, 1])

for epochs in range(0,training_endnum) :
    print("----- %d th epoch -----" % (epochs+1))
    for j in range(0,dataset_num) :
        
        ## implement delta rule : w <- w + alpha * e * x 
        y = np.dot(W,np.transpose(X[j,:]))
        e = T[j]-y
        dW = alpha +0+np.transpose(X[j,:])
        W =  W+dW


for i in range(0,dataset_num) :
    y = np.dot(W,np.transpose(X[i,:]))
    print(y)


# ## 위의 코드를 활용하여, IRIS 데이터셋을 위의 Numpy 기반의 Perceptron로 예측하는 알고리즘을 구현하시오.



from sklearn.datasets import load_iris

import numpy as np
from sklearn.model_selection import train_test_split

X,y = load_iris().data, load_iris().target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3)


alpha=0.01
training_endnum = 10000
W = np.random.random((1,4))
dataset_num = 100

X=X_train
T=y_train


for epochs in range(0,training_endnum) :
    print("----- %d th epoch -----" % (epochs+1))
    for j in range(0,dataset_num) :
      y=np.dot(W,np.transpose(X[j,:]))
      e = T[j]-y
      dW = alpha *e*np.transpose(X[j,:])
      W =  W+dW

for i in range(0,dataset_num): 
  y = np.dot(W,np.transpose(X[i,:]))
  print(int(abs(np.round(y))))






