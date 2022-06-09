# -*- coding: utf-8 -*-
"""
### 목적에 따라 활성화 함수가 다르다.

- 회귀를 하고 싶을때는 Linear를 사용하면된다.
- 맨끝 출력층에 말고 중간층에 Linear를 쓰는경우는 많지않다.
    
    사실 딥러닝은 데이터가 비선형인 문제를 해결하려고 사용한다. 선형 문제를 해결하려고 딥러닝을 거의 사용하지않는다. 층을 깊게 쌓을 이유가 없다.
    
    → 선형회귀로도 충분히 해결 가능하기 때문에 많은 cost를 소비하면서 딥러닝을 쓰지는 않는다.
    
- 확률값을 출력하고 싶으면 다중분류인경우 SoftMax, 이진분류인경우 Sigmoid함수를 사용한다.
- 보통은 중간층에 ReLU를 사용한다.(가장 많이쓰고 최적화가 잘되어있다.)
"""

from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA 

iris = load_iris()

X, y = iris.data, iris.target


train_x, test_x, train_target, test_target = train_test_split(X, y, test_size=0.3)

pca = PCA(n_components=2)
pca.fit(train_x)
new_train = pca.transform(train_x)
new_test = pca.transform(test_x)
print(new_train.shape, new_test.shape)

"""activation => loss

Linear => mse

sigmoid => binary_crossentropy

softmax => sparse_categorical_crossentropy
"""

from tensorflow.keras import models, layers
import numpy as np


# model = models.Sequential()
# model.add(layers.Dense(units=1, activation='softmax', input_shape=(2,)))
 
# sigmoid => binary_crossentropy => [ 손실 : -60.4836540222168, 정확도 : 0.6666666865348816] mse => [0.3333333432674408, 0.6666666865348816]
# relu => binary_crossentropy => [손실 : -5 , 정확도 : 0.666666] [-5.0830793380737305, 0.6666666865348816] mse => [0.0439104363322258, 0.6666666865348816]

# softmax => sparse_categorical_crossentropy = [0.07778985798358917, 0.9777777791023254]
model=models.Sequential()
model.add(layers.Dense(units=7,activation='relu',input_shape=(2,))) #units는 최소 3개 이상 (because,다중 퍼셉트론은 입력층,출력층,히든층)
# model.add(layers.Dense(units=7,activation='relu')) #units는 최소 3개 이상 (because,다중 퍼셉트론은 입력층,출력층,히든층)
# model.add(layers.Dense(units=7,activation='relu')) #units는 최소 3개 이상 (because,다중 퍼셉트론은 입력층,출력층,히든층)
# model.add(layers.Dense(units=7,activation='relu')) #units는 최소 3개 이상 (because,다중 퍼셉트론은 입력층,출력층,히든층)
# model.add(layers.Dense(units=7,activation='relu')) #units는 최소 3개 이상 (because,다중 퍼셉트론은 입력층,출력층,히든층)
# model.add(layers.Dense(units=7,activation='relu')) #units는 최소 3개 이상 (because,다중 퍼셉트론은 입력층,출력층,히든층)
# model.add(layers.Dense(units=7,activation='relu')) #units는 최소 3개 이상 (because,다중 퍼셉트론은 입력층,출력층,히든층)
# model.add(layers.Dense(units=7,activation='relu')) #units는 최소 3개 이상 (because,다중 퍼셉트론은 입력층,출력층,히든층)
# model.add(layers.Dense(units=7,activation='relu')) #units는 최소 3개 이상 (because,다중 퍼셉트론은 입력층,출력층,히든층)
# model.add(layers.Dense(units=7,activation='relu')) #units는 최소 3개 이상 (because,다중 퍼셉트론은 입력층,출력층,히든층)
# model.add(layers.Dense(units=7,activation='relu')) #units는 최소 3개 이상 (because,다중 퍼셉트론은 입력층,출력층,히든층)
model.add(layers.Dense(units=3,activation='softmax')) #units는 최소 3개 이상 (because,다중 퍼셉트론은 입력층,출력층,히든층)


# model.add(layers.Dense(units=3,activation='softmax', input_shape=(2,))) #units는 최소 3개 이상 (because,다중 퍼셉트론은 입력층,출력층,히든층)


###
# model.compile(optimizer = 'adam',  # 최적화 알고리즘 적용(찾아보면 엄청 많으니까 "딥러닝 최적화 알고리즘"이라는 키워드로 찾아보는 것도 좋다)
#               loss = 'categorical_crossentropy', # loss도 우리가 정의해줄수 있는데 우리가 풀려고 하는 문제(회귀,분류 등)에 따라 달라지니 궁금하면 찾아보자 
#               metrics = ['accuracy']) # 우리가 볼 지표, 이것도 회귀문제냐 분류문제냐에 따라 다르다 여기에서는 분류 문제이기때문에 Accuracy를 사용한다
###
# 훈련 과정에서 층에 있는 일부 뉴런을 랜덤하게 꺼서(즉, 뉴런의 출력을 0으로 만들어) 과대 적합을 만듣 규제 기법



model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics= ['acc'])

model.fit(new_train, train_target, epochs=1000, batch_size=1, verbose=1)

model.evaluate(new_test, test_target, verbose = 2) # 모델 평가

model.predict(new_test)



