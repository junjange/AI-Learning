from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from tensorflow.keras import models,layers
import numpy as np
X,y=load_iris().data,load_iris().target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33,)

model=models.Sequential()
model.add(layers.Dense(units=1,activation='linear',input_shape=(4,)))
model.compile(optimizer='sgd',loss='mse',metrics=['acc'])
model.fit(X_train,y_train,epochs=100,batch_size=1, verbose=1)



#테스트
Xtest_loss,Xtest_acc=model.evaluate(X_test,y_test)
print(Xtest_acc)
ytest=model.predict(X)
print(ytest)
