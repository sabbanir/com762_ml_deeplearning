from keras import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import numpy as np
from numpy import asarray, argmax
from numpy import sqrt
train = [1, 2, 3, 4, 5, 6, 7]
windowSize, X_train, y_train = 3, [], []
for index in range(len(train)-windowSize):
    X_train.append(train[index:index+windowSize])
    y_train.append(train[index+windowSize])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = X_train.reshape((len(X_train), 3, 1))

print(X_train)
print(y_train)


model = Sequential()
#
model.add(SimpleRNN(20,input_shape=(3,1), return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(16))
model.add(Dense(8,activation = 'tanh'))
model.add(Dense(1,activation = 'linear'))
model.compile(optimizer='adam',loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=500)
model.evaluate(X_train, y_train,batch_size=10)

loss, acc = model.evaluate(X_train, y_train,batch_size=100)

print(loss)
print(acc)

predict_x = np.array([[2,3,4]])
# predict_x = np.array([[7,8,9]])
yhat = model.predict(predict_x)
print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))

