from keras import Input, Model
from numpy import argmax
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import numpy as np
# create the dataset
X, y = make_classification(n_samples=1000, n_features=4, n_classes=2,
random_state=1)
# determine the number of input features
n_features = X.shape[1]
print(X)
print(y)
print(n_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal',
input_shape=(n_features,)))

model.add(Dense(1, activation='sigmoid'))

# opt = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=10)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)


predict_x = np.array([[1.91518414, 1.14995454, -1.52847073,0.79430654]])
yhat = model.predict(predict_x)
print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))