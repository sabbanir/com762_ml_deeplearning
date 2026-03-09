from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# load dataset
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
df = read_csv(path, header=None)

# split input and output
X, y = df.values[:, :-1], df.values[:, -1]

# convert to float
X = X.astype('float32')

# encode labels
y = LabelEncoder().fit_transform(y)

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# number of input features
n_features = X_train.shape[1]

# define model
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(n_features,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

# compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train model
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)

# evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy:", acc)

# make prediction
row2 = np.array([[5.1, 3.5, 1.4, 0.2]])

yhat = model.predict(row2)
print("Predicted probabilities:", yhat)
print("Predicted class:", argmax(yhat))