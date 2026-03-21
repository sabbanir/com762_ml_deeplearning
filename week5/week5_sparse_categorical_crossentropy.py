from keras.src.saving import load_model
from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
# load the dataset
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
df = read_csv(path, header=None)
# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
# ensure all data are floating point values
X = X.astype('float32')
# encode strings to integer
encoder = LabelEncoder()
y = encoder.fit_transform(y)
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# determine the number of input features
n_features = X_train.shape[1]

model = Sequential()
model.add(Dense(20, activation='relu', kernel_initializer='he_normal',input_shape=(n_features,)))
model.add(Dense(10, activation='tanh'))
model.add(Dense(8,activation='sigmoid'))

model.add(Dense(3,activation ='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=10)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)
predict_x = np.array([[5.1,3.5,1.4,0.2]])

yhat = model.predict(predict_x)
print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))

model.save("model.h5")

newmodel= load_model("model.h5")
newmodel.predict(predict_x)
print(encoder.classes_)
p_class = encoder.inverse_transform([argmax(yhat)])[0]
print("Predicted class :", p_class)



