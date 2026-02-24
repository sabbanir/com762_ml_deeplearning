import numpy as np
from keras import Input, Model
from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense

# load the dataset
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
df = read_csv(path, header=None)
# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
# ensure all data are floating point values
print(len(X))
X = X.astype('float32')
# encode strings to integer
y = LabelEncoder().fit_transform(y)
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
n_features = X_train.shape[1]


input_data = Input(shape=(n_features,))
hidden_layer1 = Dense(10, activation='relu')(input_data)
# hiddel_layerx = Dense(4, "relu")(hidden_layer1)
output_x = Dense(3, activation='softmax')(hidden_layer1)

model = Model (input_data,output_x)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("yvalues")
print(len(y_train))
model.fit(X_train, y_train, epochs = 100)
loss, acc = model.evaluate(X_test, y_test, verbose=0, batch_size=10)
print('Test Accuracy: %.3f' % acc)

print(n_features)
print(X)
print(y)
# model =
row2 = np.array([[5.1,3.5,1.4,0.2]])
yhat = model.predict([row2])
print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))