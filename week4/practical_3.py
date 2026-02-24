import numpy as np
from keras import Input, Model
from keras.src.layers import Dense
from pandas import read_csv
from sklearn.model_selection import train_test_split

# load the dataset
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df = read_csv(path, header=None)
# split into input and output columns
print(df.values[:, :-1])
X, y = df.values[:, :-1], df.values[:, -1]
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape, y_train.shape, y_test.shape)

n_features = X_train.shape[1]
input_x = Input(shape=(n_features,))
x = Dense(64, activation='relu')(input_x)
output_x = Dense(1, activation='relu')(x)

model = Model(input_x, output_x)
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=150)

error = model.evaluate(X_test, y_test, verbose=0,batch_size=20)

print(error)
print('MSE: %.3f, RMSE: %.3f' % (error, np.sqrt(error)))
# make a prediction

row2 = np.array([[0.00632, 18.00, 2.310, 0, 0.5380, 6.5750, 65.20, 4.0900, 1, 296.0, 15.30, 396.90, 4.98]])
yhat = model.predict([row2])
print('Predicted: %.3f' % yhat)
# Use the Functional model to develop a regression
