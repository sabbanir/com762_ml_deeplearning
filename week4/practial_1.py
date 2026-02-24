# Mean Squared Error (MSE)
from math import log

from keras import Input, Model
from keras.src import Functional

y_true = [0.000, 0.166, 0.333]
y_pred = [0.000, 0.254,0.998]

sum =0

for i in range(len(y_true)):
    sum = sum + (y_true[i] - y_pred[i])** 2
error = sum/len(y_true)
print("mean Squared Error"+str(error))



# Mean Absolute Error (MAE)
sum= 0
for i in range(len(y_true)):
    sum  = sum +(abs(y_true[i]) - abs(y_pred[i]))
error = sum/len(y_true)
print("Mean Absolute Error "+str(error))


##
import numpy as np
y_true = [[0,0,0,1], [0,0,0,1]]
y_pred = [[0.25,0.25,0.25,0.25], [0.01,0.01,0.01,0.96]]

cross_val =0

for i in range(len(y_true)):
    sum = 0
    for j in range(len(y_pred[0])):
        sum =  sum + (y_true[i][j] * log(y_pred[i][j]))
    cross_val = cross_val + sum

print("Cross Entropy value" + str(-cross_val/len(y_true)))

