from keras import Model, Sequential
from keras.layers import Input
from keras.layers import LSTM
from keras.src.layers import Dense
from numpy import array
# define model
# define model
inputs1 = Input(shape=(3, 1))
lstm1, state_h, state_c = LSTM(1,  return_state = True)(inputs1)
model = Model(inputs=inputs1, outputs=[lstm1, state_h, state_c])
# define input data
data = array([0.1, 0.2, 0.3]).reshape((1,3,1))
# make and show prediction
# Make prediction
lstm_out, hidden_state, cell_state = model.predict(data)

print("a. LSTM output (last time step):      ", lstm_out)
print("b. Hidden state (last time step):     ", hidden_state)
print("c. Cell state (last time step):       ", cell_state)

print(model.predict(data))