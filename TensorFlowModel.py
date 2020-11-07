import tensorflow as tf
from keras.optimizers import SGD
import preprocessing as pp
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dense

# preprocessing.py is common processing file for all models in this project.. the data is split and normalized
# normalized values are assigned to the model
x_train_scale = pp.x_train
x_test_scale = pp.x_test
y_train_scale = pp.y_train
y_test_scale = pp.y_test


model = tf.keras.models.Sequential()  # using pre-defined keras sequential model
model.add(Dense(11, input_dim=11, kernel_initializer='normal', activation='relu'))
# defining the input layer with 11 input nodes for neural network
model.add(Dense(45, activation='relu'))
# defining hidden layer 1 with 2670 nodes
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='relu'))
# defining output node, with one output node for neural network
model.summary()
# gives a synopsis of our neural network model

# optimization function is defined, using sgd: gradient descent optimizer with learning rate, in built optimizer used for regression
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
# cost function is defined using mean square error mse, and mean absolute error mae

model.compile(loss='mse', optimizer='sgd', metrics=['mse', 'mae', 'accuracy'])
history = model.fit(x_train_scale, y_train_scale, epochs=1000, batch_size=100, verbose=1, validation_split=0.2)
# training the model
predictions = model.predict(x_test_scale)
print(history.history.keys())
# "Loss" plot for loss vs epoch, gives us information about accuracy
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
