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
# using pre-defined keras sequential model
model = tf.keras.models.Sequential()
# defining the input layer 1 of the Neural network
model.add(Dense(11, input_dim=11, kernel_initializer='normal', activation='relu'))
# defining the input layer 2 of the Neural network
model.add(Dense(45, activation='relu'))
# defining the input layer 3 of the Neural network
model.add(Dense(25, activation='relu'))
# defining the input layer 4 of the Neural network i.e. output node
model.add(Dense(1, activation='relu'))
# gives a synopsis of our neural network model
model.summary()
# optimization function is defined, using sgd: gradient descent optimizer with learning rate
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
# cost function is defined using mean square error mse, and mean absolute error mae
model.compile(loss='mse', optimizer='sgd', metrics=['mse', 'mae'])
# training the model
history = model.fit(x_train_scale, y_train_scale, validation_data=(x_test_scale, y_test_scale), epochs=1000,
                    batch_size=150, verbose=1, validation_split=0.2)
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
