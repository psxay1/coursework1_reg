import tensorflow as tf
from keras.optimizers import SGD
import preprocessing as pp
from matplotlib.pyplot import ylabel, plot, legend, show, xlabel, title
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import KFold
import  numpy as np
# preprocessing.py is common processing file for all models in this project.. the data is split and normalized

tf.keras.backend.set_epsilon(1)

# cost function is defined using mean square error mse, and mean absolute error mae
kf = KFold(n_splits=10, shuffle=True)

features = pp.kFold_features
labels = pp.kFold_labels
scores = []
x_test_arr = []
y_test_arr = []

for train_index, test_index in kf.split(features):
    x_train, x_test, y_train, y_test = features[train_index], features[test_index], labels[train_index], labels[test_index]

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

    model.compile(loss='mse', optimizer='sgd',
                  metrics=['mse', 'mae', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error'])
    x_test_arr.append(x_test)
    y_test_arr.append(y_test)
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=35,
                              batch_size=100, verbose=0, validation_split=0.4)
    score = model.evaluate(x_test, y_test)
    scores.append(score)

print("Lasr: ", scores[-1])

# "Loss" plot for loss vs epoch, gives us information about accuracy
plot(history.history['loss'])
plot(history.history['val_loss'])
title('model loss')
ylabel('loss')
xlabel('epoch')
legend(['train', 'test'], loc='upper left')
show()