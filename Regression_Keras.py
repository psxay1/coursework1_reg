import tensorflow as tf
from tensorflow import keras
from keras.optimizers import SGD
import preprocessing as pp
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import KFold

# preprocessing.py is common processing file for all models in this project.. the data is split and normalized

tf.keras.backend.set_epsilon(1)

# cost function is defined using mean square error mse, and mean absolute error mae
kf = KFold(n_splits=10, shuffle=True)

features = pp.kFold_features
labels = pp.kFold_labels
scores = []

try:
    trained_model = keras.models.load_model('wine_model.h5')
    test_features = pp.features_test
    test_labels = pp.labels_test
    trained_model.compile(loss='mse', optimizer='sgd',
                          metrics=['mse', 'mae', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error'])
    history = trained_model.fit(test_features, test_labels)
    prediction = trained_model.predict(test_features)
    ax = plt.subplot()
    ax.scatter(features[:, 0], labels)
    plt.show()
except OSError:
    model = tf.keras.models.Sequential()
    x_test_arr = []
    y_test_arr = []
    x_train_arr = []
    y_train_arr = []
    for train_index, test_index in kf.split(features):
        x_train, x_test, y_train, y_test = features[train_index], features[test_index], labels[train_index], labels[
            test_index]

        print("-----------------------------In the loo-----------------------------")

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
                            batch_size=100, verbose=0)
        score = model.evaluate(x_test, y_test)
        scores.append(score)

    # print("Last: ", scores[-1])
    model.save('wine_model.h5')
    # "Loss" plot for loss vs epoch, gives us information about accuracy
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()