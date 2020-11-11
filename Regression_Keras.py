import tensorflow as tf
from tensorflow import keras
import preprocessing as pp
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import KFold
from math import sqrt


# preprocessing.py is common processing file for all models in this project.. the data is split and normalized

tf.keras.backend.set_epsilon(1)


def test_trained_model():
    trained_model = keras.models.load_model('wine_model.h5')
    test_features = pp.features_test
    test_labels = pp.labels_test
    trained_model.compile(loss='mse', optimizer='sgd',
                          metrics=['mse', 'mae', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error'])
    history = trained_model.fit(test_features, test_labels)
    y_pred = trained_model.predict(test_features)
    final_rmse = trained_model.evaluate(test_features, test_labels)[0]
    print("Average RMSE :---------------", final_rmse)

def train_model():
    features = pp.kFold_features
    labels = pp.kFold_labels
    rmse_arr = []
    model = tf.keras.models.Sequential()

    # defining the kFold splits (10-fold)
    kf = KFold(n_splits=10, shuffle=True)

    for train_index, test_index in kf.split(features):
        x_train, x_test, y_train, y_test = features[train_index], features[test_index], labels[train_index], labels[
            test_index]

        # using pre-defined keras sequential model
        model = tf.keras.models.Sequential()
        # defining the input layer of the Neural network
        model.add(Dense(11, input_dim=11, kernel_initializer='normal', activation='relu'))
        # defining the hidden layer 1 of the Neural network
        model.add(Dense(45, activation='relu'))
        # defining the hidden layer 2 of the Neural network
        model.add(Dense(25, activation='relu'))
        # defining the output layer of the Neural network i.e. output node
        model.add(Dense(1, activation='relu'))
        # gives a synopsis of our neural network model
        model.summary()
        # defining the metrics, optimizer and loss function for compiling
        model.compile(loss='mse', optimizer='sgd',
                      metrics=['mse', 'mae', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error'])

        history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=35,
                            batch_size=100, verbose=0)

        rmse = sqrt(model.evaluate(x_test, y_test)[0])
        rmse_arr.append(rmse)

    avg_rmse = sum(rmse_arr) / len(rmse_arr)
    # Average RMSE
    print("Average RMSE :---------------", avg_rmse)

    # Saving the trained model in a file
    model.save('wine_model.h5')

    # "Loss" plot for loss vs epoch, gives us information about accuracy
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

train_model()