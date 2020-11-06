import tensorflow as tf
import preprocessing as pp
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dense

x_train_scale = pp.x_train
x_test_scale = pp.x_test
y_train_scale = pp.y_train
y_test_scale = pp.y_test

model = tf.keras.models.Sequential()
model.add(Dense(11, input_dim=11, kernel_initializer='normal', activation='relu'))
model.add(Dense(2670, activation='relu'))
model.add(Dense(1, activation='relu'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
history = model.fit(x_train_scale, y_train_scale, epochs=1000, batch_size=150, verbose=1, validation_split=0.2)
predictions = model.predict(x_test_scale)
print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
