import tensorflow as tf
import preprocessing as pp
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.set_random_seed(13)  # to make sure the experiment is reproducible

train_x = pp.x_train
train_y = pp.y_train
test_x = pp.x_test
test_y = pp.y_test

# Network parameters

n_hidden1 = 50
n_hidden2 = 20
n_input = 12
n_output = 1

# Learning parameters

learning_constant = 0.01
training_epochs = 1500

# Defining input and output

X = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])

# defining biases and weights

b1 = tf.Variable(tf.random_normal([n_hidden1]))
b2 = tf.Variable(tf.random_normal([n_hidden2]))
b3 = tf.Variable(tf.random_normal([n_output]))
w1 = tf.Variable(tf.random_normal([n_input, n_hidden1]))
w2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2]))
w3 = tf.Variable(tf.random_normal([n_hidden2, n_output]))

# defining the neural network (relu activation function)

layer_1 = tf.nn.relu(tf.add(tf.matmul(X, w1), b1))
layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, w2), b2))
out_layer = tf.add(tf.matmul(layer_2, w3), b3)

# defining cost and optimizer
# learning rate 0.001

cost = tf.reduce_mean(tf.math.squared_difference(out_layer, y))
optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

init = tf.global_variables_initializer()

acc = []
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(1500):
        opt, cost_val = sess.run([optimizer, cost], feed_dict={X: train_x, y: train_y})
        matches = tf.equal(tf.argmax(out_layer, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(matches, 'float'))
        acc.append(accuracy.eval({X: test_x, y: test_y}))
        if epoch % 100 == 0:
            print("Epoch", epoch, "--", "Cost", cost_val)
            print("Accuracy on test set ", accuracy.eval({X: test_x, y: test_y}))
