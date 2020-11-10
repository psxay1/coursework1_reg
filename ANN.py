import tensorflow as tf
import preprocessing as pp
import matplotlib.pyplot as plt
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.set_random_seed(13)  # to make sure the experiment is reproducible

train_x = pp.x_train
train_y = pp.y_train
test_x = pp.x_test
test_y = pp.y_test

# Network parameters

n_hidden1 = 45
n_hidden2 = 25
n_input = 11
n_output = 1

# Learning parameters

learning_constant = 0.01
training_epochs = 1500

# Defining input and output

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_output])

# defining biases and weights
# b1 = tf.Variable(tf.random_normal([n_hidden1]))
# b2 = tf.Variable(tf.random_normal([n_hidden2]))
# b3 = tf.Variable(tf.random_normal([n_output]))
# w1 = tf.Variable(tf.random_normal([n_input, n_hidden1]))
# w2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2]))
# w3 = tf.Variable(tf.random_normal([n_hidden2, n_output]))

# defining the neural network (choose activation function)
# def multilayer_perceptron_1(input_X, activation_fcn):
#     #Task of neurons of first layer
#     layer_1 = activation_fcn(tf.add(tf.matmul(input_X, w1), b1))
#     print(layer_1)
#     #Task of neurons of second hidden layer
#     layer_2 = activation_fcn(tf.add(tf.matmul(layer_1, w2), b2))
#     print(layer_2)
#     #Task of neurons of output layer
#     out_layer = tf.add(tf.matmul(layer_2, w3),b3)
#     print(out_layer)
#     return out_layer

# defining biases and weights
def weights_and_biases_generator(input_layer_nodes, out_layer_nodes, *hidden_layer_nodes):
    biases = []
    weights = []

    temp = input_layer_nodes
    
    for n in hidden_layer_nodes:
        print("weights:", [temp, n])
        print("biases:", [n])

        weights.append(tf.Variable(tf.random_normal([temp, n])))
        biases.append(tf.Variable(tf.random_normal([n])))
        temp = n
      
    weights.append(tf.Variable(tf.random_normal([temp, out_layer_nodes])))
    biases.append(tf.Variable(tf.random_normal([out_layer_nodes])))
    print("weights:",[temp, out_layer_nodes])
    print("biases:", [out_layer_nodes])
   
    return weights, biases

ws, bs = weights_and_biases_generator(11,1,45,25)

# print(ws)
# print(bs)

# defining the neural network (can choose activation function, weights, biases)
def multilayer_perceptron_2(input_X, activation_fcn, weights, biases):
    wnb = list(zip(weights, biases))
    print("ZIP", wnb)
    # Task of neurons of first layer
    layer_1 = activation_fcn(tf.add(tf.matmul(input_X, wnb[0][0]), wnb[0][1]))
    d = {}
    d[0] = layer_1
    # Task of neurons of hidden layer(s)
    i = 1
    while i< len(weights)-1:
        d[i] = activation_fcn(tf.add(tf.matmul(d[i-1], wnb[i][0]), wnb[i][1]))
        i+=1
    # Task of neurons of output layer
    out_layer = tf.add(tf.matmul(d[i-1], wnb[-1][0]),wnb[-1][1])
    return out_layer

# Initialising network graph

# neural_net = multilayer_perceptron_1(X, tf.nn.sigmoid)
# neural_net = multilayer_perceptron_3(X, tf.nn.sigmoid)
# neural_net = multilayer_perceptron_2(X, tf.nn.sigmoid, [w1,w2,w3], [b1,b2,b3])
neural_net = multilayer_perceptron_2(X, tf.nn.sigmoid, ws, bs)
# print("Neural Net:", neural_net)


# defining cost and optimizer
# learning rate 0.001

# cost = tf.reduce_mean(tf.math.squared_difference(neural_net, Y))
n = len(train_x)
cost = tf.reduce_sum(tf.pow(neural_net-Y, 2)) / (2 * n)
optimizer = tf.train.GradientDescentOptimizer(learning_constant).minimize(cost)

init = tf.global_variables_initializer()

epoch_plot = []
cost_plot = []

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(1500):
        opt, cost_val = sess.run([optimizer, cost], feed_dict={X: train_x, Y: train_y})
        epoch_plot.append(epoch)
        cost_plot.append(cost_val)
        if (epoch + 1) % 100 == 0:
            c = sess.run(cost, feed_dict={X: train_x, Y: train_y})
            print("Epoch", (epoch + 1), ": cost =", c)

    training_cost = sess.run(cost, feed_dict ={X: train_x, Y: train_y})
    weight = sess.run(ws)
    bias = sess.run(bs)

    predictions = tf.matmul(weight, X)
    print(predictions)