# -*- coding: utf-8 -*-
import tensorflow as tf

# Define Global Layer
INPUT_NODE  = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(
        name        = "weights", 
        shape       = shape,
        initializer = tf.truncated_normal_initializer(stddev = 0.1))

    if regularizer != None:
        tf.add_to_collection("losses", regularizer(weights))

    return weights

def get_biases_variable(shape):
    biases = tf.get_variable(
        name        = "biases", 
        shape       = shape,
        initializer = tf.constant_initializer(0.1))

    return biases

def get_dropout_layer(inputTensor, keepProb):
    return tf.nn.dropout(inputTensor, keepProb)

def get_cov2d(inputTensor, filter):
    return tf.nn.conv2d(
        input   = inputTensor,
        filter  = filter,
        strides = [1, 1, 1, 1],
        padding = 'SAME')

def get_max_pool2x2(inputTensor):
    return tf.nn.max_pool(
        value   = inputTensor,
        ksize   = [1, 2, 2, 1],
        strides = [1, 2, 2, 1],
        padding = 'SAME')


# [Input(28x28x1)]=>[Conv1(3x3) 32]=>[MaxPool]=>[Conv2(3x3) 64]=>[MaxPool]=>[AllLink1]=>[Oupput(10)]
def inference(inputTensor, regularizer, keepProb = 0.8, actFunction = tf.nn.relu):
    x_image = tf.reshape(inputTensor, [-1, 28, 28, 1])
    with tf.variable_scope('conv2d1'):
        w_conv_1 = get_weight_variable([3, 3, 1, 32], None)
        b_conv_1 = get_biases_variable([32])
        h_conv_1 = actFunction(get_cov2d(x_image, w_conv_1) + b_conv_1)
        h_pool_1 = get_max_pool2x2(h_conv_1)

    with tf.variable_scope('conv2d2'):
        w_conv_2 = get_weight_variable([3, 3, 32, 64], None)
        b_conv_2 = get_biases_variable([64])
        h_conv_2 = actFunction(get_cov2d(h_pool_1, w_conv_2) + b_conv_2)
        h_pool_2 = get_max_pool2x2(h_conv_2)
        h_pool_2_flat = tf.reshape(h_pool_2, [-1, 7 * 7 * 64]) # Prepare AllLinked Layer Shape

    with tf.variable_scope('layer1'):
        weights = get_weight_variable([7 * 7 * 64, LAYER1_NODE], regularizer)
        biases  = get_biases_variable([LAYER1_NODE])
        layer_1 = actFunction(tf.matmul(h_pool_2_flat, weights) + biases)
        layer_1_dropout = get_dropout_layer(layer_1, keepProb)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases  = get_biases_variable([OUTPUT_NODE])
        layer_2 = tf.nn.softmax(tf.matmul(layer_1_dropout, weights) + biases)

    return layer_2



