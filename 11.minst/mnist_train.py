# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference

BATCH_SIZE              = 100
LEARNING_RATE_BASE      = 1e-3
LEARNING_RATE_DECAY     = 0.97
REGULARAZTION_RATE      = 0.0001
TRAINING_STEPS          = 80000000
MOVING_AVERAGE_DECAY    = 0.99

MODEL_SAVE_PATH = "D:\MNIST\model"
MODEL_NAME      = "model.ckpt"

def train(mnist):
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name = 'x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name = 'y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = mnist_inference.inference(x, regularizer, 1.0)

    # 滑动平均
    globalStep          = tf.Variable(0, trainable = False)
    variableAverages    = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, globalStep)
    variableAverages_op = variableAverages.apply(tf.trainable_variables())

    corssEntropy_mean = -tf.reduce_mean(tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), 1))

    #crossEntropy        = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.argmax(y_, 1), logits = y)
    #corssEntropy_mean   = tf.reduce_mean(crossEntropy)
    loss                = corssEntropy_mean + tf.add_n(tf.get_collection('losses'))
    #loss                = corssEntropy_mean

    # 学习率衰减
    learningRate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        globalStep,
        5000.0,
        LEARNING_RATE_DECAY)

    trainStep = tf.train.AdamOptimizer(learningRate).minimize(loss, global_step = globalStep)

    with tf.control_dependencies([trainStep, variableAverages_op]):
        train_op = tf.no_op(name = 'train')

    saver = tf.train.Saver()

    # Start
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            _, lossValue, step = sess.run([train_op, loss, globalStep], feed_dict = {x: xs, y_: ys})

            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g" % (step, lossValue))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = globalStep)

def main(argv = None):
    mnist = input_data.read_data_sets("/tmp/data", one_hot = True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()

