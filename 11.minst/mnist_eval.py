# -*- coding: utf-8 -*-
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import mnist_train

EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name = 'x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name = 'y-input')
    validate_feed = {x: mnist.validation.images,
                     y_: mnist.validation.labels }

    y = mnist_inference.inference(x, None, 1.0)

    correctPredition = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy         = tf.reduce_mean(tf.cast(correctPredition, tf.float32))

    variableAverage            = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
    variableAverage_to_restore = variableAverage.variables_to_restore()
    saver = tf.train.Saver(variableAverage_to_restore)

    while True:
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                globalStep = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracyScore = sess.run(accuracy, feed_dict = validate_feed)

                print("After %s training step(s), validation accuracy = %g" % (globalStep, accuracyScore))

            else:
                print("No checkpoint file found")
                return
            time.sleep(EVAL_INTERVAL_SECS)

def main(argv = None):
    mnist = input_data.read_data_sets("/tmp/data", one_hot = True)
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()




