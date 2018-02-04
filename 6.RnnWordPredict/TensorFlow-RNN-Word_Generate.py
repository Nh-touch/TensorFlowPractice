# -*- utf-8 -*- 
from __future__ import print_function
import numpy as np
import tensorflow as tf 
from tensorflow.contrib import rnn
import random 
import collections
import time

# global value 
LEARN_RATE = 0.01 
HIDDEN_NERU_NUM = 512
VOCABULARY_SIZE = 112
SEQUENCE_NUM = 3
FRAME_SIZE = 1

# read data from file name
def read_data(fname): 
    with open(fname) as f:
        content = f.readlines() 
        content = [x.strip() for x in content]
        content = [content[i].split() for i in range(len(content))] 
        content = np.array(content)
        content = np.reshape(content, [-1, ]) 
    return content

# make vocabulary dictionary
def build_dataset(words): 
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary) 
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
    return dictionary, reverse_dictionary

# get a RNN Net Result 
def RNN(x, weight, bias): 
    # reshpae input to fit rnn_layer's need 
    x = tf.reshape(x, [-1, SEQUENCE_NUM]) 
    x = tf.split(x, SEQUENCE_NUM, 1)
    
    # define rnn layer 
    rnn_layer = rnn.MultiRNNCell([rnn.BasicLSTMCell(HIDDEN_NERU_NUM), rnn.BasicLSTMCell(HIDDEN_NERU_NUM)] , state_is_tuple=True)

    # link input to rnn_layer 
    outputs, state = rnn.static_rnn(rnn_layer, x, dtype = tf.float32) 
    return tf.add(tf.matmul(outputs[-1], weight), bias, name = "rnn_pred") 

# define the graph
def form_word_prediction_forward_graph():
    graph0 = tf.Graph()
    with graph0.as_default():
        with tf.name_scope("global_counter"): 
            counter = tf.get_variable("g_counter", [], dtype=tf.int32, trainable=False, initializer=tf.initializers.zeros) 

        with tf.name_scope("train_variables"): 
            weights = tf.get_variable("out_weights", [HIDDEN_NERU_NUM, VOCABULARY_SIZE], dtype=tf.float32, trainable=True, initializer=tf.initializers.truncated_normal(0.1)) 
            bias = tf.get_variable("out_bias", [1, VOCABULARY_SIZE], dtype=tf.float32, trainable=True, initializer=tf.initializers.constant(0.1))

        with tf.name_scope("train_data_source"): 
            input_data = tf.placeholder(dtype = tf.float32, shape = [None, SEQUENCE_NUM, FRAME_SIZE], name = "x-input") # batchsize * sequence * frames
            label_data = tf.placeholder(dtype = tf.float32, shape = [None, VOCABULARY_SIZE], name = "y-input") # scalar (vocabulary size)

        with tf.variable_scope("data_forward_process"): 
            with tf.name_scope("output_layer"): 
                logit = RNN(input_data, weights, bias) 
                print(logit.name) 

        with tf.name_scope("loss_layer"): 
            # get cross_entry func 
            loss_ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logit, labels = label_data), name = "loss_ce")

        with tf.name_scope("accuracy_layer"):
            correct_pred = tf.equal(tf.argmax(logit, 1), tf.argmax(label_data, 1))

        with tf.name_scope("summary"): 
            tf.summary.histogram("output", logit)
            tf.summary.histogram("weightsall", weights)
            tf.summary.histogram("bias", bias)
            tf.summary.scalar("loss_mse", loss_ce) 
            tf.summary.scalar("global_counter", counter)

        with tf.name_scope("output_layer"): 
            summary_merge_op = tf.summary.merge_all() # Merge/MergeSummary:0 
            init_op = tf.global_variables_initializer() # init 
            print(init_op.name)
            with tf.control_dependencies([tf.assign_add(counter, 1)]):
                train_op = tf.train.RMSPropOptimizer(LEARN_RATE).minimize(loss_ce)
                print(train_op.name) 
            accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name = "accuracy_op")
                
        return graph0 

def feed_once_rnn_word_prediction(sess, train_data, train_output):
    input_data = sess.graph.get_tensor_by_name("train_data_source/x-input:0")
    label_data = sess.graph.get_tensor_by_name("train_data_source/y-input:0") 
    train_op = sess.graph.get_operation_by_name("output_layer/RMSProp") 
    accuracy = sess.graph.get_tensor_by_name("output_layer/accuracy_op:0")
    loss = sess.graph.get_tensor_by_name("loss_layer/loss_ce:0") 
    pred = sess.graph.get_tensor_by_name("data_forward_process/output_layer/rnn_pred:0") 
    _, acc, loss, onehot_pred = sess.run([train_op, accuracy, loss, pred], feed_dict = {input_data: train_data, label_data: train_output}) 

    return [acc, loss, onehot_pred] 

def predict_once_rnn_word_prediction(sess, train_data):
    input_data = sess.graph.get_tensor_by_name("train_data_source/x-input:0") 
    pred = sess.graph.get_tensor_by_name("data_forward_process/output_layer/rnn_pred:0") 
    onehot_pred = sess.run(pred, feed_dict = {input_data: train_data}) 
    return onehot_pred

def main(_):
    # read file 
    training_file = 'rnn_training.txt'
    training_data = read_data(training_file)
##    print(training_data)
    # make dictionary 
    dictionary, reverse_dictionary = build_dataset(training_data) 
    vocab_size = len(dictionary)
    
    # make process graph
    graph0 = form_word_prediction_forward_graph() # tensorboard recorder 
    writer = tf.summary.FileWriter("D:/tensorlog", graph0)
    
    with tf.Session(graph = graph0) as sess:
        # training data: 
        offset = random.randint(0, SEQUENCE_NUM + 1)
        end_offset = SEQUENCE_NUM + 1
        acc_total = 0 
        loss_total = 0
        step = 0
        tf.global_variables_initializer().run() 
        for step in range(50000): 
            if offset > (len(training_data) - end_offset): 
                offset = random.randint(0, SEQUENCE_NUM + 1)
                
            symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(offset, offset + SEQUENCE_NUM) ] 
            symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, SEQUENCE_NUM, 1]) 
            symbols_out_onehot = np.zeros([vocab_size], dtype = float)
            symbols_out_onehot[dictionary[str(training_data[offset + SEQUENCE_NUM])]] = 1.0
            symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1])
            acc, loss, onehot_pred = feed_once_rnn_word_prediction(sess, symbols_in_keys, symbols_out_onehot) 
            loss_total += loss
            acc_total += acc 
            if (step + 1) % 1000 == 0:
                print("Iter= " + str(step+1) + ", Average Loss= " + \
                      "{:.6f}".format(loss_total/1000) + ", Average Accuracy= " + \
                      "{:.2f}%".format(100*acc_total/1000)) 
                acc_total = 0
                loss_total = 0
                symbols_in = [training_data[i] for i in range(offset, offset + SEQUENCE_NUM)]
                symbols_out = training_data[offset + SEQUENCE_NUM] 
                symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
                print("%s - [%s] vs [%s]" % (symbols_in, symbols_out, symbols_out_pred)) 
            offset += (SEQUENCE_NUM + 1)
            
        print("Optimization Finished!") 

        while True: 
            prompt = "%s words: " % SEQUENCE_NUM
            sentence = input(prompt) 
            sentence = sentence.strip()
            words = sentence.split(' ') 
            if len(words) != SEQUENCE_NUM: 
                continue

            try: 
                symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))] 
                for i in range(32): 
                    keys = np.reshape(np.array(symbols_in_keys), [-1, SEQUENCE_NUM, 1])  
                    onehot_pred = predict_once_rnn_word_prediction(sess, keys)
                    onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())

                    sentence = "%s %s" % (sentence, reverse_dictionary[onehot_pred_index]) 
                    symbols_in_keys.append(onehot_pred_index)
                    symbols_in_keys = symbols_in_keys[1:]
                    print(symbols_in_keys)

                print(sentence) 
            except:
                print("Word not in dictionary")
        sess.close()
    writer.flush() 
    writer.close() 

if __name__ == "__main__":
    tf.app.run()

