# TensorFlow TrainNet 
import tensorflow as tf 
from helpers import lazy_property 
import Params as cfg 
import RNN as rnn 
import FCN as fcn 
import os 
import numpy as np 
import random 
import collections 
import time 
import shutil 

# 常量 __XXX__ 
# 内部变量或方法 _xxx 
# 公开变量或方法 xxx 
# 定义参数 

##############网络结构########################### 
#   RNN -> FCN 
################################################## 
##############子网络定义######################### 
# RNN0 
# 子网络名称 
RNN_NAME = 'RNN-FCNNet-RNN' 
# RNN元素数量 
RNN_CELL_NUM = 512 
# RNN类型 
RNN_CELL_BUILDER = tf.contrib.rnn.BasicLSTMCell 
# RNN层数 
RNN_LAYER_NUM = 2 
# RNN时间序列 
RNN_SEQUENCE_NUM = 3 
# RNN帧大小 
RNN_FRAME_SIZE = 1 
# RNN词汇量 
RNN_VOCABULARY_SIZE = 112 

# FCN0 
# 子网络名称 
FCN_NAME = 'RNN-FCNNet-FCN' 
# 网络结构（从第二层开始） 
FCN_NET_SHAPE = [RNN_VOCABULARY_SIZE] 
# 最后一层激活函数 
FCN_FINAL_ACTIVATE_FUNC = None 

##############训练网络定义####################### 
# 网络名称 
TRAIN_NET_NAME = 'RNN-FCNNet' 
# 网络组成描述 
TRAIN_NET_DESCRIPTION = '[RNN]->[FCN]' 
# 输入数据结构 
INPUT_SHAPE = [None, RNN_SEQUENCE_NUM, RNN_FRAME_SIZE] 
# 输出数据结构 
OUTPUT_SHAPE = [None, RNN_VOCABULARY_SIZE] 
# 学习速率 
LEARN_RATE = 0.01 
# 从上次继续训练 
FLAG_RESTORE_LAST = True 
# 是否打开统计
FLAG_OPEN_SUMMARY = True 
# 梯度下降函数 
TRAIN_LOSS_FUNC = tf.train.RMSPropOptimizer 

# 特定网络定义 
class TrainNet(object): 
    #------------默认函数定义-----------# 
    def __init__(self, train_param, fcn_param = None, rnn_param = None): 
        # param 
        self.__reset_param(train_param) 

        # build net 
        self._rnn_net = rnn.RNN(rnn_param) 

        # transpose 
        output_transposed = None 
        with self._graph.as_default(): 
            output_transposed = tf.transpose(self._rnn_net.outputs_struct, [1, 0, 2]) 

        # link net 
        self.__link_two_net(output_transposed[-1], fcn_param) 
        self._fcn_net = fcn.FCN(fcn_param) 

        # get final output 
        self._output = self._fcn_net.outputs_struct 

        # generate loss/accuracy optimizer 
        self.__loss 
        self.__accuracy 
        self.__optimizer 

        # prepare summary 
        if self._need_summary: 
            if not os.path.exists(self._model_dir): 
                os.mkdir(self._model_dir) 

            self._writer = tf.summary.FileWriter(self._model_dir, self._graph) 

    def __del__(self): 
        if self._writer is not None: 
            self._writer.flush() 
            self._writer.close()

    #------------内部函数定义-----------# 
    def __reset_param(self, param): 
        # Set Parameters 
        self._input_struct = param.input_struct 
        self._graph = param.graph 
        self._name = param.name 
        self._learn_rate = param.learn_rate 
        self._optimizer = param.optimizer 
        self._output_label = param.output_label 
        self._model_dir = param.model_dir + '/' + self._name 
        self._restore_last = param.restore_last 
        self._description = param.description 
        self._need_summary = param.need_summary 
        self._writer = None 

    def __link_two_net(self, out_port, param): 
        param.set_input_struct(out_port) 

    #------------外部接口定义-----------# 
    # 初始化当前网络 
    def init(self): 
        with self._graph.as_default(): 
            return tf.global_variables_initializer() 

    # 训练当前网络 
    def train(self, sess, train_input, train_output): 
        _ = sess.run([self.__optimizer], feed_dict = {self._input_struct: train_input, self._output_label: train_output}) 

    # 获取当前网络损失 保存Summary 
    def refresh_state(self, sess, train_input, train_output, step): 
        loss, accuracy, pred, summary = sess.run([self.__loss, self.__accuracy, self._output, self.__summarier], feed_dict = {self._input_struct: train_input, self._output_label: train_output}) 

        if self._writer is not None: 
            self._writer.add_summary(summary, global_step = step) 

        return [loss, accuracy, pred] 

    # 保存当前模型 
    def save(self, sess): 
        saver = tf.train.Saver() 
        if not os.path.exists(self._model_dir): 
            os.mkdir(self._model_dir)

        saver.save(sess, os.path.join(self._model_dir, _self._name)) 

    # 恢复模型 
    def restore(self, sess): 
        if self._restore_last: 
            if not os.path.exists(self._model_dir): 
                return 

            saver = tf.train.Saver() 
            try: 
                saver.restore(sess, os.path.join(self._model_dir, self._name)) 
            except: 
                return 

    # 训练结果可视化 
    def visualize(self): 
        pass 

    # 打印当前模型信息 
    def dump(self): 
        print("**********Train Begin*******") 
        print("* name :", self._name) 
        print("* learn_rate :", self._learn_rate) 
        print("* optimizer :", self._optimizer) 
        print("* input :", self._input_struct.shape) 
        print("* outputs :", self._output_label.shape) 
        print("* restore_last:", self._restore_last) 
        print("* model_dir :", self._model_dir) 
        print("* description :", self._description) 
        print("* summary_flg :", self._need_summary) 
        print("**********Train End*********") 
        self._rnn_net.dump() 
        self._fcn_net.dump() 

    # 当前网络的损失函数(这里的lazy_property为了防止多次创建) 放到专门的loss类中，因为是针对整个网络而言的 
    @lazy_property 
    def __loss(self): 
        with self._graph.as_default(): 
            with tf.name_scope("loss_layer"): 
                return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self._output, labels = self._output_label), name = "loss_ce") 

    # 当前网络的准确率函数 acc类，针对不同的网络类型而言,可能放到外部的组合网络的类中实现 
    @lazy_property 
    def __accuracy(self): 
        with self._graph.as_default(): 
            with tf.name_scope("accuracy_layer"): 
                correct_pred = tf.equal(tf.argmax(self._output, 1), tf.argmax(self._output_label, 1)) 
                return tf.reduce_mean(tf.cast(correct_pred, tf.float32), name = "accuracy_op") 

    # 当前网络的梯度下降算法 loss类 
    @lazy_property 
    def __optimizer(self): 
        with self._graph.as_default(): 
            tf.summary.scalar("loss", self.__loss)
            tf.summary.scalar("accuracy", self.__accuracy) 
            return self._optimizer(self._learn_rate).minimize(self.__loss) 

    # Summary 
    @lazy_property 
    def __summarier(self): 
        with self._graph.as_default(): 
            if self._writer is not None: 
                return tf.summary.merge_all() 

            return tf.no_op() 

    #--------------属性定义------------# 
    # 输入tensor结构 
    @lazy_property 
    def input_struct(self): 
        return self._input_struct 

    # 输出tensor结构 
    @lazy_property 
    def outputs_struct(self): 
        return self._output 

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

if __name__ == "__main__": 
    # d dump、d save、d summary, link， 
    # visual 自定义的可视化，以及accuracy、召回率等基础指标的tensorboard打印 

    # 数据 
    # read file 
    training_file = 'rnn_training.txt' 
    training_data = read_data(training_file) 

    # make dictionary 
    dictionary, reverse_dictionary = build_dataset(training_data) 
    vocab_size = len(dictionary) 

    # 设置网络参数 
    train_param = cfg.TrainParams(input_struct = tf.placeholder(dtype = tf.float32, shape = INPUT_SHAPE, name = 'input_data') 
                                , output_label = tf.placeholder(dtype = tf.float32, shape = OUTPUT_SHAPE, name = 'output_data') 
                                , learn_rate   = LEARN_RATE 
                                , name         = TRAIN_NET_NAME 
                                , description  = TRAIN_NET_DESCRIPTION
                                , restore_last = FLAG_RESTORE_LAST 
                                , need_summary = FLAG_OPEN_SUMMARY 
                                , optimizer    = TRAIN_LOSS_FUNC)

    rnn_param = cfg.RNNParams(input_struct = train_param.input_struct
                            , name         = RNN_NAME
                            , cell_num     = RNN_CELL_NUM 
                            , cell_builder = RNN_CELL_BUILDER 
                            , layer_num    = RNN_LAYER_NUM)

    fcn_param = cfg.FCNParams(input_struct  = None 
                            , net_shape     = FCN_NET_SHAPE 
                            , name          = FCN_NAME 
                            , activate_ffc  = FCN_FINAL_ACTIVATE_FUNC)
    # 构造网络 
    rnn_net = TrainNet(train_param, fcn_param, rnn_param) 
    rnn_net.dump() 

    # feed & train: 
    with tf.Session(graph = rnn_net._graph) as sess: 
        # training data: 
        offset = random.randint(0, RNN_SEQUENCE_NUM + 1) 
        end_offset = RNN_SEQUENCE_NUM + 1 
        acc_total = 0 
        loss_total = 0 
        step = 0 

        sess.run(rnn_net.init()) 
        rnn_net.restore(sess) 
        for step in range(50000): 
            if offset > (len(training_data) - end_offset): 
                offset = random.randint(0, RNN_SEQUENCE_NUM + 1) 

            symbols_in_keys = [[dictionary[str(training_data[i])]] for i in range(offset, offset + RNN_SEQUENCE_NUM)] 
            #print(symbols_in_keys) 
            symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, RNN_SEQUENCE_NUM, 1]) 
            #print(symbols_in_keys) 
            symbols_out_onehot = np.zeros([vocab_size], dtype = float) 
            symbols_out_onehot[dictionary[str(training_data[offset + RNN_SEQUENCE_NUM])]] = 1.0 
            symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1]) 
            rnn_net.train(sess, symbols_in_keys, symbols_out_onehot) 
            loss, acc, onehot_pred = rnn_net. refresh_state(sess, symbols_in_keys, symbols_out_onehot, step) 

            loss_total += loss 
            acc_total += acc 
            if (step + 1) % 1000 == 0: 
                print(loss_total.shape) 
                print("Iter= " + str(step+1) + ", Average Loss= " + \
                      "{:.6f}".format(loss_total/1000) + ", Average Accuracy= " + \
                      "{:.2f}%".format(100*acc_total/1000)) 

                acc_total = 0 
                loss_total = 0 
                symbols_in = [training_data[i] for i in range(offset, offset + RNN_SEQUENCE_NUM)] 
                symbols_out = training_data[offset + RNN_SEQUENCE_NUM] 
                symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())] 
                print("%s - [%s] vs [%s]" % (symbols_in, symbols_out, symbols_out_pred))
                
            offset += (RNN_SEQUENCE_NUM + 1)
                
        print("Optimization Finished!") 
                
        while True: 
            prompt = "%s words: " % RNN_SEQUENCE_NUM
            sentence = input(prompt) 
            sentence = sentence.strip() 
            words = sentence.split(' ') 
            if len(words) != RNN_SEQUENCE_NUM: 
                continue 
            try: 
                symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))] 
                for i in range(32): 
                    keys = np.reshape(np.array(symbols_in_keys), [-1, RNN_SEQUENCE_NUM, 1]) 
                    print(keys.shape) 
                    onehot_pred = predict_once_rnn_word_prediction(sess, keys) 
                    onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval()) 
                    sentence = "%s %s" % (sentence, reverse_dictionary[onehot_pred_index]) 
                    symbols_in_keys = symbols_in_keys[1:] 
                    symbols_in_keys.append(onehot_pred_index) 
                    print(sentence) 
            except: 
                print("Word not indictionary") 
    rnn_net.dump() 
    rnn_net.save(sess) 
    sess.close()

