# TensorFlow TrainNet
import tensorflow as tf
from helpers import lazy_property 
import Params as cfg
import RNN as rnn
import FCN as fcn 
import CNN as cnn
import os 
import numpy as np
import random
import collections
import time
import shutil
from tensorflow.examples.tutorials.mnist import input_data 
# 常量 __XXX__
# 内部变量或方法 _xxx 
# 公开变量或方法 xxx
# 定义参数
##############网络结构########################### #
# RNN -> FCN # 
#################################################
##############子网络定义######################### 
# CNN0 
# 子网络名称 
CNN_NAME = 'MnistCNNNet-CNN'
# 图片宽度
CNN_IMAGE_WIDTH = 28
# 图片高度
CNN_IMAGE_HEIGHT = 28
# 图片深度 
CNN_IMAGE_DEPTH = 1
# 分类数量
CNN_SORT_TYPES = 10
# CNN卷积层定义
CNN_CONV_L1 = [3, 3, 1, 32]
CNN_CONV_L2 = [3, 3, 32, 64] 
CNN_CONV_L3 = [3, 3, 64, 128] 
# RNN词汇量 
RNN_VOCABULARY_SIZE = 112
# FCN0
# 子网络名称
FCN_NAME = 'MnistCNNNet-FCN' 
# 网络结构（从第二层开始） 
FCN_NET_SHAPE = [200, 10] 
# 最后一层激活函数 
FCN_FINAL_ACTIVATE_FUNC = tf.nn.softmax 

###训练网络定义#######################
# 网络名称 
TRAIN_NET_NAME = 'MnistCNNNet' 
# 网络组成描述
TRAIN_NET_DESCRIPTION = '[CNN]->[FCN]' 
# 输入数据结构
INPUT_SHAPE = [None, CNN_IMAGE_HEIGHT, CNN_IMAGE_WIDTH, CNN_IMAGE_DEPTH] 
# 输出数据结构
OUTPUT_SHAPE = [None, CNN_SORT_TYPES] 
# 学习速率 
LEARN_RATE = 0.001 
# 从上次继续训练
FLAG_RESTORE_LAST = True
# 是否打开统计
FLAG_OPEN_SUMMARY = True
# 梯度下降函数 
TRAIN_LOSS_FUNC = tf.train.AdamOptimizer 
# 特定网络定义
class TrainNet(object):
    #------------默认函数定义-----------# 
    def __init__(self, train_param, fcn_param = None, cnn_Param = None):
        # param 
        self.__reset_param(train_param) 
        # build net
        self._cnn_net = cnn.CNN(cnn_Param) 
        with self._graph.as_default(): 
            with tf.name_scope(CNN_NAME): 
                self._cnn_net.add_unit(name = 'conv3x3x32' 
                                     , conv_filter = [CNN_CONV_L1] 
                                     , conv_stride = [[1, 1, 1, 1]] 
                                     , pool_ksize = [[1, 2, 2, 1]] 
                                     , pool_stride = [[1, 2, 2, 1]] 
                                     , activate_func = tf.nn.relu 
                                     , norm = True)

                self._cnn_net.add_unit(name = 'conv3x3x64' 
                                     , conv_filter = [CNN_CONV_L2] 
                                     , conv_stride = [[1, 1, 1, 1]] 
                                     , pool_ksize = [[1, 2, 2, 1]] 
                                     , pool_stride = [[1, 2, 2, 1]] 
                                     , activate_func = tf.nn.relu 
                                     , norm = True) 

                self._cnn_net.add_unit(name = 'conv3x3x128' 
                                     , conv_filter = [CNN_CONV_L3] 
                                     , conv_stride = [[1, 1, 1, 1]] 
                                     , pool_ksize = [[1, 2, 2, 1]] 
                                     , pool_stride = [[1, 2, 2, 1]] 
                                     , activate_func = tf.nn.relu 
                                     , norm = True) 
                self._cnn_net.flat_output() 

        # link net 
        self.__link_two_net(self._cnn_net.outputs_struct, fcn_param) 

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
        self._is_test = param.is_test 
        self._cur_step = param.cur_step 

    def __link_two_net(self, out_port, param): 
        param.set_input_struct(out_port) 

    #------------外部接口定义-----------# 
    # 初始化当前网络 
    def init(self): 
        with self._graph.as_default(): 
            return tf.global_variables_initializer() 

    # 训练当前网络 
    def train(self, sess, train_input, train_output, step): 
        train_input = np.reshape(np.array(train_input), [-1, CNN_IMAGE_WIDTH, CNN_IMAGE_HEIGHT, CNN_IMAGE_DEPTH]) 
        _ = sess.run([self.__optimizer], feed_dict = {self._input_struct: train_input, self._output_label: train_output, self._is_test: False, self._cur_step: step}) 

    # 获取当前网络损失 保存Summary 
    def refresh_state(self, sess, train_input, train_output, step): 
        train_input = np.reshape(np.array(train_input), [-1, CNN_IMAGE_WIDTH, CNN_IMAGE_HEIGHT, CNN_IMAGE_DEPTH]) 

        loss, accuracy, pred, summary = sess.run([self.__loss, self.__accuracy, self._output, self.__summarier] 
                                                , feed_dict = {self._input_struct: train_input, self._output_label: train_output, self._is_test: False, self._cur_step: step}) 
        if self._writer is not None: 
            self._writer.add_summary(summary, global_step = step) 

        return [loss, accuracy, pred] 

    # 保存当前模型 
    def save(self, sess): 
        saver = tf.train.Saver() 
        if not os.path.exists(self._model_dir):
            os.mkdir(self._model_dir) 

        saver.save(sess, os.path.join(self._model_dir, self._name)) 

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
        print("* name:", self._name) 
        print("* learn_rate :", self._learn_rate) 
        print("* optimizer :", self._optimizer) 
        print("* input :", self._input_struct.shape) 
        print("* outputs :", self._output_label.shape) 
        print("* restore_last:", self._restore_last) 
        print("* model_dir :", self._model_dir) 
        print("* description :", self._description) 
        print("* summary_flg :", self._need_summary) 
        print("**********Train End*********") 
        self._cnn_net.dump() 
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

if __name__ == "__main__": 
    # 设置网络参数 
    train_param = cfg.TrainParams(input_struct = tf.placeholder(dtype = tf.float32, shape = INPUT_SHAPE, name = 'input_data') 
                                , output_label = tf.placeholder(dtype = tf.float32, shape = OUTPUT_SHAPE, name = 'output_data') 
                                , is_test = tf.placeholder(dtype = tf.bool, name = 'is_test') 
                                , cur_step = tf.placeholder(dtype = tf.int32, name = 'cur_step') 
                                , learn_rate = LEARN_RATE 
                                , name = TRAIN_NET_NAME
                                , description = TRAIN_NET_DESCRIPTION 
                                , restore_last = FLAG_RESTORE_LAST 
                                , need_summary = FLAG_OPEN_SUMMARY 
                                , optimizer = TRAIN_LOSS_FUNC) 

    cnn_param = cfg.CNNParams(input_struct = train_param.input_struct 
                            , name = CNN_NAME 
                            , is_test = train_param.is_test 
                            , cur_step = train_param.cur_step) 

    fcn_param = cfg.FCNParams(input_struct = None 
                            , net_shape = FCN_NET_SHAPE 
                            , name = FCN_NAME 
                            , activate_ffc = FCN_FINAL_ACTIVATE_FUNC) 

    # 构造网络 
    cnn_net = TrainNet(train_param, fcn_param, cnn_param) 
    # 读入数据： 
    mnist = input_data.read_data_sets("./", one_hot = True) 
    # feed & train: 
    with tf.Session(graph = cnn_net._graph) as sess:
        # training data: 
        sess.run(cnn_net.init())
        cnn_net.restore(sess) 
        for i in range(80000): 
            xs, ys = mnist.train.next_batch(100) 
            cnn_net.train(sess, xs, ys, i) 
            loss_value, _, _ =cnn_net.refresh_state(sess, xs, ys, i) 
            if i % 10 == 0:
                print("After %d training step(s), loss on training batchis %g" % (i, loss_value)) 

        cnn_net.dump() 
        cnn_net.save(sess) 
        sess.close()

