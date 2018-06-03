import tensorflow as tf 
from helpers 
import lazy_property 
import RNN as rnn 
import FCN as fcn 
import os 
import time 
# 特定网络定义 
class RFNN(object): 
    #------------默认函数定义-----------# 
    def __init__(self, train_param, fcn_param = None, rnn_Param = None): 
        # param 
        self.__reset_param(train_param) 

        # build net 
        self._rnn_net = rnn.RNN(rnn_Param) 

        # transpose 
        output_transposed = None 
        with self._graph.as_default(): 
            output_transposed = tf.transpose(self._rnn_net.outputs_struct, [1, 0, 2]) 

        # link net 
        self.__link_two_net(output_transposed[-1], fcn_param) 
        self._fcn_net = fcn.FCN(fcn_param) 

        # get final output 
        self._output = self._fcn_net.outputs_struct 

        # set Learning Rate 
        self.__generate_learn_rate() 

        # generate loss/accuracy optimizer 
        self._loss 
        self._accuracy 
        self._optimizer 

        # prepare summary 
        if self._need_summary: 
            self.__generate_summary() 

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
        self._optimizer_fn = param.optimizer 
        self._output_label = param.output_label 
        self._model_dir = param.model_dir + '/' + self._name 
        self._restore_last = param.restore_last 
        self._description = param.description 
        self._need_summary = param.need_summary 
        self._writer = None 
        self._is_test = param.is_test 
        self._learn_decay = param.learn_decay 
        self._moving_avg = param.moving_avg 
        self._regularizer = param.regularizer 

    def __link_two_net(self, out_port, param): 
        param.set_input_struct(out_port) 

    def __generate_learn_rate(self): 
        with self._graph.as_default(): 
            self._global_step = tf.Variable(0, trainable = False) 
            if self._learn_decay['enable'] == True: 
                self._learn_rate = tf.train.exponential_decay(self._learn_decay['orig_rate'] 
                                                            , self._global_step 
                                                            , self._learn_decay['steps_for_decay'] 
                                                            , self._learn_decay['decay_rate'] 
                                                            , staircase = True) 

    def __generate_summary(self): 
        if not os.path.exists(self._model_dir): 
            os.mkdir(self._model_dir) 

        self._writer = tf.summary.FileWriter(self._model_dir, self._graph) 

    #------------外部接口定义-----------# 
    # 初始化当前网络 
    def init(self): 
        with self._graph.as_default(): 
            return tf.global_variables_initializer() 

    # 保存当前模型 建议每经过一定的迭代数后 保存一次模型结果 
    def save(self, sess): 
        saver = tf.train.Saver() 
        if not os.path.exists(self._model_dir): 
            os.mkdir(self._model_dir) 

        saver.save(sess, os.path.join(self._model_dir, self._name), global_step = self._global_step) 

    # 恢复模型 
    def restore(self, sess, is_test = False): 
        if self._restore_last: 
            if not os.path.exists(self._model_dir): 
                return ckpt = tf.train.get_checkpoint_state(self._model_dir)

        if ckpt and ckpt.model_checkpoint_path: 
            if self._moving_avg['enable'] == True and is_test == True: 
                ema = tf.train.ExponentialMovingAverage(self._moving_avg['orig_rate']) 
                ema_restore = ema.variables_to_restore() 
                saver = tf.train.Saver(ema_restore) 
            else: 
                saver = tf.train.Saver() 
                
            restore_global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1] 

        try: 
            saver.restore(sess, ckpt.model_checkpoint_path) 
            self._global_step = restore_global_step 
        except: 
            self._global_step = restore_global_step 

    # 训练结果可视化 
    def visualize(self): 
        pass 

    # 打印当前模型信息 
    def dump(self): 
        print("**********Train Begin*******") 
        print("* name :", self._name) 
        print("* learn_rate :", self._learn_rate) 
        print("* optimizer :", self._optimizer_fn) 
        print("* input :", self._input_struct.shape) 
        print("* outputs :", self._output_label.shape) 
        print("* restore_last:", self._restore_last) 
        print("* model_dir :", self._model_dir) 
        print("* description :", self._description) 
        print("* summary_flg :", self._need_summary) 
        print("* learn_decay :", self._learn_decay['enable']) 
        print("* mov_avg_flg :", self._moving_avg['enable']) 
        print("* orig_step :", self._global_step) 
        print("**********Train End*********") 
        self._rnn_net.dump() 
        self._fcn_net.dump() 

    # 当前网络的损失函数(这里的lazy_property为了防止多次创建) 放到专门的loss类中，因为是针对整个网络而言的 
    @lazy_property 
    def _loss(self): 
        with self._graph.as_default(): 
            with tf.name_scope("loss_layer"): 
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self._output, labels = self._output_label), name = "loss_ce") 

                if self._regularizer != None: 
                    loss = loss + tf.add_n(tf.get_collection('losses')) 
                    return loss 

    # 当前网络的准确率函数 acc类，针对不同的网络类型而言,可能放到外部的组合网络的类中实现 
    @lazy_property 
    def _accuracy(self): 
        with self._graph.as_default(): 
            with tf.name_scope("accuracy_layer"): 
                correct_pred = tf.equal(tf.argmax(self._output, 1), tf.argmax(self._output_label, 1)) 
                return tf.reduce_mean(tf.cast(correct_pred, tf.float32), name = "accuracy_op") 

    # 当前网络的梯度下降算法 loss类 
    @lazy_property 
    def _optimizer(self): 
        with self._graph.as_default(): 
            tf.summary.scalar("loss", self._loss) 
            tf.summary.scalar("accuracy", self._accuracy) 

            if self._moving_avg['enable'] == False: 
                return self._optimizer_fn(self._learn_rate).minimize(loss = self._loss , global_step = self._global_step) 

            # set moving avg 
            tmp_optimizer = self._optimizer_fn(self._learn_rate).minimize(loss = self._loss , global_step = self._global_step) 
            ema = tf.train.ExponentialMovingAverage(self._moving_avg['orig_rate'], self._global_step) ema_op = ema.apply(tf.trainable_variables()) 
            with tf.control_dependencies([tmp_optimizer, ema_op]): 
                return tf.no_op(name = 'train') 

    # Summary 
    @lazy_property 
    def _summarier(self): 
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

