# TensorFlow 各层封装类 封装一些常用的层
import tensorflow as tf
class LayerProvider(object):
    #------------默认函数定义-----------#
    # 类基本成员 初始化函数
    def __init__(self): 
        pass 

    def __del__(self):
        pass
    #------------内部函数定义-----------# 
    # 创建权重偏置等参数矩阵 
    def __create_matrix(self, name, shape, initializer, trainable = True): 
        return tf.get_variable(name = name 
                             , shape = shape 
                             , dtype = tf.float32 
                             , initializer = initializer 
                             , trainable = trainable) 

    #------------外部接口定义-----------# 
    # wx + b 
    def basic_layer(self 
                  , name 
                  , layer_in 
                  , w_shape = None 
                  , b_shape = None 
                  , w_initializer = tf.initializers.truncated_normal(0.1) 
                  , b_initializer = tf.initializers.constant(0.1)):
        with tf.name_scope(name): 
            layer_out = layer_in 
            # wx 
            if w_shape is not None: 
                w_basic = self.__create_matrix(name = name + '_w' 
                                             , shape = w_shape 
                                             , initializer = w_initializer) 
                layer_out = tf.matmul(layer_out, w_basic) 
            # +b 
            if b_shape is not None: 
                b_basic = self.__create_matrix(name = name + '_b' 
                                             , shape = b_shape 
                                             , initializer = b_initializer) 
                layer_out = layer_out + b_basic 

            return layer_out 

    # activate 
    def activate_layer(self, name, layer_in, fun_activate): 
        with tf.name_scope(name): 
            if fun_activate is not None: 
                return fun_activate(layer_in, name = name) 

            return layer_in 
    # dropout 
    def dropout_layer(self, name, layer_in, keep_prob = 1.0): 
        with tf.name_scope(name): 
            return tf.nn.dropout(layer_in, keep_prob, name = name) 

    # batch norm 
    def batch_norm_layer(self, name, layer_in, step_in, is_test_in): 
        with tf.name_scope(name): 
            offset = tf.Variable(tf.zeros(layer_in.get_shape()[1:])) 
            scale = tf.Variable(tf.ones(layer_in.get_shape()[1:])) 
            bnepsilon = 1e-5 
            mean, variance = tf.nn.moments(x = layer_in, axes = [0]) 
            exp_moving_avg = tf.train.ExponentialMovingAverage(0.95, step_in) 
            def mean_var_with_update(): 
                new_moving_avg = exp_moving_avg.apply([mean, variance]) 
                with tf.control_dependencies([new_moving_avg]): 
                    return tf.identity(mean), tf.identity(variance) 

            if is_test_in == True: 
                mean = exp_moving_avg.average(mean) 
                variance = exp_moving_avg.average(variance) 
            else: 
                mean, variance = mean_var_with_update() 

            return tf.nn.batch_normalization(layer_in, mean, variance, offset, scale, bnepsilon) 
    # conv_2d 
    def conv2d_layer(self 
                   , name 
                   , layer_in 
                   , w_shape 
                   , b_shape 
                   , w_initializer = tf.initializers.truncated_normal(0.1) 
                   , b_initializer = tf.initializers.constant(0.1) 
                   , strides = [1, 1, 1, 1] 
                   , padding = 'SAME'): 
        with tf.name_scope(name): 
            w_conv = self.__create_matrix(name = name + '_w' 
                                        , shape = w_shape 
                                        , initializer = w_initializer) 
            b_conv = self.__create_matrix(name = name + '_b' 
                                        , shape = b_shape 
                                        , initializer = b_initializer) 
            return tf.nn.conv2d(input = layer_in
                              , filter = w_conv
                              , strides = strides
                              , padding = padding
                              , name = name) + b_conv 

    # max pooling 
    def maxpooling_layer(self 
                       , name 
                       , layer_in 
                       , ksize 
                       , strides = [1, 2, 2, 1] 
                       , padding = 'SAME'): 
        with tf.name_scope(name): 
            return tf.nn.max_pool(value = layer_in
                                , ksize = ksize
                                , strides = strides 
                                , padding = padding
                                , name = name) 

    # average pooling 
    def avgpooling_layer(self 
                       , name 
                       , layer_in 
                       , ksize 
                       , strides = [1, 2, 2, 1] 
                       , padding = 'SAME'): 
        with tf.name_scope(name): 
            return tf.nn.avg_pool(value = layer_in
                                , ksize = ksize
                                , strides = strides
                                , padding = padding
                                , name = name)

    # lstm 
    def rnn_lstm_layer(self 
                     , name 
                     , layer_in 
                     , cell_num 
                     , layer_num 
                     , keep_prob = 1.0): 
        # 创建基本rnn单元 
        def get_one_cell(): 
            basic_rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units = cell_num) 

            # 添加dropout属性，如无需dropout，可将keep_prob设置为1.0，默认为无需dropout 
            basic_rnn_cell = tf.contrib.rnn.DropoutWrapper(basic_rnn_cell, output_keep_prob = keep_prob) 

            return basic_rnn_cell
        with tf.variable_scope(name): 
            # 创建多层 rnn 
            multi_rnn_layer = tf.contrib.rnn.MultiRNNCell([get_one_cell() for _ in range(layer_num)], state_is_tuple = True) 
            # 创建初始state 根据inputs中的batch_size 
            init_state = multi_rnn_layer.zero_state(tf.shape(layer_in)[0], dtype=tf.float32) 
            # 构建动态rnn 
            layer_out, states = tf.nn.dynamic_rnn(cell = multi_rnn_layer
                                                , inputs = layer_in
                                                , initial_state = init_state)

            return [layer_out, states] 

    # gru 
    def rnn_gru_layer(self): 
        # 创建基本rnn单元 
        def get_one_cell(): 
            basic_rnn_cell = tf.contrib.rnn.BasicGRUCell(num_units = cell_num) 
            # 添加dropout属性，如无需dropout，可将keep_prob 设置为1.0，默认为无需dropout 
            basic_rnn_cell = tf.contrib.rnn.DropoutWrapper(basic_rnn_cell, output_keep_prob = keep_prob) 
            return basic_rnn_cell 

        with tf.variable_scope(name): 
            # 创建多层rnn 
            multi_rnn_layer = tf.contrib.rnn.MultiRNNCell([get_one_cell() for _ in range(layer_num)], state_is_tuple = True) 
            # 创建初始state 根据inputs中的batch_size 
            init_state = multi_rnn_layer.zero_state(tf.shape(layer_in)[0], dtype=tf.float32) 
            # 构建动态rnn 
            layer_out, states = tf.nn.dynamic_rnn( cell = multi_rnn_layer, inputs = self.layer_in, initial_state = init_state) 
            return [layer_out, states]

