# TensorFlow 全连接网络FCN类
import tensorflow as tf 
from helpers import lazy_property 
import Params as cfg 

# 常量 __XXX__ 
# 内部变量或方法 _xxx 
# 公开变量或方法 xxx 
# 定义参数 

class FCN(object): 
    #------------默认函数定义-----------# 
    # 类基本成员 初始化函数 
    def __init__(self, param): 
        # param 
        self.__reset_param(param) 

        # init operator val 
        self._refresh_count = 0 
        self.__create_fcn() 

    #------------内部函数定义-----------# 
    def __reset_param(self, param): 
        # Set Parameters 
        self._input_struct = param.input_struct 
        self._keep_prob    = param.keep_prob 
        self._graph        = param.graph 
        self._name         = param.name 
        self._net_shape    = param.net_shape 
        self._activate_mfc = param.activate_mfc 
        self._activate_ffc = param.activate_ffc 
        self._initialier_w = param.initialier_w 
        self._initialier_b = param.initialier_b 

    # 创建外部输入参量 
    def __create_palceholders(self): 
        pass 

    # 创建权重偏置等参数矩阵 
    def __create_matrix(self, name, shape, initializer, trainable = True): 
            return tf.get_variable(name         = name 
                                 , shape        = shape 
                                 , dtype        = tf.float32 
                                 , initializer  = initializer 
                                 , trainable    = trainable) 

    # 创建tensorflow rnn 训练图: 
    # 输入inputs 
    # 返回ouputs(依据net_shape列表中最后数字) 
    def __create_fcn(self): 
        # 创建基本fcn层 
        def add_layer(input_tensor, output_size, layer_name, activate_func = None, is_need_dropout = False): 
            # raw formular y = ax + b 
            w1 = self.__create_matrix(name          = 'weights_' + layer_name 
                                    , shape         = [input_tensor.get_shape()[1].value, output_size] 
                                    , initializer   = self._initialier_w) 

            b1 = self.__create_matrix(name          = 'biases_' + layer_name 
                                    , shape         = [1, output_size] 
                                    , initializer   = self._initialier_b) 

            layer_output = tf.matmul(input_tensor, w1) + b1 

            # add drop out 
            if is_need_dropout: 
                layer_output = tf.nn.dropout(layer_output, self._keep_prob) 

            # TODO batch normalization 

            # activete 
            if activate_func is not None: 
                layer_output = activate_func(layer_output) 

            return layer_output 

        with self._graph.as_default(): 
            with tf.variable_scope(self._name + '_' + str(self._refresh_count)): 
                # 根据net_shape创建多层fcn 
                layer_output = self._input_struct 
                for i in range(0, len(self._net_shape)): 
                    activate_func = self._activate_mfc 
                    is_need_dropout = True 
                    if (i == (len(self._net_shape) - 1)): 
                        activate_func = self._activate_ffc 
                        is_need_dropout = False 

                    layer_output = add_layer(layer_output 
                                           , self._net_shape[i] 
                                           , 'hidden_layer%d' % i 
                                           , activate_func 
                                           , is_need_dropout) 
                # save outputs 
                self._outputs = layer_output 

            # 权重、偏置 Summary 
            for item in tf.trainable_variables(): 
                tf.summary.histogram(item.name, item) 

    #------------外部接口定义-----------# 
    # 刷新当前模型配置 
    def refresh_config(self, param): 
        # param 
        self.__reset_param(param) 

        # init operator val 
        self._refresh_count += 1 
        self.__create_fcn() 

    # 打印当前模型信息 
    def dump(self): 
        print("**********FCN Begin*******") 
        print("* name           :", self._name + '_' + str(self._refresh_count)) 
        print("* net_shape      :", [self._input_struct.get_shape()[1].value] + self._net_shape) 
        print("* activate_mfc   :", self._activate_mfc)
        print("* activate_ffc   :", self._activate_ffc) 
        print("* keep_prob      :", self._keep_prob)
        print("* input          :", self._input_struct.shape) 
        print("* outputs        :", self._outputs.shape)
        print("* variables      :") 
        with self._graph.as_default(): 
            for item in tf.trainable_variables(): 
                print('* ', item.name, ' ', item.shape, ' ', item.dtype) 
        print("**********FCN End*********") 

    #--------------属性定义------------# 
    # 输入tensor结构 
    @lazy_property 
    def input_struct(self): 
        return self._input_struct 

    # 输出tensor结构 
    @lazy_property
    def outputs_struct(self): 
        return self._outputs 

# test
##if __name__ == "__main__": 
##    params = cfg.FCNParams(input_struct = tf.placeholder(dtype = tf.float32, shape = [None, 10]) 
##                         , net_shape    = [256, 128, 20] 
##                         , activate_mfc = tf.nn.softmax 
##                         , activate_ffc = tf.nn.softmax) 
##
##    fcn = FCN(params) 
##    fcn.dump() 
##
##    graph = tf.Graph() 
##    with graph.as_default(): 
##        input = tf.placeholder(dtype = tf.float32, shape = [None, 50]) 
##
##    param2 = cfg.FCNParams(input_struct = input 
##                         , net_shape    = [256, 128, 20] 
##                         , activate_mfc = tf.nn.tanh 
##                         , activate_ffc = tf.nn.softmax) 
##
##    fcn.refresh_config(param2) 

