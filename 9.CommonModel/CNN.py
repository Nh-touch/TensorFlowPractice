# TensorFlow 全连接网络CNN类
import tensorflow as tf
from helpers import lazy_property
import Params as cfg
import TFLayer as tl

class CNN(object): 
    #------------默认函数定义-----------# 
    # 类基本成员 初始化函数 
    def __init__(self, param):
        # param 
        self.__reset_param(param) 

        # init operator val 
        self._refresh_count = 0
        self._outputs = self._input_struct
        self._tl      = tl.LayerProvider()

    #------------内部函数定义-----------# 
    def __reset_param(self, param): 
        # Set Parameters 
        self._input_struct = param.input_struct 
        self._keep_prob    = param.keep_prob
        self._graph        = param.graph 
        self._name         = param.name
        self._initialier_w = param.initialier_w
        self._initialier_b = param.initialier_b 
        self._is_test      = param.is_test 
        self._cur_step     = param.cur_step 

    # 创建tensorflow cnn 训练图:
    #------------外部接口定义-----------# 
    # 对于CNN网络，不进行一次性创建，提供接口供外部进行调用来创建网络。 
    def add_unit(self, name, conv_filter, conv_stride, pool_ksize = None, pool_stride = None, activate_func = None, norm = False): 
        # 创建卷积层 
        with self._graph.as_default(): 
            with tf.variable_scope(self._name + '_' + name + '_' + str(self._refresh_count)): 
                layer_output = self._outputs 
                n_count = 0 
                for item in zip(conv_filter, conv_stride): 
                    n_count += 1
                    layer_output = self._tl.conv2d_layer(name     = 'conv' + str(item[0][0]) + 'x' + str(item[0][0]) + 'x' + str(item[0][3]) + '_' + str(n_count)
                                                       , layer_in = layer_output
                                                       , w_shape  = item[0]
                                                       , b_shape  = [item[0][3]] 
                                                       , strides  = item[1])
                    # activate 
                    if activate_func is not None:
                        layer_output = self._tl.activate_layer(name         = 'activate'
                                                             , layer_in     = layer_output
                                                             , fun_activate = activate_func) 

                    # pooling 
                    if pool_stride is not None: 
                        for item in zip(pool_ksize, pool_stride): 
                            layer_output = self._tl.maxpooling_layer(name     = 'maxpool'
                                                                   , layer_in = layer_output
                                                                   , ksize    = item[0]
                                                                   , strides  = item[1]) 

                    # batch norm
                    if norm:
                        layer_output = self._tl.batch_norm_layer(name = 'batchnorm'
                                                               , layer_in = layer_output
                                                               , step_in = self._cur_step
                                                               , is_test_in = self._is_test)

                    self._outputs = layer_output 

    # 平铺输出结果，并进行Summary 便于连接FCN等其他网络 
    def flat_output(self): 
        # 平铺数据维度 
        shape_flatted = 1; 
        for item in self._outputs.get_shape()[1:]: 
            shape_flatted = shape_flatted * int(item) 

        self._outputs = tf.reshape(self._outputs, [-1, shape_flatted]) 

        # 权重、偏置 Summary 
        for item in tf.trainable_variables(): 
            tf.summary.histogram(item.name, item) 

    # 打印当前模型信息 
    def dump(self): 
        print("**********CNN Begin*******") 
        print("* name :", self._name + '_' + str(self._refresh_count)) 
        print("* keep_prob :", self._keep_prob)
        print("* input :", self._input_struct.shape) 
        print("* outputs :", self._outputs.shape) 
        print("* variables :") 
        with self._graph.as_default(): 
            for item in tf.trainable_variables(): 
                print('* ', item.name, ' ', item.shape, ' ', item.dtype) 
        print("**********CNN End*********") 

    #--------------属性定义------------# 
    # 输入tensor结构 
    @lazy_property 
    def input_struct(self): 
        return self._input_struct 

    # 输出tensor结构 
    @lazy_property 
    def outputs_struct(self): 
        return self._outputs

if __name__ == "__main__": 
    params = cfg.CNNParams(input_struct = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28, 3]) 
                         , is_test = tf.placeholder(dtype = tf.bool) 
                         , cur_step = tf.placeholder(dtype = tf.int32)) 
    
    cnn = CNN(params) 
    cnn.add_unit(name = 'conv5x5x32' 
               , conv_filter = [[5, 5, 3, 32], [3, 3, 32, 64]] 
               , conv_stride = [[1, 1, 1, 1] , [1, 1, 1, 1]]
               , pool_ksize = [[1, 2, 2, 1]] 
               , pool_stride = [[1, 2, 2, 1]] 
               , activate_func = tf.nn.relu)

    cnn.dump()

