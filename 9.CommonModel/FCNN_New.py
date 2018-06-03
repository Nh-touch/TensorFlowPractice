# TensorFlow 全连接网络FCN类 
import tensorflow as tf 
from helpers import lazy_property 
import Params as cfg 
import TFLayers as tl 

# 常量 __XXX__ 
# 内部变量或方法 _xxx 
# 公开变量或方法 xxx 
# 定义参数 
##flags = tf.app.flags 
##flags.DEFINE_integer("stock_count", 100, "Stock count [100]") 
class FCN(object): 
    #------------默认函数定义-----------# 
    # 类基本成员 初始化函数 
    def __init__(self, param): 
        # param 
        self.__reset_param(param) 

        # init operator val 
        self._refresh_count = 0 

        # get layer provider 
        self._lp            = tl.LayerProvider() 
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
        self._is_test      = param.is_test 
        self._regularizer  = param.regularizer 

    # 创建tensorflow rnn 训练图: 
    # 输入inputs 
    # 返回ouputs(依据net_shape列表中最后数字) 
    def __create_fcn(self): 
        # 创建基本fcn层 
        def add_layer(input_tensor, output_size, layer_name, activate_func = None, is_need_dropout = False): 
            # raw formular y = ax + b 
            layer_output = self._lp.basic_layer(name            = layer_name 
                                              , layer_in        = input_tensor 
                                              , w_shape         = [input_tensor.  get_shape()[1].value, output_size] 
                                              , b_shape         = [1, output_size] 
                                              , w_initializer   = self._initialier_w 
                                              , b_initializer   = self._initialier_b 
                                              , regularizer     = self._regularizer) 

            # add drop out 
            if is_need_dropout and self._is_test == False: 
                layer_output = self._lp.dropout_layer(name      = "fcn_dropout" 
                                                    , layer_in  = layer_output 
                                                    , keep_prob = self._keep_prob) 
            # TODO batch normalization 

            # activete 
            if activate_func is not None: 
                layer_output = self._lp.activate_layer(name         = "fcn_activate" 
                                                     , layer_in     = layer_output 
                                                     , fun_activate = activate_func) 

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
                #save outputs 
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
        print("* name :", self._name + '_' + str(self._refresh_count)) 
        print("* net_shape :", [self._input_struct.get_shape()[1].value] + self._net_shape) 
        print("* activate_mfc:", self._activate_mfc)
        print("* activate_ffc:", self._activate_ffc) 
        print("* keep_prob :", self._keep_prob) 
        print("* input :", self._input_struct.shape) 
        print("* outputs :", self._outputs.shape) 
        print("* variables :") 
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

