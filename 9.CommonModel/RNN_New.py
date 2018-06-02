# TensorFlow RNN类 
import tensorflow as tf 
from helpers 
import lazy_property 
import Params as cfg
import TFLayers as tl 
# 常量 __XXX__ 
# 内部变量或方法 _xxx 
# 公开变量或方法 xxx 
# 定义参数 
##flags = tf.app.flags 
##flags.DEFINE_integer("stock_count", 100, "Stock count [100]") 
class RNN(object): 
    #------------默认函数定义-----------# 
    # 类基本成员 初始化函数 
    def __init__(self, param, input_struct = None): 
        # param 
        self.__reset_param(param) 

        # init operator val 
        self._refresh_count = 0 

        # get layer provider 
        self._lp = tl.LayerProvider() 
        self._outputs, self._states = self.__create_rnn() 

    #------------内部函数定义-----------# 
    def __reset_param(self, param): 
        # Set Parameters 
        self._input_struct = param.input_struct 
        self._frame_size = param.frame_size 
        self._layer_num = param.layer_num 
        self._cell_num = param.cell_num 
        self._cell_builder = param.cell_builder 
        self._keep_prob = param.keep_prob 
        self._graph = param.graph 
        self._name = param.name 
        self._is_test = param.is_test 
        self._regularizer = param.regularizer 

    # 创建tensorflow rnn 训练图: 
    # 输入inputs 
    # 返回[outputs列表，states列表] 
    def __create_rnn(self): 
        with self._graph.as_default(): 
            # 构建动态rnn 
            outputs, states = self._lp.rnn_lstm_layer(name = self._name + '_' + str(self._refresh_count)
                                                    , layer_in = self._input_struct
                                                    , cell_num = self._cell_num
                                                    , layer_num = self._layer_num) 
            # 权重、偏置 Summary 
            for item in tf.trainable_variables(): 
                tf.summary.histogram(item.name, item) 
                
        return [outputs, states] 

    #------------外部接口定义-----------# 
    # 刷新当前模型配置 
    def refresh_config(self, param): 
        # param 
        self.__reset_param(param) 

        # init operator val 
        self._refresh_count += 1 
        self._outputs, self._states = self.__create_rnn() 

    # 打印当前模型信息 
    def dump(self): 
        print("**********RNN Begin*******") 
        print("* name :", self._name + '_' + str(self._refresh_count)) 
        print("* frame_size :", self._frame_size) 
        print("* layer_num :", self._layer_num) 
        print("* cell_num :", self._cell_num) 
        print("* cell_builder:", self._cell_builder.__name__) 
        print("* keep_prob :", self._keep_prob) 
        print("* input :", self._input_struct.shape) 
        print("* outputs :", self._outputs.shape) 
        print("* variables :") 
        with self._graph.as_default(): 
            for item in tf.trainable_variables(): 
                print('* ', item.name, ' ', item.shape, ' ', item.dtype) 
        print("**********RNN End*********") 

    #--------------属性定义------------# 
    # 当前RNN网络层数
    @lazy_property
    def layer_num(self): 
        return self._layer_num

    # 输入tensor结构
    @lazy_property
    def input_struct(self):
        return self._input_struct

    # 输出tensor结构 
    @lazy_property 
    def outputs_struct(self): 
        return self._outputs 

    # States Tensor结构 
    @lazy_property 
    def states_struct(self): 
        return self._states

