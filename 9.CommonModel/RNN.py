# TensorFlow RNN类
import tensorflow as tf
from helpers import lazy_property

class RNN(object): 
    #------------默认函数定义-----------# 
    # 类基本成员 初始化函数
    def __init__(self, param, input_struct = None):
        # param
        self.__reset_param(param)
        # init operator val 
        self._refresh_count         = 0
        self._outputs, self._states = self.__create_rnn() 

    #------------内部函数定义-----------#
    def __reset_param(self, param):
        # Set Parameters
        self._input_struct  = param.input_struct 
        self._frame_size    = param.frame_size
        self._layer_num     = param.layer_num
        self._cell_num      = param.cell_num 
        self._cell_builder  = param.cell_builder
        self._keep_prob     = param.keep_prob 
        self._graph         = param.graph 
        self._name          = param.name 

    # 创建tensorflow rnn 训练图: 
    # 输入inputs 
    # 返回[outputs列表，states列表]
    def __create_rnn(self): 
        # 创建基本rnn单元 
        def get_one_cell(): 
            basic_rnn_cell = self._cell_builder(num_units = self._cell_num) 

            # 添加dropout属性，如无需dropout，可将keep_prob设置为1.0，默认为无需dropout
            basic_rnn_cell = tf.contrib.rnn.DropoutWrapper(basic_rnn_cell, output_keep_prob = self._keep_prob)

            return basic_rnn_cell

        with self._graph.as_default():
            with tf.variable_scope(self._name + '_' + str(self._refresh_count)): 
                # 创建多层rnn 
                multi_rnn_layer = tf.contrib.rnn.MultiRNNCell([get_one_cell() for _ in range(self._layer_num)], state_is_tuple = True)

                # 创建初始state 根据inputs中的batch_size 
                init_state = multi_rnn_layer.zero_state(tf.shape(self._input_struct)[0], dtype=tf.float32)

                # 构建动态rnn 
                outputs, states = tf.nn.dynamic_rnn(cell = multi_rnn_layer
                                                  , inputs = self._input_struct
                                                  , initial_state = init_state) 

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
        self._refresh_count        += 1
        self._outputs, self._states = self.__create_rnn() 

    # 打印当前模型信息
    def dump(self):
        print("**********RNN Begin*******")
        print("* name           :", self._name + '_' + str(self._refresh_count)) 
        print("* frame_size     :", self._frame_size)
        print("* layer_num      :", self._layer_num)
        print("* cell_num       :", self._cell_num)
        print("* cell_builder   :", self._cell_builder.__name__)
        print("* keep_prob      :", self._keep_prob) 
        print("* input          :", self._input_struct.shape)
        print("* outputs        :", self._outputs.shape)
        print("* variables      :")
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

# test
##if __name__ == "__main__":
##    params = Param(input_struct = tf.placeholder(dtype = tf.float32, shape = [None, None, 10]))
##    rnn = RNN(params)
##    rnn.dump() 
##    graph = tf.Graph()
##    with graph.as_default(): 
##        input = tf.placeholder(dtype = tf.float32, shape = [None, None, 50]) 
##    param2 = Param(input_struct = input)
##    rnn.refresh_config(param2)
##    rnn.dump()

