# TensorFlow RNN类
import tensorflow as tf
from helpers import lazy_property

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

    # 创建外部输入参量 RNN中不需要 
    def __create_palceholders(self):
        pass 
    # 创建权重 RNN中不需要 
    def __create_weight(self): 
        pass 
    # 创建偏置 RNN中不需要
    def __create_bias(self):
        pass
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

    # 当前RNN网络的损失函数(这里的lazy_property为了防止多次创建) 放到专门的loss类中，因为是针对整个网络而言的 
    @lazy_property
    def __loss(self):
        pass
    # 当前RNN网络的准确率函数 acc类，针对不同的网络类型而言,可能放到外部的组合网络的类中实现
    @lazy_property 
    def __accuracy(self): 
        pass
    # 当前RNN网络的梯度下降算法 loss类
    @lazy_property
    def __optimizer(self):
        pass
    #------------外部接口定义-----------#
    # 初始化当前网络
    def init(self):
        pass
    # 训练当前网络
    def train(self):
        pass
    # 训练结果可视化
    def visualize(self): 
        pass
    # 保存当前模型
    def save(self):
        pass
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

##if __name__ == "__main__":
##    params = Param(input_struct = tf.placeholder(dtype = tf.float32, shape = [None, None, 10]))
##    rnn = RNN(params)
##    rnn.dump() 
##    graph = tf.Graph()
##    with graph.as_default(): 
##        input = tf.placeholder(dtype = tf.float32, shape = [None, None, 50]) 
##    param2 = Param(input_struct = input)
##    rnn.refresh_config(param2)
##    # TODO 增加打印到tensorboard的功能和输出一些图像的功能:在外部进行，内部只打印基本的权重（外部收到网络的output可以进 行打印？根据需求）
##    # TODO 从可读性角度来看，没有连接的过程。最好加入这个在代码层面进行显式连接的过程 这个在外部封装的时候，实现linkto方法就行
##    rnn.dump()

