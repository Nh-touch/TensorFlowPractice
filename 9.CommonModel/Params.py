# TensorFlow Param配置类 
import tensorflow as tf 

# 常量 __XXX__ 
# 内部变量或方法 _xxx 
# 公开变量或方法 xxx 
# 参数信息基类

class Param(object): 
    #------------默认函数定义-----------# 
    # 类基本成员 初始化函数 
    def __init__(self , input_struct , name = 'default'): 
        # 通用网络参数信息 
        self.input_struct = input_struct 
        self.graph = None 
        if input_struct is not None: 
            self.graph = input_struct.graph
            
        self.name = name 

    # 私有函数定义 
    def __check(self, data): 
        pass 

    def set_input_struct(self, input_struct): 
        self.input_struct = input_struct 
        if input_struct is not None: 
            self.graph = input_struct.graph 

# 递归网络参数信息类 
class RNNParams(Param): 
    #------------默认函数定义-----------# 
    # 类基本成员 初始化函数 
    def __init__(self
               , input_struct 
               , name = 'default' 
               , keep_prob = 1.0 
               , layer_num = 2 
               , cell_num = 512 
               , cell_builder = tf.contrib.rnn.BasicLSTMCell): 
        Param.__init__(self, input_struct, name) 

        # RNN网络参数 
        self.frame_size   = input_struct.shape[2] if (len(input_struct.shape) == 3) else frame_size 
        self.layer_num    = layer_num 
        self.cell_num     = cell_num 
        self.cell_builder = cell_builder 
        self.keep_prob    = keep_prob 

    # 私有函数定义 
    def __check(self, data): 
        pass 

# 全连接网络参数信息类 
class FCNParams(Param): 
    #------------默认函数定义-----------# 
    # 类基本成员 初始化函数 
    def __init__(self 
               , input_struct 
               , net_shape 
               , name = 'default' 
               , keep_prob = 1.0 
               , activate_mfc = tf.nn.tanh 
               , activate_ffc = None 
               , initialier_w = tf.initializers.truncated_normal(0.1) 
               , initialier_b = tf.initializers.constant(0.1)): 
        Param.__init__(self, input_struct, name) 

        # 全连接网络(FullyConnected)网络参数 
        # net shape 从input获取作为shape[0],netshape列表中的第一元素作为shape1 
        self.net_shape      = net_shape 
        self.activate_mfc   = activate_mfc 
        self.activate_ffc   = activate_ffc 
        self.keep_prob      = keep_prob 
        self.initialier_w   = initialier_w 
        self.initialier_b   = initialier_b 

    # 私有函数定义 
    def __check(self, data): 
        pass 

# 训练网络用的参数配置 一些开关、优化函数等 
class TrainParams(Param): 
    #------------默认函数定义-----------# 
    # 类基本成员 初始化函数 
    def __init__(self 
               , input_struct 
               , output_label 
               , name = 'default' 
               , learn_rate = 0.001 
               , optimizer = tf.train.GradientDescentOptimizer 
               , model_dir = '.' 
               , restore_last = False 
               , description = '[]' 
               , need_summary = True): 
        Param.__init__(self, input_struct, name) 

        # 网络训练参数(TrainNet)网络参数 
        self.learn_rate     = learn_rate 
        self.optimizer      = optimizer 
        self.output_label   = output_label 
        self.model_dir      = model_dir 
        self.restore_last   = restore_last 
        self.description    = description 
        self.need_summary   = need_summary 

    # 私有函数定义 
    def __check(self, data): 
        pass

