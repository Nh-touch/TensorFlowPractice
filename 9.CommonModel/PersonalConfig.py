import tensorflow as tf 
# 定义参数 
##############网络结构########################### # 
# RNN -> FCN # ################################################# 
##############子网络定义######################### 
# RNN 0 
# 子网络名称 
RNN_NAME = 'STPRED_T1-RNN' 
# RNN元素数量 
RNN_CELL_NUM = 512 
# RNN 类型 
RNN_CELL_BUILDER = tf.contrib.rnn.BasicLSTMCell 
# RNN层数 
RNN_LAYER_NUM = 2
# RNN时间序列(Random) 
RNN_SEQUENCE_NUM = 3 
# RNN帧大小 
RNN_FRAME_SIZE = 11 
# RNN输出分类信息大小 
RNN_OUTPUT_SIZE = 20 
# FCN0 
# 子网络名称 
FCN_NAME = 'STPRED_T1-FCN' 
# 网络结构（从第二层开始 ） 
FCN_NET_SHAPE = [256, 20] 
# 最后一层激活函数 
FCN_FINAL_ACTIVATE_FUNC = tf.nn.softmax 
######## ######训练网络定义####################### 
# 网络名称 
TRAIN_NET_NAME = 'STPRED_T1' 
# 网络组成描述
TRAIN_NET_DESCRIPTION = '[RNN]->[FCN]' 
# 网络输出的分类信息大小 
TRAIN_NET_OUTPUT_SIZE = 20 
# 输入数据结构 
INPUT_SHAPE = [None, None, RNN_FRAME_SIZE] 
# 输出数据结构 
OUTPUT_SHAPE = [None, TRAIN_NET_OUTPUT_SIZE] 
# 学习速率 
LEARN_RATE = 0.01 
# 从上次继续训练 
FLAG_RESTORE_LAST = True 
# 是否打开统计 
FLAG_OPEN_SUMMARY = True 
# 梯度下降函数 
TRAIN_LOSS_FUNC = tf.train.RMSPropOptimizer 
# 最小的Batch大小 
TRA IN_MINI_BATCH_SIZE = 25 
# 学习率自动衰减相关配置 采用以下配置后，LearnRate失效 
TRAIN_ORIG_LEARN_RATE = 0.01 
TRAIN_ STEPS_4_DECAY = 10000
TRAIN_LEARN_DECAY_RATE = 0.98 
# 滑动平均相关配置，用于提升测试集准确率 
TRAIN_MOVING_AVG_D ECAY = 0.98 
# 正则化参数 
TRAIN_L2_REGULAR_RATE = 0.001 

