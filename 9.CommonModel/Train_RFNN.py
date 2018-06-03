# TensorFlow TrainNet 
import tensorflow as tf 
import RFNN as rfnn 
import Params as cfg 
from numpy.random 
import RandomState 
import pandas as bPd 
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler 
from sklearn.cross_validation import train_test_split 
import numpy as np 
import random 
from StpredConfig import * 
# 特定网络定义 
class TrainNet(rfnn.RFNN): 
    #------------默认函数定义-----------# 
    def __init__(self, train_param, fcn_param = None, rnn_Param = None): 
        rfnn.RFNN.__init__(self, train_param, fcn_param, rnn_param) 
        
    def __del__(self): 
        rfnn.RFNN.__del__() 

    #------------外部接口定义-----------# 
    # 训练当前网络 
    def train(self, sess, train_input, train_output, step): 
        _ = sess.run([self._optimizer], feed_dict = {self._input_struct: train_input, self._output_label: train_output, self._is_test: False}) 

    # 获取当前网络损失 保存Summary 
    def refresh_state(self, sess, train_input, train_output, is_test, step): 
        loss, accuracy, pred, summary = sess.run([self._loss, self._accuracy, self._output, self._summarier]
                                               , feed_dict = {self._input_struct: train_input, self._output_label: train_output, self._is_test: is_test}) 

        if self._writer is not None: 
            self._writer.add_summary(summary, global_step = step) 

        return [loss, accuracy, pred] 

# read data from file name 
def read_data(fname): 
    with open(fname) as f: 
        content = f.readlines() content = [x.strip() for x in content] 
        content = [content[i].split() for i in range(len(content))] 
        content = np.array(content) content = np.reshape(content, [-1, ]) 
        return content 

# make vocabulary dictionary 
def build_dataset(words): 
    count = collections.Counter(words).most_common() 
    dictionary = dict() 
    for word, _ in count: 
        dictionary[word] = len(dictionary) 
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
    return dictionary, reverse_dictionary 

#--------------Read Data & Generate Batch --------------------# 
def generate_batch(data_x, data_y): 
    data_len = len(data_y) 
    rand_timestep = random.randint(5, 50) 
    rand_estimate_len = random.randint(1, 20) 
    output_x = [] 
    output_y = [] 
    for i in range(0, TRAIN_MINI_BATCH_SIZE): 
        rand_index = random.randint(rand_timestep - 1, data_len - rand_estimate_len) 

    for j in range(rand_index - rand_timestep + 1, rand_index + 1): 
        record = data_x[j] 
        new_record = record.tolist() + [rand_estimate_len] 
        #print(record.type) 
        output_x.append(new_record) 
        #print(output_x) 
        y_index = rand_index + rand_estimate_len - 1 
        record_y = np.array(data_y[y_index:y_index + 1]) 
        output_y.append(record_y.tolist()) 

    batch_x = np.reshape(np.array(output_x), [TRAIN_MINI_BATCH_SIZE, rand_timestep, RNN_FRAME_SIZE]) 
    batch_y = np.reshape(np.array(output_y), [TRAIN_MINI_BATCH_SIZE, 20]) 
    #print(batch_x, batch_y) 
    return [batch_x, batch_y] 

def generate_test_batch(data_x, data_y): 
    data_len = len(data_y) 
    rand_timestep = random.randint(20, 50) 
    rand_estimate_len = random.randint(1, 20) 
    output_x = [] 
    output_y = [] 
    for i in range(0, TRAIN_MINI_BATCH_SIZE): 
        rand_index = random.randint(rand_timestep - 1, data_len - rand_estimate_len) 

    for j in range(rand_index - rand_timestep + 1, rand_index + 1): 
        record = data_x[j] 
        new_record = record.tolist() + [rand_estimate_len] 
        #print(record.type) 
        output_x.append(new_record) 
        #print(output_x) 
        y_index = rand_index + rand_estimate_len - 1 
        record_y = np.array(data_y[y_index:y_index + 1]) 
        output_y.append(record_y.tolist()) 
        batch_x = np.reshape(np.array(output_x), [TRAIN_MINI_BATCH_SIZE, rand_timestep, RNN_FRAME_SIZE]) 
        batch_y = np.reshape(np.array(output_y), [TRAIN_MINI_BATCH_SIZE, 20]) 

    # print(batch_x, batch_y) 
    return [batch_x, batch_y, rand_timestep, rand_estimate_len] 

g_Column_name = ["OpenPrice", "ClosePrice", "ChangePrice", "ChangeScale", "MinPrice", "MaxPrice", "DealMount",
                "DealPrice", "HandChange", "Time2Number", "HistoryDate", "-9", "-8", "-7", "-6", "-5", "-4", "-3"
                , "-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"] 

if __name__ == "__main__":
    # 配置训练网络learn rate decay功能 
    learn_rate_decay = dict(enable = True 
                          , orig_rate = TRAIN_ORIG_LEARN_RATE 
                          , steps_for_decay = TRAIN_STEPS_4_DECAY 
                          , decay_rate = TRAIN_LEARN_DECAY_RATE) 

    # 配置测试网络moving average功能 
    moving_average = dict(enable = True , orig_rate = TRAIN_MOVING_AVG_DECAY) 
    # 设置网络参数 
    train_param = cfg.TrainParams(input_struct = tf.placeholder(dtype = tf.float32, shape = INPUT_SHAPE, name = 'input_data') 
                                , output_label = tf.placeholder(dtype = tf.float32, shape = OUTPUT_SHAPE, name = 'output_data') 
                                , learn_rate = LEARN_RATE 
                                , name = TRAIN_NET_NAME 
                                , description = TRAIN_NET_DESCRIPTION 
                                , restore_last = FLAG_RESTORE_LAST 
                                , need_summary = FLAG_OPEN_SUMMARY 
                                , optimizer = TRAIN_LOSS_FUNC 
                                , is_test = tf.placeholder(dtype = tf.bool, name = 'is_test') 
                                , learn_decay = learn_rate_decay 
                                , moving_avg = moving_average 
                                , regularizer = tf.contrib.layers.l2_regularizer(TRAIN_L2_REGULAR_RATE)) 

    rnn_param = cfg.RNNParams(input_struct = train_param.input_struct 
                            , name = RNN_NAME 
                            , cell_num = RNN_CELL_NUM 
                            , cell_builder = RNN_CELL_BUILDER 
                            , layer_num = RNN_LAYER_NUM 
                            , is_test = train_param.is_test 
                            , regularizer = train_param.regularizer) 

    fcn_param = cfg.FCNParams(input_struct = None 
                            , net_shape = FCN_NET_SHAPE 
                            , name = FCN_NAME 
                            , activate_ffc = FCN_FINAL_ACTIVATE_FUNC 
                            , is_test = train_param.is_test 
                            , regularizer = train_param.regularizer) 

    # 构造网络 
    stpred_net = TrainNet(train_param, fcn_param, rnn_param) 
    stpred_net.dump() 

    # 读取数据 
    # Read Data(From Web) 
    dir = "D:\Stock_000671.csv" 
    tData = bPd.read_csv(dir, encoding = "ISO-8859-1") 
    # ============================================================================= # 
    ############### Data Validation ##################### 
    # ============================================================================= # 
    Replace NAN 
    tData = tData.replace
    =============================================================== # 
    ############### Data Preproces ##################### 
    # =========================================================================
    al = 0 
    loss_total = 0 
    tf.global_variables_initializer().run() 
    raw_data_size = 3000 
    for i in range(800000): 
        batch_x, batch_y = generate_batch(x_train[0:raw_data_size], y_train[0:raw_data_size]) 
        # start to train 
        stpred_net.train(sess, batch_x, batch_y, i) 
        loss, _, layer_graph = stpred_net.refresh_state(sess, batch_x, batch_y, True, i) 
        loss_total += loss 
        #【间断性数据拟合检查】 
        if i % 2 == 0 and i > 0: 
            batch_test_x, batch_test_y, rnd_timestep, rnd_estimatelen = generate_test_batch(x_train[raw_data_size:], y_train[raw_data_size:]) 
            _, ac, pred= stpred_net.refresh_state(sess, batch_test_x, batch_test_y, True, i) 
            #print(rnn_output) 
            # 计算交叉熵： 
            #total_cross_entropy = sess.run( 
            # cross_en 0 stpred_net.save(sess)

