# -*- coding: utf-8 -*-
"""
this is a tensorflow test
"""
import pandas as bPd
import numpy as bNp
import tensorflow as tf
import random
import os
from numpy.random import RandomState
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from tensorflow.contrib import rnn
import ConfigData as cfg



# =============================================================================
# ############### Super Params Define #####################
# =============================================================================
# Basic
g_tLayerDimension = [256, 30, 20]
g_funActivateFunc = tf.nn.tanh
g_funLastActFunc  = tf.nn.softmax
g_nTotalSteps     = 8000000
g_nMiniBatchSize  = 25
g_nCheckRound     = 500
#Optimize
g_fOriLearnRate   = 0.001
g_nStep4RateDecay = 10000
g_fLearnDecayRate = 0.98
g_fKeepProb       = 1.0
g_bIsEnalbeBN     = True
g_fL2Regularation = 0.05
#Check
g_fExpectAccruacy = 1.65
# define Custmize Data
# Create DataSet: define column name
#g_Column_name = ['Team1Goal', 'Team1Los', 'Team1Yellow',
#                 'Team1Red', 'Team1WinRate', 'Team1PinRate', 'Team1LosRate', 'Team2Goal', 'Team2Los',
#                 'Team2Yellows', 'Team2Reds', 'Team2WinRate', 'Team2PinRate', 'Team2LosRate', 'Judge1Rate', 'Judge2Rate', 'Judge3Rate', '3', '1', '0']

## Old
#g_Column_name = ['Team1Goal', 'Team1Los', 'Team1Yellow', 'Team1Red', 'Team1WinRate', 'Team1PinRate', 'Team1LosRate', 'Team1GuessWin', 'Team1MatchCount',
#                 'Team2Goal', 'Team2Los', 'Team2Yellows', 'Team2Reds', 'Team2WinRate', 'Team2PinRate', 'Team2LosRate', 'Team2GuessWin', 'Team2MatchCount',
#                 'JudgeTrend', '3', '1', '0']

#g_Column_name = ["OpenPrice", "ClosePrice", "ChangePrice", "ChangeScale", "MinPrice", "MaxPrice", "DealMount", "DealPrice", "HandChange", 
#                 "HistoryDate", "Predict1", "Predict2", "Predict3"]

g_Column_name = cfg.g_Column_name

# define Continues!
g_bIsContinuesFromLast = True


# Variable for RNN
# hidden layer for RNN
g_rnn_HIDDEN_NERU_NUM = cfg.g_rnn_HIDDEN_NERU_NUM 
g_rnn_SEQUENCE_NUM = cfg.g_rnn_SEQUENCE_NUM
g_rnn_FRAME_SIZE = cfg.g_rnn_FRAME_SIZE
g_stock_id = cfg.g_stock_id
g_strModelDir = cfg.g_strModelDir

# =============================================================================
# ############### Define Functions #####################
# =============================================================================
def get_cov2d(inputTensor, filter):
    return tf.nn.conv2d(
        input   = inputTensor,
        filter  = filter,
        strides = [1, 1, 1, 1],
        padding = 'SAME')
    
def get_max_pool2x2(inputTensor):
    return tf.nn.max_pool(
        value   = inputTensor,
        ksize   = [1, 2, 2, 1],
        strides = [1, 2, 2, 1],
        padding = 'SAME')

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(
    name        = "weights", 
    shape       = shape,
    initializer = tf.truncated_normal_initializer(stddev = 1.0))

    if regularizer != None:
        tf.add_to_collection("losses", regularizer(weights))
    return weights

def get_biases_variable(shape):
    biases = tf.get_variable(
    name        = "biases", 
    shape       = shape,
    initializer = tf.constant_initializer(0.1))

    return biases

def get_batch_norm(inputTensor, curStep, isTestMode = False):
    offset     = tf.Variable(tf.zeros([inputTensor.get_shape()[1].value]))
    scale      = tf.Variable(tf.ones([inputTensor.get_shape()[1].value]))
    bnepsilon  = 1e-5

    mean, variance = tf.nn.moments(x = inputTensor, axes = [0])

    expMovingAvg   = tf.train.ExponentialMovingAverage(0.95, curStep)
    def mean_var_with_update():
        newMovingAvg = expMovingAvg.apply([mean, variance])
        with tf.control_dependencies([newMovingAvg]):
            return tf.identity(mean), tf.identity(variance)
    #fMean            = tf.cond(bIsTest, lambda: fMean, lambda: 0.0)
    #fVariance        = tf.cond(bIsTest, lambda: fVariance, lambda: 1.0)

    if isTestMode == True:
        mean = expMovingAvg.average(mean)
        variance = expMovingAvg.average(variance)
    else:
        mean, variance = mean_var_with_update()

    return tf.nn.batch_normalization(inputTensor, mean, variance, offset, scale, bnepsilon)

# -----------------------------------------------------
# def rnn related functions
# -----------------------------------------------------
def RNN(x, weight, bias): 
    # reshpae input to fit rnn_layer's need 
    #x = tf.reshape(x, [-1, g_rnn_SEQUENCE_NUM * g_rnn_FRAME_SIZE]) 
    #x = tf.split(x, g_rnn_SEQUENCE_NUM * g_rnn_FRAME_SIZE, 1)

    #x = tf.unstack(x, g_rnn_SEQUENCE_NUM, 1)

    # define rnn layer 
    rnn_layer = rnn.MultiRNNCell([rnn.BasicLSTMCell(g_rnn_HIDDEN_NERU_NUM, forget_bias = 1.0, activation = tf.nn.tanh), 
                                  rnn.BasicLSTMCell(g_rnn_HIDDEN_NERU_NUM, forget_bias = 1.0, activation = tf.nn.tanh)] , state_is_tuple=True)

    # link input to rnn_layer 
    #outputs, state = rnn.static_rnn(rnn_layer, x, dtype = tf.float32) 
    initial_state = rnn_layer.zero_state(g_nMiniBatchSize, dtype = tf.float32)
    outputs, state = tf.nn.dynamic_rnn(rnn_layer, x, initial_state = initial_state, dtype = tf.float32)
    outputs = tf.transpose(outputs, [1, 0, 2])
    #outputs, state = tf.nn.dynamic_rnn(rnn_layer, x, dtype = tf.float32) 

    return tf.add(tf.matmul(outputs[-1], weight), bias, name = "rnn_pred") 


def build_net(dx, dy, nTrainDataSize, bIsNormed, nLearnRate):
    # Function Define
    # 添加隐藏层的专用函数，入参：上一层的输出Tensor, 上一层的输出大小，本层输出的大小，layer的名称， 以及本层所使用的激活函数
    # 允许不定义激活函数的情况，不定义激活函数就用默认的激活函数，即y=x作为激活
    def add_layer(tInputs, nIn_size, nOut_size, strLayer_name, nDataSize, fActivation_function=None, bIsNormed=False, isNeedDropOut = False):
        tWeights = tf.Variable(tf.truncated_normal([nIn_size, nOut_size], stddev=1.0))
        tBiases = tf.Variable(tf.zeros([1, nOut_size]) + 0.1)
        tWxplusb = tf.matmul(tInputs, tWeights) + tBiases
        if isNeedDropOut:
            tWxplusb = tf.nn.dropout(tWxplusb, keep_prob)
        tOutputs = tWxplusb

        # BatchNorm
        if bIsNormed:
            tOutputs    = get_batch_norm(tOutputs, nIterations, bIsTesing)

        if fActivation_function is not None:
            tOutputs = fActivation_function(tWxplusb)

        # regularization add Loss collection
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(g_fL2Regularation / nDataSize)(tOutputs))
        #tf.summary.histogram(strLayer_name+'/outputs',tOutputs)
        return tOutputs

    #if bIsNormed:
    #    dx = get_batch_norm(dx, nIterations, bIsTesing)

    with tf.variable_scope("rnn_variables"):
        w_rnn2dnn = get_weight_variable([g_rnn_HIDDEN_NERU_NUM, 256], None)
        b_rnn2dnn = get_biases_variable([256])

    # get rnn results
    tCurLayer = RNN(dx, w_rnn2dnn, b_rnn2dnn)

    #x_firstTeam,x_secondTeam = tf.split(dx, [9, 12], 1)
    #print(x_firstTeam.shape)
    #print(x_secondTeam.shape)
    #with tf.variable_scope('seperateData1'):
    #    w_sepd_1 = get_weight_variable([9, 256], None)
    #    b_sepd_1 = get_biases_variable([256])
    #    h_sepd_1 = tf.nn.softmax(tf.matmul(x_firstTeam, w_sepd_1) + b_sepd_1)

    #with tf.variable_scope('seperateData2'):
    #    w_sepd_2 = get_weight_variable([12, 256], None)
    #    b_sepd_2 = get_biases_variable([256])
    #    h_sepd_2 = tf.nn.softmax(tf.matmul(x_secondTeam, w_sepd_2) + b_sepd_2)

    #x_collect = tf.concat([h_sepd_1, h_sepd_2], 1)
    #print(x_collect.shape)
    #x_image = tf.reshape(x_collect, [-1, 16, 16, 2])
    #with tf.variable_scope('conv2d1'):
    #    w_conv_1 = get_weight_variable([3, 3, 2, 16], None)
    #    b_conv_1 = get_biases_variable([16])
    #    h_conv_1 = tf.nn.relu(get_cov2d(x_image, w_conv_1) + b_conv_1)
    #    h_pool_1 = get_max_pool2x2(h_conv_1)
    #    h_pool_1_flat = tf.reshape(h_conv_1, [-1, 16 * 16 * 16]) # Prepare AllLinked Layer Shape


    #with tf.variable_scope('conv2d2'):
    #    w_conv_2 = get_weight_variable([3, 3, 16, 32], None)
    #    b_conv_2 = get_biases_variable([32])
    #    h_conv_2 = tf.nn.relu(get_cov2d(h_pool_1, w_conv_2) + b_conv_2)
    #    h_pool_2 = get_max_pool2x2(h_conv_2)

    #with tf.variable_scope('conv3d3'):
    #    w_conv_3 = get_weight_variable([3, 3, 32, 64], None)
    #    b_conv_3 = get_biases_variable([64])
    #    h_conv_3 = tf.nn.relu(get_cov2d(h_pool_2, w_conv_3) + b_conv_3)
    #    h_pool_3 = get_max_pool2x2(h_conv_3)

    # Apply Layers
    #tCurLayer = h_pool_1_flat
    print(tCurLayer.shape)

    tCurLayer_bn = get_batch_norm(tCurLayer, nIterations, bIsTesing)
    tCurLayer = tf.nn.tanh(tCurLayer_bn)

    #tCurLayer = tf.nn.softmax(tCurLayer)

    # Connect RNN2DNN Layer
    for i in range(1, len(g_tLayerDimension)):
        fActFunc = g_funActivateFunc
        isNeedDropOut = True
        if i == (len(g_tLayerDimension) - 1):
            fActFunc       = g_funLastActFunc
            isNeedDropOut = False
        tCurLayer = add_layer(tCurLayer, g_tLayerDimension[i - 1], g_tLayerDimension[i], 'hiddenLayer_%d' % i, nTrainDataSize, fActFunc, g_bIsEnalbeBN, isNeedDropOut)
        print(tCurLayer.shape)

    tCurLayer_bn = tCurLayer

    # define loss function
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(dy * tf.log(tf.clip_by_value(tCurLayer, 1e-10, 1.0)), 1))
    #cross_entropy = tf.reduce_mean(tf.square(tf.reshape(tCurLayer, [-1]) - tf.reshape((dy), [-1])))
    #fLossFunc = cross_entropy + tf.add_n(tf.get_collection('losses'))
    fLossFunc = cross_entropy 
    #tf.summary.scalar('loss', cross_entropy)

    # define verse deliver function
    #train_step = tf.train.AdagradOptimizer(nLearnRate).minimize(fLossFunc, global_step = global_step)
    train_step = tf.train.AdamOptimizer(nLearnRate).minimize(fLossFunc, global_step = global_step)

    return [train_step, fLossFunc, tCurLayer_bn]

def generate_batch(data_x, data_y):
    data_len = len(data_y)

    rand_timestep = random.randint(5, 50)
    rand_estimate_len = random.randint(1, 20)

    output_x = []
    output_y = []
    for i in range(0, g_nMiniBatchSize):
        rand_index = random.randint(rand_timestep - 1, data_len - rand_estimate_len)
        for j in range(rand_index - rand_timestep + 1, rand_index + 1):
            record = data_x[j]
            new_record = record.tolist() + [rand_estimate_len]
            #print(record.type)
            output_x.append(new_record)
            #print(output_x)
        y_index = rand_index + rand_estimate_len - 1
        record_y = bNp.array(data_y[y_index:y_index + 1])
        output_y.append(record_y.tolist())

    batch_x = bNp.reshape(bNp.array(output_x), [g_nMiniBatchSize, rand_timestep, g_rnn_FRAME_SIZE + 1]) 
    batch_y = bNp.reshape(bNp.array(output_y), [g_nMiniBatchSize, 20])

    #print(batch_x, batch_y)
    return [batch_x, batch_y]

def generate_test_batch(data_x, data_y):
    data_len = len(data_y)

    rand_timestep = random.randint(20, 50)
    rand_estimate_len = random.randint(1, 20)

    output_x = []
    output_y = []
    for i in range(0, g_nMiniBatchSize):
        rand_index = random.randint(rand_timestep - 1, data_len - rand_estimate_len)
        for j in range(rand_index - rand_timestep + 1, rand_index + 1):
            record = data_x[j]
            new_record = record.tolist() + [rand_estimate_len]
            #print(record.type)
            output_x.append(new_record)
            #print(output_x)
        y_index = rand_index + rand_estimate_len - 1
        record_y = bNp.array(data_y[y_index:y_index + 1])
        output_y.append(record_y.tolist())

    batch_x = bNp.reshape(bNp.array(output_x), [g_nMiniBatchSize, rand_timestep, g_rnn_FRAME_SIZE + 1]) 
    batch_y = bNp.reshape(bNp.array(output_y), [g_nMiniBatchSize, 20])

    #print(batch_x, batch_y)
    return [batch_x, batch_y, rand_timestep, rand_estimate_len]


# Read Data(From Web)
dir = "E:\\Stock\\"+ g_stock_id +"\\Stock_" + g_stock_id + ".csv"
tData = bPd.read_csv(dir, encoding = "ISO-8859-1")

# =============================================================================
# ############### Data Validation #####################
# =============================================================================
# Replace NAN
tData = tData.replace(to_replace='', value=bNp.nan)
# Drop NAN line
tData = tData.dropna(how='any')
# OutputData
tData.shape

#from sklearn.preprocessing import PolynomialFeatures
#pData = PolynomialFeatures().fit_transform(tData[g_Column_name[0:14]])

#print(pData.shape)
# =============================================================================
# ######### From Source Split Test Samples & Train Samples ############
# =============================================================================
#x_train, x_test, y_train, y_test = train_test_split(tData[g_Column_name[0:9]],
#                                                    tData[g_Column_name[10:13]],
#                                                    test_size=0.25,
#                                                    random_state=30)

x_train = tData[g_Column_name[0:cfg.g_rnn_FRAME_SIZE]]
y_train = tData[g_Column_name[11:31]]

# =============================================================================
# ############### Data Preproces #####################
# =============================================================================
ss = StandardScaler()
x_train = ss.fit_transform(x_train)

# =============================================================================
# ######### Form TensorFlow ############
# =============================================================================
# start a Session
dataset_size = len(y_train)
# define inputdata room
x = tf.placeholder(tf.float32, shape=(None, None, cfg.g_rnn_FRAME_SIZE + 1), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 20), name='y-input')
# define dropout
keep_prob = tf.placeholder(tf.float32)
# define curStep
nIterations = tf.placeholder(tf.int32)
bIsTesing   = tf.placeholder(tf.bool)

# define decay rate
global_step = tf.Variable(0, trainable = False)
nLearnRate  = tf.train.exponential_decay(g_fOriLearnRate, global_step, g_nStep4RateDecay, g_fLearnDecayRate, staircase = True)

# build net
train_step, cross_entropy, layer_graph = build_net(x, y_, dataset_size, g_bIsEnalbeBN, nLearnRate)


with tf.Session() as sess:
    #merged = tf.summary.merge_all()  
    #train_write = tf.summary.FileWriter("logs/train",sess.graph)
    #test_write = tf.summary.FileWriter("logs/test",sess.graph)
    acc_total = 0 
    loss_total = 0

    tf.global_variables_initializer().run()

    if g_bIsContinuesFromLast == True:
        model_dir = g_strModelDir
        model_name = "StockModelNow"
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(model_dir, model_name))

    raw_data_size = 3000

    for i in range(g_nTotalSteps):
        batch_x, batch_y = generate_batch(x_train[0:raw_data_size], y_train[0:raw_data_size])
        #print("!!!!!!!!!!!!!!!!")
        #start = i % (dataset_size - g_nMiniBatchSize * g_rnn_SEQUENCE_NUM)
        #end = start + g_nMiniBatchSize * g_rnn_SEQUENCE_NUM

        #batch_x = bNp.reshape(bNp.array(x_train[start:end]), [g_nMiniBatchSize, g_rnn_SEQUENCE_NUM, g_rnn_FRAME_SIZE]) 
        #batch_y = y_train[start + g_rnn_SEQUENCE_NUM - 1:end:g_rnn_SEQUENCE_NUM]
        #print(batch_x, batch_y)
        # start to train
        _, loss, rslt = sess.run([train_step, cross_entropy, layer_graph], feed_dict={x: batch_x, y_: batch_y, keep_prob: g_fKeepProb, bIsTesing: False, nIterations: i})
        loss_total += loss
        #outputer = open('D:\Resultss', 'w')
        #outputer.write(str(rslt))
        #outputer.close()

         #【间断性数据拟合检查】
        if i % g_nCheckRound == 0 and i > 0:
            batch_test_x, batch_test_y, rnd_timestep, rnd_estimatelen = generate_test_batch(x_train[raw_data_size:], y_train[raw_data_size:])

            correct_prediction = tf.equal(tf.argmax(layer_graph, 1), tf.argmax(y_, 1))
            correct_prediction_1 = tf.equal(tf.argmax(layer_graph, 1) + 1, tf.argmax(y_, 1))
            correct_prediction_2 = tf.equal(tf.argmax(layer_graph, 1) - 1, tf.argmax(y_, 1))
            correct_prediction = correct_prediction | correct_prediction_1 | correct_prediction_2
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            #rslt = sess.run([layer_graph], feed_dict={x: batch_test_x, keep_prob: 1.0, bIsTesing: True, nIterations: 1})
            rslt, ac, pred= sess.run([layer_graph, accuracy, correct_prediction], feed_dict = {x: batch_test_x, y_: batch_test_y, keep_prob: 1.0, bIsTesing: True, nIterations: i})

            #print(rnn_output)
            # 计算交叉熵：
            #total_cross_entropy = sess.run(
            #    cross_entropy, feed_dict={x: x_train, y_: y_train, keep_prob: 1.0, bIsTesing: False, nIterations: i})
            print("After %d training steps(s) cross entropy on all data is %g" % (i, loss_total / g_nCheckRound))
            print("timestep:", rnd_timestep)
            print("estimatelen:", rnd_estimatelen)
            print("accuracy:",ac)

            # 训练数据正确率：
            #correct_prediction = tf.equal(tf.argmax(layer_graph, 1), tf.argmax(y_, 1))
            #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            #ac = accuracy.eval({x: x_train, y_: y_train, keep_prob: 1.0, bIsTesing: True, nIterations: i})
            #print(ac)

            ## 测试数据正确率
            #correct_prediction_test = tf.equal(tf.argmax(layer_graph, 1), tf.argmax(y_, 1))
            #accuracy_test = tf.reduce_mean(tf.cast(correct_prediction_test, tf.float32))
            #bc = accuracy_test.eval({x: x_test, y_: y_test, keep_prob: 1.0, bIsTesing: True, nIterations: i})
            #print(bc)

            # 生成可视化表格数据
            #train_result = sess.run(merged, feed_dict = {x:x_train, y_:y_train, keep_prob: 1.0})
            #test_result = sess.run(merged, feed_dict = {x:x_test, y_:y_test, keep_prob: 1.0})
            #train_write.add_summary(train_result, i)  
            #test_write.add_summary(test_result, i)

             #输出当前符合拟合条件的模型信息：
            if (loss_total / g_nCheckRound) < g_fExpectAccruacy:
                f = open("D:\stockout.txt", "a+")
                print("StockAnalysis.py 详细信息：", file =f)
                print("%r，LearnRate: %f batch_size: %d，激活函数：softmax，keep_prob: %f" % (g_tLayerDimension, g_fOriLearnRate, g_nMiniBatchSize, g_fKeepProb), file = f)
                print("-------------------------------------------------\n")
                f.close()

                # 保存模型
                saver = tf.train.Saver()
                model_dir = g_strModelDir
                model_name = "StockModelNow"
                if not os.path.exists(model_dir):
                    os.mkdir(model_dir)
                saver.save(sess, os.path.join(model_dir, model_name))

                g_fExpectAccruacy = (loss_total / 500)
            loss_total = 0

## =============================================================================
## ############### Logistic Regression #####################
## =============================================================================
# lr = LogisticRegression()
# lr.fit(x_train, y_train)
# lr_y_predict = lr.predict(x_test)
#
## =============================================================================
## ############### Calc Report #####################
## =============================================================================
# from sklearn.metrics import classification_report
#
# print 'Accuracy of LR Classifier:', lr.score(x_test, y_test)
# print classification_report(y_test, lr_y_predict, target_names = ['Benign', 'Malignant'])
#
# print lr_y_predict


