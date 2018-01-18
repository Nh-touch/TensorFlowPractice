# -*- utf-8 -*-
import tensorflow as tf

# global value 
FEATURE_NUM = 2
BATCH_SIZE = 4
LEARN_RATE = 0.01

# embed data
data_x = [[1,2], [2,4], [4,8], [16,8], [2,2], [3,0]]
data_y = [[0,0,1], [0,0,1], [0,0,1], [1,0,0], [0,1,0], [1,0,0]] 
def feed_once_softmax_regression(sess, train_data, train_output):
    input_data = sess.graph.get_tensor_by_name("train_data_source/x-input:0") 
    label_data = sess.graph.get_tensor_by_name("train_data_source/y-input:0")
    train_op = sess.graph.get_operation_by_name("output_layer/GradientDescent")
    _ = sess.run([train_op], feed_dict = {input_data: train_data, label_data: train_output}) 

def get_current_loss_softmax_regression(sess, train_data, train_output):
    input_data = sess.graph.get_tensor_by_name("train_data_source/x-input:0") 
    label_data = sess.graph.get_tensor_by_name("train_data_source/y-input:0") 
    loss_op = sess.graph.get_tensor_by_name("loss_layer/loss_ce:0") 
    output_op = sess.graph.get_tensor_by_name("data_forward_process/output_layer/output:0")
    summary_merge_op = sess.graph.get_tensor_by_name("output_layer/Merge/MergeSummary:0")
    op_group = tf.tuple([loss_op, summary_merge_op, output_op]) 
    loss_mse, summary, output= sess.run(op_group, feed_dict = {input_data: train_data, label_data: train_output})
    print(output) 
    return loss_mse, summary 

def write_to_summary_softmax_regression(writer, summary, step):
    writer.add_summary(summary, global_step = step)

def print_variable_softmax_regression(sess): 
    weights = sess.graph.get_tensor_by_name("v_weights:0")
    bias = sess.graph.get_tensor_by_name("v_bias:0") 
    print(sess.run(weights))
    print(sess.run(bias)) 

def form_softmax_regression_forward_graph():
    graph0 = tf.Graph() 
    with graph0.as_default(): 
        with tf.name_scope("global_counter"): 
            counter = tf.get_variable("g_counter", [], dtype=tf.int32, trainable=False, initializer=tf.initializers.zeros) 

        with tf.name_scope("train_variables"):
            weights = tf.get_variable("v_weights", [FEATURE_NUM, 3], dtype=tf.float32, trainable=True, initializer=tf.initializers.truncated_normal(0.1))
            bias = tf.get_variable("v_bias", [1, 3], dtype=tf.float32, trainable=True, initializer=tf.initializers.constant(0.1))

        with tf.name_scope("train_data_source"):
            input_data = tf.placeholder(dtype = tf.float32, shape = [None, FEATURE_NUM], name = "x-input") # batchsize * features
            label_data = tf.placeholder(dtype = tf.float32, shape = [None, 3], name = "y-input") # scalar 

        with tf.variable_scope("data_forward_process"): 
            with tf.name_scope("output_layer"):
                z = tf.matmul(input_data, weights) + bias
                output = tf.nn.softmax(z, name = "output") 

        with tf.name_scope("loss_layer"): 
            # p1 = y_ * log(y) 
            p1 = label_data * tf.log(tf.clip_by_value(output, 1e-10, 1.0)) 
            # p2 = (1 - y_) * log(1 - y) 
            p2 = tf.subtract(1.0, label_data) * tf.log(tf.clip_by_value(tf.subtract(1.0, output), 1e-10, 1.0))
            # loss_ce = -(sum(p1+p2)) 
            loss_ce = tf.negative(tf.reduce_sum(tf.add(p1, p2)), name = "loss_ce") 

        with tf.name_scope("summary"): 
            tf.summary.histogram("output", output)
            tf.summary.histogram("weightsall", weights)
            tf.summary.histogram("bias", bias) 
            tf.summary.scalar("loss_mse", loss_ce) 
            tf.summary.scalar("global_counter", counter) 

        with tf.name_scope("output_layer"): 
            summary_merge_op = tf.summary.merge_all() # Merge/MergeSummary:0 
            init_op = tf.global_variables_initializer() # init 
            with tf.control_dependencies([tf.assign_add(counter, 1)]):
                train_op = tf.train.GradientDescentOptimizer(LEARN_RATE).minimize(loss_ce)

    return graph0 

def main(_): 
    graph0 = form_softmax_regression_forward_graph() 
    writer = tf.summary.FileWriter("D:/tensorlog", graph0) 
    dataset_size = len(data_y) 
    with tf.Session(graph = graph0) as sess:
        sess.run(sess.graph.get_operation_by_name("output_layer/init"))
        for i in range(200000):
            start = (i * BATCH_SIZE) % dataset_size
            end = min(start + BATCH_SIZE, dataset_size) 
            feed_once_softmax_regression(sess, data_x[start:end], data_y[start:end]) 
            if i % 500 == 0:
                total_mean_suqre_error, summary = get_current_loss_softmax_regression(sess, data_x, data_y) 
                write_to_summary_softmax_regression(writer, summary, i) 
                print("After %d training steps(s) mean square error on all data is " % (i)) 
                print(total_mean_suqre_error)
                print_variable_softmax_regression(sess) 
        sess.close()
    writer.flush() 
    writer.close()

if __name__ == "__main__":
    tf.app.run()

