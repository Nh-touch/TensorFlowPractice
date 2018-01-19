# -*- utf-8 -*-
import tensorflow as tf
# global value
FEATURE_NUM = 2
BATCH_SIZE = 2
LEARN_RATE = 0.0001 
# embed data 
data_x = [[1,2], [2,4], [4,8], [8,16], [-1,-2], [-5,-10]] 
data_y = [[3], [6], [12], [24], [-3], [-15]]
def feed_once_linear_regression(sess, train_data, train_output):
    input_data = sess.graph.get_tensor_by_name("train_data_source/x-input:0") 
    label_data = sess.graph.get_tensor_by_name("train_data_source/y-input:0") 
    train_op = sess.graph.get_operation_by_name("output_layer/GradientDescent")
    _ = sess.run([train_op], feed_dict = {input_data: train_data, label_data: train_output})

def get_current_loss_linear_regression(sess, train_data, train_output):
    input_data = sess.graph.get_tensor_by_name("train_data_source/x-input:0") 
    label_data = sess.graph.get_tensor_by_name("train_data_source/y-input:0")
    loss_op = sess.graph.get_tensor_by_name("loss_layer/loss_mse:0")
    summary_merge_op = sess.graph.get_tensor_by_name("output_layer/Merge/MergeSummary:0")
    loss_mse, summary= sess.run([loss_op, summary_merge_op], feed_dict = {input_data: train_data, label_data: train_output}) 
    return loss_mse, summary

def write_to_summary_linear_regression(writer, summary, step):
    writer.add_summary(summary, global_step = step)

def print_variable_linear_regression(sess):
    weights = sess.graph.get_tensor_by_name("v_weights:0") 
    bias = sess.graph.get_tensor_by_name("v_bias:0") 
    print(sess.run(weights)) 
    print(sess.run(bias)) 

def form_linear_regression_forward_graph():
    graph0 = tf.Graph() 
    with graph0.as_default(): 
        with tf.name_scope("train_variables"):
            weights = tf.get_variable("v_weights", [FEATURE_NUM, 1], dtype=tf.float32, trainable=True, initializer=tf.initializers.constant(1.0))
            bias = tf.get_variable("v_bias", [1, 1], dtype=tf.float32, trainable=True, initializer=tf.initializers.constant(0.1)) 

        with tf.name_scope("train_data_source"):
            input_data = tf.placeholder(dtype = tf.float32, shape = [None, FEATURE_NUM], name = "x-input") # batchsize * features
            label_data = tf.placeholder(dtype = tf.float32, shape = [None, 1], name = "y-input") # scalar

        with tf.variable_scope("data_forward_process"):
            with tf.name_scope("output_layer"): 
                output = tf.matmul(input_data, weights) + bias 

        with tf.name_scope("loss_layer"):
            loss_mse = tf.reduce_mean(tf.square(label_data - output), name = "loss_mse") 

        with tf.name_scope("summary"): 
            tf.summary.histogram("output", output)
            tf.summary.histogram("weights1", weights[0,0]) 
            tf.summary.histogram("weights2", weights[1,0])
            tf.summary.histogram("weightsall", weights)
            tf.summary.histogram("bias", bias)
            tf.summary.scalar("loss_mse", loss_mse)
        with tf.name_scope("output_layer"):
            summary_merge_op = tf.summary.merge_all() # Merge/MergeSummary:0
            init_op = tf.global_variables_initializer() # init 
            train_op = tf.train.GradientDescentOptimizer(LEARN_RATE).minimize(loss_mse) 
            print(summary_merge_op.name) 

    return graph0 

def main(_): 
    graph0 = form_linear_regression_forward_graph() 
    writer = tf.summary.FileWriter("D:/tensorlog", graph0)
    dataset_size = len(data_y) 
    with tf.Session(graph = graph0) as sess:
        sess.run(sess.graph.get_operation_by_name("output_layer/init"))
        for i in range(1000): 
            start = (i * BATCH_SIZE) % dataset_size 
            end = min(start + BATCH_SIZE, dataset_size) 
            feed_once_linear_regression(sess, data_x[start:end], data_y[start:end])
            if i % 100 == 0:
                total_mean_suqre_error, summary = get_current_loss_linear_regression(sess, data_x, data_y) 
                write_to_summary_linear_regression(writer, summary, i)
                print("After %d training steps(s) mean square error on all data is " % (i)) 
                print(total_mean_suqre_error)
                print_variable_linear_regression(sess) 
        sess.close() 
    writer.flush() 
    writer.close() 

if __name__ == "__main__":
    tf.app.run()

