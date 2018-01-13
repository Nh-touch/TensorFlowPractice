# -*- utf-8 -*- 
import tensorflow as tf 
def feed_once(sess, input_tensor): 
    input_data = sess.graph.get_tensor_by_name("transmission/input_variable/input_data:0") 
    global_step_op = sess.graph.get_tensor_by_name("update_layer/global_step_increase:0")
    train_op = sess.graph.get_tensor_by_name("update_layer/global_total_increase:0")
    summary_merge_op = sess.graph.get_tensor_by_name("global_op/Merge/MergeSummary:0")
    _, step, summary= sess.run([train_op, global_step_op, summary_merge_op], feed_dict = {input_data: input_tensor})
    return step, summary 

def write_to_summary(writer, summary, step):
    writer.add_summary(summary, global_step = step)

def form_graph(): 
    graph0 = tf.Graph() 
    with graph0.as_default():
        with tf.variable_scope("global_variable"): 
            global_step = tf.get_variable("global_step", [], dtype=tf.int32, trainable=False, initializer=tf.zeros_initializer)
            global_total_result = tf.Variable(0.0, trainable=False, dtype=tf.float32, name = "global_total_result")
        with tf.variable_scope("transmission"):
            with tf.name_scope("constant_variable"):
                k = tf.constant(5, name = "constant_k")
            with tf.name_scope("input_variable"):
                input_data = tf.placeholder(dtype = tf. int32, shape = [None], name = "input_data")
            with tf.name_scope("calc_layer"):
                a = tf.reduce_prod(input_data, name = "reduce_prod_a")
                b = tf.reduce_sum(input_data, name = "reduce_sum_b")
            with tf.name_scope("output_layer"):
                c = tf.add(a, b, name = "add_c")
                d = tf.multiply(a, b, name = "mul_d")
                e = tf.add(c, d, name = "add_e")
                output = tf.add(e, k, name = "add_f")
        # 共享变量操作（中文注释）
        with tf.variable_scope("global_variable", reuse = True):
            global_step = tf.get_variable("global_step", [], dtype=tf.int32, initializer=tf.zeros_initializer)

        with tf.name_scope("update_layer"):
            global_step_op = tf.assign_add(global_step, 1, name = "global_step_increase")
            global_total_op = tf.assign_add(global_total_result, tf.cast(output, tf.float32), name = "global_total_increase")

        with tf.name_scope("summary"):
            avg = tf.div(global_total_result, tf.cast(global_step, tf.float32), name = "div")
            tf.summary.scalar("output", output)
            tf.summary.scalar("total", global_total_result)
            tf.summary.scalar("average", avg)

        with tf.name_scope("global_op"):
            summary_merge_op = tf.summary.merge_all() # Merge/MergeSummary:0 
            init_op = tf.global_variables_initializer() # init return graph0

    return graph0

def main(_):
    graph0 = form_graph()
    writer = tf.summary.FileWriter("D:/tensorlog", graph0)
    with tf.Session(graph = graph0) as sess: 
        sess.run(sess.graph.get_operation_by_name("global_op/init")) 
        step, summary = feed_once(sess, [2, 5, 9]) 
        write_to_summary(writer, summary, step) 
        step, summary = feed_once(sess, [9])
        write_to_summary(writer, summary, step) 
        step, summary = feed_once(sess, [9, 6])
        write_to_summary(writer, summary, step)
        step, summary = feed_once(sess, [3, 5, 8, 0, 7])
        write_to_summary(writer, summary, step)
        step, summary = feed_once(sess, [3, 5, 7]) 
        write_to_summary(writer, summary, step) 
        step, summary = feed_once(sess, [1, 5])
        write_to_summary(writer, summary, step)
        step, summary = feed_once(sess, [2, 1, 3]) 
        write_to_summary(writer, summary, step) 
        sess.close()
    writer.flush() 
    writer.close() 

if __name__ == "__main__":
    tf.app.run()

