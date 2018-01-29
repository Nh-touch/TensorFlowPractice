import tensorflow as tf
def read_jpg_from_img(filenames_queue): 
    # read img data from jpg file 
    img_load_queue = tf.train.string_input_producer(filenames_queue)
    img_reader = tf.WholeFileReader()
    _, img_raw_data = img_reader.read(img_load_queue) 
    img_data = tf.image.decode_jpeg(img_raw_data) 
    return img_data 

def write_jpg_to_tfrecord(tf_data): 
    # writer to TFRecord 
    TFRecord_writer = tf.python_io.TFRecordWriter("./img_data.tfrecord") 
    rslt = TFRecord_writer.write(tf_data.SerializeToString()) 
    TFRecord_writer.close()

def read_data_from_tfrecord(filenames_queue):
    data_load_queue = tf.train.string_input_producer(filenames_queue) 
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(data_load_queue) 
    features = tf.parse_single_example(serialized_example, features={ 'label':tf.FixedLenFeature([],tf.string), 'image_raw':tf.FixedLenFeature([],tf.string) })
    img_jpg = tf.decode_raw(features['image_raw'], tf.uint8) 
    img_data = tf.reshape(img_jpg, [3, 3, 3])
    label_data = tf.cast(features['label'], tf.string)
    return img_data, label_data

small_img_data = read_jpg_from_img(["smallimg.jpg"]) # read from TFRecord 
img_tfrecord, label_tfrecord = read_data_from_tfrecord(["./img_data.tfrecord"]) 
load_op = tf.tuple([img_tfrecord, label_tfrecord]) 
writer = tf.summary.FileWriter("D:/tensorlog", tf.get_default_graph())

with tf.Session() as sess:
    # start queue(for img_load_queue) 
    coord = tf.train.Coordinator() 
    threads = tf.train.start_queue_runners(coord=coord) # get jpeg img data & print 
    jpg_tensor = sess.run(small_img_data) 
    print(jpg_tensor) 
    img_height, img_width, img_channel = jpg_tensor.shape 
    print(img_channel) # form label & img TFRecord
    tf_data = tf.train.Example(features = tf.train.Features( 
        feature = { 
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'\x01'])), 
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[jpg_tensor.tobytes()]))})) 
    write_jpg_to_tfrecord(tf_data) # read data
    img, label = sess.run(load_op) 
    print(img,'\n\n',label)
    print(tf.equal(jpg_tensor, img).eval()) 
    coord.request_stop()
    coord.join(threads)

