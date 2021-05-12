import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True, validation_size = 5000)

tf.flags.DEFINE_string('checkpoint_dir','./runs/{timestamp}/checkpoints','Checkpoint directory from training run')
FLAGS = tf.flags.FLAGS
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        X = graph.get_operation_by_name("X").outputs[0]
        Y = graph.get_operation_by_name("Y").outputs[0]
        hypothesis = graph.get_operation_by_name("hypothesis").outputs[0]
        correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        test_accuracy = sess.run(accuracy, feed_dict = {X : mnist.test.images, Y : mnist.test.labels})
        print('Test Max Accuracy : ', test_accuracy)
