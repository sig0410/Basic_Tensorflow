import tensorflow as tf
import time
import os
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)

# Define model

X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])
# flatten한 것을 다시 28*28로 변경
# -1은 None과 같으며 Depth는 1
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev = 0.01))
# Filter : 3*3이고 32개의 Filter
L1 = tf.nn.conv2d(X_img, W1, strides = [1, 1, 1, 1], padding = 'SAME')
# [?, 28, 28, 32 ] : 28*28의 Activation Map이 32개
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
# ksize는 필터사이즈 2*2의 Maxpooling
# output = [?, 14, 14, 32]
# (28 - 2) / 2 + 1 : 14


# convolution layer 하나가 완성

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev = 0.01))
L2 = tf.nn.conv2d(L1, W2, strides = [1, 1, 1, 1], padding = 'SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
# output = [?, 7, 7, 64]
# (14 - 2) / 2 + 1 : 7

L2_flat = tf.reshape(L2, [-1, 7*7*64])
# Flatten하는 과정 (Vector 형태로)

W3 = tf.get_variable('W3', shape = [7 * 7 * 64, 10], initializer = tf.contrib.layers.xavier_initializer())
# 7 * 7 * 64의 벡터에서 최종적으로 10개의 output을 도출하는 것
b = tf.Variable(tf.random_normal([10]))
hypothesis = tf.nn.xw_plus_b(L2_flat, W3, b, name = 'hypothesis')



correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Define cost/loss & optimizer

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits = hypothesis, labels = Y
))

learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize((cost))

# Initialize

sess = tf.Session()
sess.run(tf.global_variables_initializer())

training_epochs = 15
batch_size = 128

# Train model

max = 0
early_stopped = 0
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X : batch_xs, Y : batch_ys}
        c, _= sess.run([cost, optimizer], feed_dict = feed_dict)
        avg_cost += c / total_batch
        # 각 batch_size만큼의 Cost를 전체 데이터에서 batch_size만큼 나눈 값을 더하면 한 epoch당 평균 cost가 나옴
    print('Epoch : ', '04%d' %(epoch + 1), 'training_cost : ', '{:9f}'.format(avg_cost))

    test_accuracy = sess.run(accuracy, feed_dict = {X : mnist.test.images, Y : mnist.test.labels})

    # Early Stopping
    if test_accuracy > max:
        max = test_accuracy
        early_stopped = epoch + 1

print('Test Max Accuracy :', max)
print('Early stopped Time : ', early_stopped)
