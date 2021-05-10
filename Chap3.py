import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot = True, validation_size = 5000)

X = tf.placeholder(tf.float32, [None, 784], name = 'X')
Y = tf.placeholder(tf.float32, [None, 10], name = 'Y')

# 기존 가중치 설정
W1 = tf.Variable(tf.random_normal([784, 256]))
b1 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([256, 256]))
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.random_normal([256, 10]))
b3 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.nn.xw_plus_b(L2, W3, b3, name = 'hypothesis')

# 가중치 초기화
# 기존에는 정규분포를 따르는 임의의 값을 정하는데 만약 초기값을 0으로 지정하면 역전파시 모든 가중치 값이 똑같이 갱신되기 때문에 학습이 제대로 이뤄지지 않음
W1 = tf.get_variable('W1', shape = [784, 256], initializer = tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.get_variable('W2', shape = [256, 256], initializer = tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.get_variable('W3', shape = [256, 10], initializer = tf.contrib.layers.xavier_initializer() )
b3 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L2, W3) + b3

# Drop Out + Initializer
keep_prob = tf.placeholder(tf.float32, name ="keep_prob")

W1 = tf.get_variable('W1', shape = [784, 256], initializer = tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob = keep_prob )

W2 = tf.get_variable('W2', shape = [256, 256], initializer = tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob = keep_prob)

W3 = tf.get_variable('W3', shape = [256, 10], initializer = tf.contrib.layers.xavier_initializer() )
b3 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L2, W3) + b3

# Weight Decay , Weight Restriction
# 높은 가중치에 대해 제약을 주며 오버피팅을 막는 방법
l2_loss = 0
W1 = tf.Variable(tf.random_normal([784, 256]))
b1 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
l2_loss += tf.nn.l2_loss(W1)
l2_loss += tf.nn.l2_loss(b1)

W2 = tf.Variable(tf.random_normal([256, 256]))
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
l2_loss += tf.nn.l2_loss(W2)
l2_loss += tf.nn.l2_loss(b2)

W3 = tf.Variable(tf.random_normal([256, 10]))
b3 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.nn.xw_plus_b(L2, W3, b3, name = 'hypothesis')
l2_loss += tf.nn.l2_loss(W3)
l2_loss += tf.nn.l2_loss(b3)


correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
# 추정값과 실제값이 맞는지 틀린지 True, False로 나오는 것
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# True, False로 나오는 것을 Float형태로 형 변환

import time
import os
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# tf.variable에 랜덤값 넣어줌
l2_loss_lambda = 0.001
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hypothesis, labels = Y))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hypothesis, labels = Y)) + l2_loss_lambda * l2_loss
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp))
checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

saver = tf.train.Saver(tf.global_variables(), max_to_keep = 3)

# Early Stopping
learning_rate = 0.001
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize((cost))

training_epochs = 30
batch_size = 100

max = 0
early_stopped = 0
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X : batch_xs, Y : batch_ys } # dropout을 사용할때 지정해주는 것 keep_prob
        c, _ = sess.run([cost, optimizer], feed_dict = feed_dict)
        avg_cost += c / total_batch

    print('Epoch:','%04d' % (epoch + 1), 'training cost = ', '{:9f}'.format(avg_cost))
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    valid_accuracy = sess.run(accuracy, feed_dict = {X : mnist.validation.images, Y : mnist.validation.labels})
    # dropout을 사용할때 지정해주는 것 keep_prob
    print('Valid Accuracy : ', valid_accuracy)
    if valid_accuracy > max:
        max = valid_accuracy
        early_stopped = epoch + 1
        saver.save(sess, checkpoint_prefix, global_step = early_stopped)

print('Validation Max Accuracy : ', max)
print('Early Stopped Time : ', early_stopped)

test_accuracy = sess.run(accuracy, feed_dict = {X : mnist.test.images, Y : mnist.test.labels})
# dropout을 사용할때 지정해주는 것 keep_prob
print('Test Accuracy : ', test_accuracy)
