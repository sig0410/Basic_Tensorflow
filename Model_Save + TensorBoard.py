import tensorflow as tf
import time
import os
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)

# Define model

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.random_normal([784, 500]))
b1 = tf.Variable(tf.random_normal([500]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([500, 300]))
b2 = tf.Variable(tf.random_normal([300]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.random_normal([300, 200]))
b3 = tf.Variable(tf.random_normal([200]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)


W4 = tf.Variable(tf.random_normal([200, 10]))
b4 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L3, W4) + b4

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


summary_op = tf.summary.scalar('accuracy', accuracy)
# accuracy로 요약

timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp))
# 현재 디렉토리에 runs라는 폴더를 만들고 시간순으로 정리

train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
# 현재 디렉토리에 Runs가 있고 시간순으로 정리되고 summaries가 있고 뒤에 train이 존재
train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
# train을 했을때 요약해주는 Writer

val_summary_dir = os.path.join(out_dir, 'summaries', 'valid')
# 현재 디렉토리에 Runs가 있고 시간순으로 정리되고 summaries가 있고 뒤에 valid이 존재
val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)
# valid을 했을때 요약해주는 Writer

# Save Model
checkpoint_dir = os.path.abspath((os.path.join(out_dir, 'checkpoints')))
checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

saver = tf.train.Saver(tf.global_variables(), max_to_keep = 3)
# 모델을 가장 최근 3개 기준으로 저장

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
        c, _, a= sess.run([cost, optimizer, summary_op], feed_dict = feed_dict)
        avg_cost += c / total_batch
        # 각 batch_size만큼의 Cost를 전체 데이터에서 batch_size만큼 나눈 값을 더하면 한 epoch당 평균 cost가 나옴
    print('Epoch : ', '04%d' %(epoch + 1), 'training_cost : ', '{:9f}'.format(avg_cost))

    train_summary_writer.add_summary(a,early_stopped)
    # a가 summary_op이고 early_stopped을 넣는건 그 전에 넣어야 해서
    # 한 epoch가 끝났으니 저장하는 것


    test_accuracy, summaries= sess.run([accuracy, summary_op], feed_dict = {X : mnist.test.images, Y : mnist.test.labels})
    val_summary_writer.add_summary(summaries, early_stopped)

    print('Test Accuracy : ', test_accuracy)

    # Early Stopping
    if test_accuracy > max:
        max = test_accuracy
        early_stopped = epoch + 1
        saver.save(sess, checkpoint_prefix, global_step = early_stopped)
        # early stop된 모델을 저장

# 이렇게 되면 학습하면서 summary_writer에 입력해줌
# 저장된 writer를 텐서보드로 볼 수 있음

# 터미널에서 Tensorboard를 볼 수 있음
# tensorboard --logdir=./runs/ 를 입력하면 그래프를 볼 수 있음
