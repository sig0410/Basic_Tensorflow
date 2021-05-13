import tensorflow as tf
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets.cifar10 import load_data

# from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)
# Define model

(x_train, y_train), (x_test, y_test) = load_data()

print(np.shape(x_train))
print(np.shape(y_train))
# 32*32의 이미지이고 RGB 값이며 50,000개의 데이터가 있다
print(np.shape(x_test))
print(np.shape(y_test))
# 라벨이 스칼라값으로 되어 있음 추후 분류를 위해 원핫인코딩

# print(y_train[1])
# # 스칼라 값으로 되어 있음
# plt.imshow(x_train[1])
# plt.show()

# exit(1)
X = tf.placeholder(tf.float32, [None, 32, 32, 3], name = 'X')
Y = tf.placeholder(tf.float32, [None, 10], name = 'Y')
keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev = 0.01))
# 3*3 에 depth가 3이고 32개 존재
# Filter : 3*3이고 32개의 Filter
L1 = tf.nn.conv2d(X, W1, strides = [1, 1, 1, 1], padding = 'SAME')

L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
L1 = tf.nn.dropout(L1, keep_prob = keep_prob)
# ksize는 필터사이즈 2*2의 Maxpooling
# output : 16*16*32


W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev = 0.01))
# 3*3에 depth가 32인 filter 64개
L2 = tf.nn.conv2d(L1, W2, strides = [1, 1, 1, 1], padding = 'SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
L2 = tf.nn.dropout(L2, keep_prob = keep_prob)
# output : 8*8*64

W3 = tf.Variable(tf.random_normal([3,3,64,128], stddev = 0.01))
L3 = tf.nn.conv2d(L2, W3, strides = [1, 1, 1, 1], padding = 'SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
L3 = tf.nn.dropout(L3, keep_prob = keep_prob)
# output : 4*4*128

L3_flat = tf.reshape(L3, [-1, 4*4*128])
# Flatten하는 과정 (Vector 형태로)

W4 = tf.get_variable('W4', shape = [4 * 4 * 128, 128], initializer = tf.initializer.he_normal())
# 바로 클래스로 분류하지 않고 FCL을 하나 더 쓰기위해 128
b4 = tf.Variable(tf.random_normal([128]))
FC1 = tf.nn.relu(tf.nn.xw_plus_b(L3_flat, W4, b4))
FC1 = tf.nn.dropout(FC1, keep_prob = keep_prob)

W5 = tf.get_variable('W5', shape = [128, 64], initializer = tf.initializers.he_normal())
b5 = tf.Variable(tf.random_normal[64])
FC2 = tf.nn.relu(tf.nn.xw_plus_b(FC1, W5, b5))
FC2 = tf.nn.dropout(FC2, keep_prob=keep_prob)

W6 = tf.get_variable('W6', shape = [64, 10], initializer = tf.initializers.he_normal() )
b6 - tf.Variable(tf.random_normal[10])
hypothesis = tf.nn.xw_plus_b(FC2, W6, 6, name = 'hypothesis')




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

timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp))
# 현재 디렉토리에 runs라는 폴더를 만들고 시간순으로 정리

# Save Model
checkpoint_dir = os.path.abspath((os.path.join(out_dir, 'checkpoints')))
checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

saver = tf.train.Saver(tf.global_variables(), max_to_keep = 3)
# 모델을 가장 최근 3개 기준으로 저장

training_epochs = 3
batch_size = 128

max = 0.0
early_stopped = 0

(x_train_val, y_train_val), (x_test, y_test) = load_data()

# Data shuffle
shuffle_indices = np.random.permutation(np.arange(len(y_train_val)))
# y_train_val의 길이만큼 랜덤하게 섞어줌

shuffled_x = np.asarray(x_train_val[shuffle_indices])
shuffled_y = y_train_val[shuffle_indices]

val_sample_index = -1 * int(0.1 * float(len(y_train_val)))
# validation의 인덱스를 결정
# validation은 train의 10%만 뒤에서 부터 인덱스 설정

x_train, x_val = shuffled_x[:val_sample_index], shuffled_x[val_sample_index:]
y_train, y_val = shuffled_y[:val_sample_index], shuffled_y[val_sample_index:]
# train과 validation을 나눠줌
x_test = np.asarray(x_test)

# one_hot
y_train_one_hot = np.eye(10)[y_train]
# y_train : (10, 10, 1)
y_train_one_hot = np.squeeze(y_train_one_hot, axis = 1)
# (10,10,1)을 (10,10)으로 차원을 줄이기 위해 사용
# squeeze : [[2],[1],[5]] -> [2, 1, 5]
y_test_one_hot = np.eye(10)[y_test]
y_test_one_hot = np.squeeze(y_test_one_hot, axis = 1)
y_val_one_hot = np.eye(10)[y_val]
y_val_one_hot = np.squeeze(y_val_one_hot, axis = 1)

# NextBatch

def next_batch(batch_size, data):
    data = np.array(data)
    np.random.seed(10)
    shuffle_indices = np.random.permutation((np.arange(len(data))))
    shuffled_data = data[shuffle_indices]
    # epoch가 끝날때 마다 shuffle 해줌, 순서에 대해 학습을 예방하기 위해
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        # 각 배치가 시작할때 시작 인덱스
        end_index = min((batch_num + 1) * batch_size, len(data))
        # 마지막부분은 batch_size보다 작을 수 있어서 고려
        yield shuffled_data[start_index:end_index]
        # yield 공부

# Train model


for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(len(y_train) / batch_size)
    batches = next_batch(batch_size, list(zip(x_train, y_train_one_hot)))
    # zip은 x와 y가 한쌍인 데이터
    for batch in batches:
        batch_xs, batch_ys = zip(*batch)
        # zip(*batch) 이거 모르겠음
        feed_dict = {X : batch_xs, Y : batch_ys, keep_prob : 0.8}
        c, _, a = sess.run([cost, optimizer], feed_dict = feed_dict)
        avg_cost += c / total_batch
    print('Epoch : ', '04%d' %(epoch + 1), 'training_cost : ', '{:9f}'.format(avg_cost))

    test_accuracy = sess.run(accuracy, feed_dict = {X : x_val, Y : y_val_one_hot, keep_prob : 1.0})
    print('Test Accuracy :', test_accuracy)
    # Early Stopping
    if test_accuracy > max:
        max = test_accuracy
        early_stopped = epoch + 1

print('Test Max Accuracy :', max)
print('Early stopped Time : ', early_stopped)
