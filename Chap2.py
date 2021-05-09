import tensorflow as tf
#
# x_data = [[1,2]]
#
# X = tf.placeholder(tf.float32, shape = [None, 2])
# W = tf.Variable(tf.random_normal([2,1]), name = 'weight')
# b = tf.Variable(tf.random_normal([1]), name = 'bias')
#
# hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     prediction = sess.run(hypothesis, feed_dict = {X : x_data})
#     print(prediction)
#
#
# x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]
# y_data = [[0], [0], [0], [1], [1], [1]]
#
# X = tf.placeholder(tf.float32, shape = [None, 2])
# Y = tf.placeholder(tf.float32, shape = [None, 1])
# W = tf.Variable(tf.random_normal([2,1]), name = 'weight')
# b = tf.Variable(tf.random_normal([1]), name = 'bias')
#
# hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
# # Layer를 통해 나온 값
# cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
# # 에러를 구하는 코드
#
# train = tf.train.GradientDescentOptimizer(learning_rate= 0.01).minimize(cost)
# # Optimizer 정의
# # error를 최소화히기 위해 error를 정의한 cost를 넣음
# predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
#
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype = tf.float32))
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for step in range(10001):
#         cost_val,_ = sess.run([cost, train], feed_dict = {X : x_data, Y : y_data})
#         if step % 200 == 0:
#             print(step, cost_val)
#     h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict = {X : x_data, Y : y_data})
#     print('\n Hypothesis : ', h, '\n Correct(Y) :', c, '\n Accuracy :', a)

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)
#
# print(np.shape(mnist.train.images))
# print(np.shape(mnist.train.labels))
# print(np.shape(mnist.test.images))
# print(np.shape(mnist.test.labels))
#
# # plt.imshow(
# #     mnist.train.images[1].reshape(28,28),
# #     cmap = 'Greys',
# #     interpolation = 'nearest',
# # )
# # plt.show()

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# Layer 구성
W1 = tf.Variable(tf.random_normal([784, 256]))
b1 = tf.Variable(tf.random_normal([256]))
# 값들을 사용자가 정의한 크기에 맞게 랜덤하게 구성해줌
# Input : 784 , Output : 256
L1 = tf.nn.relu(tf.matmul(X, W1)+b1)
# Activation Function

W2 = tf.Variable(tf.random_normal([256,256]))
b2 = tf.Variable(tf.random_normal([256]))
# Input : 256 , Output : 256
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
# Activation Function

W3 = tf.Variable(tf.random_normal([256,10]))
b3 = tf.Variable(tf.random_normal([10]))
# Input : 256 , Output : 10
hypothesis = tf.matmul(L2, W3) + b3
# Activation Function


softmax_result = tf.nn.softmax(hypothesis)
# # 확률값으로 나옴
# Output in Softmax Function
# # 0.1 0.02 .... 0.3 <- 총 10개
# sess = tf.Session()
#
# sess.run(tf.global_variables_initializer())
# # global_variables_initializer() : 위의 variable에 랜덤값을 부여
# x, y = mnist.train.next_batch(1) # 테스트를 위해 하나만 가져옴
#
# # 데이터를 내가 설계한 네트워크에 넣어줌
# feed_dict = {X : x, Y : y}
# s, h = sess.run([softmax_result, hypothesis], feed_dict = feed_dict)
# # []는 두개 전부 확인해보고 싶을때 값이 두개가 나옴
# print(s)
# print(h)

# Loss Function(Cross Entropy Error)
cost1 = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(softmax_result), axis = 1))
# Y : [0 1 0 0 0 0..] softmax_result : [0.1 0.05 0.3 ...]
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hypothesis, labels = Y))
# reduce_mean을 하는 이유 : Scalarfㅗ 값을 받고 싶어서

# x = tf.placeholder(tf.float32, [None,3])
# mean = tf.reduce_mean(x, axis = 1)
# # reduce_mean : [BATCH , 10]을 [BATCH]로 차원을 줄여는 역할

# sess = tf.Session()
# print(sess.run(mean, feed_dict = {x : [[1.5, 0.5, 1.0],[1., 2., 3.]]}))
#
# Gradient Descent
learning_rate = 0.001
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
# 최적화 기법과 Learning Rate 설정

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# Layer에 대해 랜덤값 부여

train_epochs = 15
batch_size = 100

# for epoch in range(train_epochs):
#     avg_cost = 0
#     total_batch = int(mnist.train.num_examples / batch_size)
#     # total_batch : 한 epoch에서 몇번의 반복이 일어나는지
#
#     for i in range(total_batch):
#         batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#         # next_batch : train_data에서 batch_size만큼 가져오는 것
#         feed_dict = {X : batch_xs, Y : batch_ys} # 한 epoch당 배치사이즈로 나누어진 데이터를 학습
#         c, _ = sess.run([cost, optimizer], feed_dict = feed_dict)
#         avg_cost += c / total_batch
#         # 평균적인 cost
#
#     print('Epoch : ', '%4d' %(epoch + 1), 'training cost = ', '{:9f}'.format(avg_cost))
#
max = 0
early_stopped_time = 0

for epoch in range(train_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs , batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X : batch_xs, Y : batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict = feed_dict)
        avg_cost += c / total_batch
    print('Epoch : ', '%4d' %(epoch + 1), 'training cost = ', '{:9f}'.format(avg_cost))
    # Accuracy

    correct_prediction = tf.equal(tf.argmax(hypothesis,1), tf.argmax(Y, 1))
    # argmax : 가장 큰 값의 인덱스를 가져오는것이고 두번째 값을 가져와

    # 추정값의 가장 큰 값을 가져와서 실제값과 비교하는 것 (위치 비교)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # correct_prediction은 현재 [True, False ..] 형태이므로 형변환을 해야함
    # tf.cast : 데이터의 형 변환
    test_accuracy = sess.run(accuracy, feed_dict = {X : mnist.test.images, Y : mnist.test.labels})

    print('Test Accuracy : ', test_accuracy)
    if test_accuracy > max:
        max = test_accuracy
        early_stopped_time = epoch + 1
print('Test Max Accuracy :', max)
print('Early stopped Time : ', early_stopped_time)
