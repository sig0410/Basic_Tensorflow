import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)
img = mnist.train.images[0].reshape(28,28)
# train데이터에서 한개의 img만 가져오기
# 28*28로 변경
sess = tf.InteractiveSession()
# tf.InteractiveSession() : 기존의 Session과 다른점은 중간에 run없이 바로 값을 확인 가능
# Test해보는 용도로 많이 사용

# Convolution
img = img.reshape(-1, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 5], stddev = 0.01))
# Filter 정의
# Filter Structure : [W, H, d_in, filter 갯수]
conv2d = tf.nn.conv2d(img, W1, strides = [1, 2, 2, 1], padding = 'SAME')
# Convolution 정의
# strides : [1, W, H, 1]
print(conv2d)

sess.run(tf.global_variables_initializer())
conv2d_img = conv2d.eval()
# conv2d.eval : conv2d의 값을 그대로 가져오는 것
# sess.run()을 대신 한다고 생각
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(14,14), cmap = 'gray')

plt.show()
# Filter를 5개로 정의했기 때문에 각 Filter마다 나온 값들이 이미지형태로 나온다

# Pooling

mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)
img = mnist.train.images[0].reshape(28,28)
# 28*28로 변경
sess = tf.InteractiveSession()

# Convolution
img = img.reshape(-1, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 5], stddev = 0.01))
conv2d = tf.nn.conv2d(img, W1, strides = [1, 2, 2, 1], padding = 'SAME')
pool = tf.nn.max_pool(conv2d, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
print(pool)
sess.run(tf.global_variables_initializer())

pool_img = pool.eval()
pool_img = np.swapaxes(pool_img, 0, 3)
for i, one_img in enumerate(pool_img):
    plt.subplot(1, 5, i+1), plt.imshow(one_img.reshape(7, 7), cmap = 'gray')

plt.show()