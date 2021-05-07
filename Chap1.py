import tensorflow as tf

hello = tf.constant('bb')
sess = tf.Session()
print(sess.run(hello))

node1 = tf.constant(3.0, tf.float32)
# Constant : 상수
# node1은 3.0이라는 상수를 가지는 노드이다
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

print(node3)

sess = tf.Session()
# 활성화
print(sess.run([node1, node2]))
print('node3 : ', sess.run(node3))

# Placeholder : 프로그램 중 값을 변경할 수 있는 가역변수
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
sess = tf.Session()

print(sess.run(adder_node, feed_dict={a:3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a:[1,3], b: [2,4]}))

y = tf.add(a,b)
# adder_node를 이렇게 변경가능

sess = tf.Session()

print(sess.run(y, feed_dict={a:3, b: 4.5}))

x_data = [[1,2]]

X = tf.placeholder(tf.float32, shape = [None, 2])
# shape : 데이터가 몇개 들어올지 몰라서 None이고 데이터의 차원을 지정

W = tf.Variable(tf.random_normal([2,1]), name = 'weight')
# weight값 지정
# random_normal : 초기에 랜덤하게 정규분포를 따르도록 값을 정의, [2,1]은 2개의 input을 받아서 1개의 output을 도출

b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypotesis = tf.sigmoid(tf.matmul(X, W) + b)
# X와 W를 행렬곱하고 b를 더함
hypotesis1 = tf.nn.relu(tf.matmul(X, W) + b)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # with를 통해 session이 활성화되어 있는 부분을 지정가능
    # global_variables_initializer : 랜덤 변수 초기화해줌 (랜덤한 값을 넣어줌)

    prediction = sess.run(hypotesis1, feed_dict={X:x_data})
    print(prediction)
    # weight에서 2개의 input을 받아서 1개의 output을 도출하도록 정의했기 때문에 Scalar값이 나옴
    # sigmoid로 지정했기 때문에 0 ~ 1사이값이 나옴
    # relu함수를 지정하면 x가 양수면 그대로 나오고 음수이면 0으로 도