
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np


# In[2]:

# 下载 mnist 数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/', one_hot = True)


# In[3]:

# 一张图片是28*28，FNN是一次性把数据输入到网络，RNN把它分成块
chunk_size = 28
chunk_n = 28

rnn_size = 256

n_output_layer = 10 # 输出层


# In[4]:

X = tf.placeholder(tf.float32, [None, chunk_n, chunk_size])
Y = tf.placeholder(tf.float32)


# In[5]:

# 定义待训练的神经网络
def recurrent_neural_network(data):
    layer = {'w_':tf.Variable(tf.random_normal([rnn_size, n_output_layer])),
             'b_':tf.Variable(tf.random_normal([n_output_layer]))}
    
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    
    data = tf.transpose(data, [1, 0, 2])
    data = tf.reshape(data, [-1, chunk_size])
    data = tf.split(data, chunk_n, 0)
    outputs, status = tf.contrib.rnn.static_rnn(lstm_cell, data, dtype=tf.float32)
    
    output = tf.add(tf.matmul(outputs[-1], layer['w_']), layer['b_'])
    
    return output


# In[6]:

# 每次使用100调数据进行训练
batch_size = 100


# In[7]:

# 使用数据训练神经网络
def train_neural_network(X, Y):
    predict = recurrent_neural_network(X)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)
    
    epochs = 13
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        epoch_loss = 0
        for epoch in range(epochs):
            for i in range(int(mnist.train.num_examples/batch_size)):
                x, y = mnist.train.next_batch(batch_size)
                x = x.reshape([batch_size, chunk_n, chunk_size])
                _, c = session.run([optimizer, cost_func], feed_dict={X:x,Y:y})
                epoch_loss += c
            print(epoch, ' : ', epoch_loss)
        
        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        print('准确率：', accuracy.eval({X:mnist.test.images.reshape(-1, chunk_n, chunk_size), Y:mnist.test.labels}))      


# In[8]:

train_neural_network(X, Y)

