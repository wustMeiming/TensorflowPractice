
# coding: utf-8

# In[1]:

import os
import random
import tensorflow as tf
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# In[2]:

f = open('lexcion.pickle', 'rb')
lex = pickle.load(f)
f.close()


# In[3]:

def get_random_line(file, point):
    file.seek(point)
    file.readline()
    return file.readline()


# In[4]:

# 从文件中随机选择n条记录
def get_n_random_line(file_name, n=150):
    lines = []
    file = open(file_name, encoding='latin-1')
    total_bytes = os.stat(file_name).st_size 
    for i in range(n):
        random_point = random.randint(0, total_bytes)
        lines.append(get_random_line(file, random_point))
    file.close()
    return lines


# In[5]:

def get_test_dataset(test_file):
    with open(test_file, encoding='latin-1') as f:
        test_x = []
        test_y = []
        lemmatizer = WordNetLemmatizer()
        for line in f:
            label = line.split(':%:%:%:')[0]
            tweet = line.split(':%:%:%:')[1]
            words = word_tokenize(tweet.lower())
            words = [lemmatizer.lemmatize(word) for word in words]
            features = np.zeros(len(lex))
            for word in words:
                if word in lex:
                    features[lex.index(word)] = 1

            test_x.append(list(features))
            test_y.append(eval(label))
    return test_x, test_y


# In[7]:

test_x, test_y = get_test_dataset('testing.csv')


# In[8]:

##############################################################################
input_size = len(lex)
num_classes = 3


# In[9]:

X = tf.placeholder(tf.int32, [None, input_size])
Y = tf.placeholder(tf.float32, [None, num_classes])


# In[10]:

dropout_keep_prob = tf.placeholder(tf.float32)
 
batch_size = 90


# In[16]:

def neural_network():
    # embedding layer
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
        embedding_size = 128
        W = tf.Variable(tf.random_uniform([input_size, embedding_size], -1.0, 1.0))
        embedded_chars = tf.nn.embedding_lookup(W, X)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
    # convolution + maxpool layer
    num_filters = 128
    filter_sizes = [3,4,5]
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]))
            conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID")
            h = tf.nn.relu(tf.nn.bias_add(conv, b))
            pooled = tf.nn.max_pool(h, ksize=[1, input_size - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
            pooled_outputs.append(pooled)

    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    # dropout
    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
    # output
    with tf.name_scope("output"):
        W = tf.get_variable("W", shape=[num_filters_total, num_classes], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
        output = tf.nn.xw_plus_b(h_drop, W, b)

    return output


# In[14]:

def train_neural_network():
    output = neural_network()

    optimizer = tf.train.AdamOptimizer(1e-3)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars)

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        lemmatizer = WordNetLemmatizer()
        i = 0
        while True:
            batch_x = []
            batch_y = []

            #if model.ckpt文件已存在:
            #    saver.restore(session, 'model.ckpt')  恢复保存的session
            try:
                lines = get_n_random_line('training.csv', batch_size)
                for line in lines:
                    label = line.split(':%:%:%:')[0]
                    tweet = line.split(':%:%:%:')[1]
                    words = word_tokenize(tweet.lower())
                    words = [lemmatizer.lemmatize(word) for word in words]

                    features = np.zeros(len(lex))
                    for word in words:
                        if word in lex:
                            features[lex.index(word)] = 1  # 一个句子中某个词可能出现两次,可以用+=1，其实区别不大

                    batch_x.append(list(features))
                    batch_y.append(eval(label))

                _, loss_ = sess.run([train_op, loss], feed_dict={X:batch_x, Y:batch_y, dropout_keep_prob:0.5})
                print(loss_)
            except Exception as e:
                print(e)

            if i % 10 == 0:
                predictions = tf.argmax(output, 1)
                correct_predictions = tf.equal(predictions, tf.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))
                accur = sess.run(accuracy, feed_dict={X:test_x[0:50], Y:test_y[0:50], dropout_keep_prob:1.0})
                print('准确率:', accur)

            i += 1


# In[17]:

train_neural_network()

