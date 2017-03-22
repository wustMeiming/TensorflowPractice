
# coding: utf-8

# In[1]:

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


# In[2]:

train_x, train_y, test_x, test_y = tflearn.datasets.mnist.load_data(one_hot=True)

train_x = train_x.reshape(-1, 28, 28, 1)
test_x = test_x.reshape(-1, 28, 28, 1)


# In[3]:

# 定义神经网络模型
conv_net = input_data(shape=[None, 28, 28, 1], name='input')
conv_net = conv_2d(conv_net, 32, 2, activation='relu')
conv_net = max_pool_2d(conv_net, 2)
conv_net = conv_2d(conv_net, 64, 2, activation='relu')
conv_net = max_pool_2d(conv_net, 2)
conv_net = fully_connected(conv_net, 1024, activation='relu')
conv_net = dropout(conv_net, 0.8)
conv_net = fully_connected(conv_net, 10, activation='softmax')
conv_net = regression(conv_net, optimizer='adam', loss='categorical_crossentropy', name='output')


# In[4]:

model = tflearn.DNN(conv_net)


# In[5]:

# 训练
model.fit({'input':train_x}, {'output':train_y}, n_epoch=13,
         validation_set=({'input':test_x},{'output':test_y}),
         snapshot_step=300, show_metric=True, run_id='mnist')


# In[6]:

model.save('mnist.model') # 保存模型


# In[8]:

'''
model.load('mnist.model') # 加载模型
model.predict([test_x[1]])# 预测
'''

