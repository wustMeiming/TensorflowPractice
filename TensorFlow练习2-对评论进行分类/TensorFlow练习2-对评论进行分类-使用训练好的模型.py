
# coding: utf-8

# In[1]:

import tensorflow as tf
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np


# In[2]:

f = open('lexcion.pickle', 'rb')
lex = pickle.load(f)
f.close()


# In[3]:

n_input_layer = len(lex) # 输入层

n_layer_1 = 2000 # hide layer
n_layer_2 = 2000 # hide layer(隐藏层)听着很神秘，其实就是除输入输出层外的中间层

n_output_layer = 3 # 输出层


# In[4]:

def neural_network(data):
   # 定义第一层"神经元"的权重和biases
    layer_1_w_b = {'w_':tf.Variable(tf.random_normal([n_input_layer, n_layer_1])), 'b_':tf.Variable(tf.random_normal([n_layer_1]))}
    # 定义第二层"神经元"的权重和biases
    layer_2_w_b = {'w_':tf.Variable(tf.random_normal([n_layer_1, n_layer_2])), 'b_':tf.Variable(tf.random_normal([n_layer_2]))}
    # 定义输出层"神经元"的权重和biases
    layer_output_w_b = {'w_':tf.Variable(tf.random_normal([n_layer_2, n_output_layer])), 'b_':tf.Variable(tf.random_normal([n_output_layer]))}

    # w·x+b
    layer_1 = tf.add(tf.matmul(data, layer_1_w_b['w_']), layer_1_w_b['b_'])
    layer_1 = tf.nn.relu(layer_1)  # 激活函数
    layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w_']), layer_2_w_b['b_'])
    layer_2 = tf.nn.relu(layer_2 ) # 激活函数
    layer_output = tf.add(tf.matmul(layer_2, layer_output_w_b['w_']), layer_output_w_b['b_'])
    
    return layer_output 


# In[5]:

X = tf.placeholder('float')


# In[6]:

def prediction(tweet_text):
    predict = neural_network(X)
    
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(session, 'model.ckpt')
        
        lemmatizer = WordNetLemmatizer()
        words = word_tokenize(tweet_text.lower())
        words = [lemmatizer.lemmatize(word) for word in words]
        
        features = np.zeros(len(lex))
        for word in words:
            if word in lex:
                features[lex.index(word)] = 1
        
        #print(predict.eval(feed_dict={X:[features]})) [[val1, val2, val3]]
        res = session.run(tf.argmax(predict.eval(feed_dict={X:[features]}), 1))
        return res


# In[7]:

prediction('I am very happe')

