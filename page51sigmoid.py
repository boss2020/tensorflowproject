import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow_datasets as tfds
import tensorflow as tf
x = tf.constant(value=[[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]])
W = tf.random.uniform([10,5],-0.1,0.1,tf.float32)
b = tf.Variable(tf.zeros(shape=[5],dtype=tf.float32))
#创建一个1*5的全零矩阵
h = tf.nn.sigmoid(x@W+b)
print(h)
