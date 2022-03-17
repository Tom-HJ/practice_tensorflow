'''Test program for Tensorflow'''

'''import tensorflow as tf'''
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

A = tf.placeholder(tf.float32, None, 'A')
B = tf.placeholder(tf.float32, None, 'B')
C = A + B

with tf.Session() as s:
    ans = s.run(C, {A: 2.5, B: 2.0})
    print(ans)