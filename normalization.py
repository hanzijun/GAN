import tensorflow as tf
import numpy as np

def batchnormalization(matrix):

    var_a=tf. placeholder(tf.float32,matrix.shape)
    nor_a=tf.layers.batch_normalization(var_a,training=True)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    e_value=sess.run(nor_a, feed_dict={var_a: matrix})
    sess.close()
    return e_value

def MINMAXNormalization(matrix):

    amin, amax = matrix.min(), matrix.max()
    nor_a =(matrix-amin)/(amax-amin)
    return nor_a

def scalerNormalization(matrix):
    """"
    The Column Method
    """
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    nor_a=   min_max_scaler.fit_transform(matrix)
    return nor_a