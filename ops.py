
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

#定义偏置项
def bias(name, shape, bias_start = 0.0, trainable = True):

    dtype = tf.float32

    var = tf.get_variable(name, shape, tf.float32, trainable = trainable,
                          initializer = tf.constant_initializer(bias_start, dtype=dtype))
    return  var


#定义权重
def weight(name, shape, stddev = 0.02, trainable = True):

    dtype = tf.float32

    var = tf.get_variable(name, shape, tf.float32, trainable = trainable,
                          initializer=tf.random_normal_initializer(stddev = stddev, dtype =  dtype))

    return var



#定义全连接层
def full_connected(value, output_shape, name = 'full_connect', with_w = False):

    shape = value.get_shape().as_list()

    with tf.variable_scope(name):
        weights = weight('weight', [shape[1], output_shape], 0.02)
        biases = bias('bias', [output_shape], 0.0)

    if with_w:
        return tf.matmul(value, weights) + biases , weights, biases
    else:
        return tf.matmul(value, weights) + biases
    
    

#relu函数:  y = x  x>0;
#           y = 0  x<0
#lrelu函数: y = x   x>0
#           y = ax  x<0, a非常小，例如0.0002
#
def lrelu(x, leak = 0.2, name = 'lrelu'):

    with tf.variable_scope(name):
        return tf.maximum(x, leak*x, name = name)
    
    

#relu函数
def relu(value, name = 'relu'):

    with tf.variable_scope(name):
        return tf.nn.relu(value)
    
    

#解卷积层
def deconv2d(value, output_shape, k_h = 5, k_w = 5, strides = [1,2,2,1],
             name = 'deconv2d', with_w = False):

    with tf.variable_scope(name):
        weights = weight('weight', [k_h, k_w, output_shape[-1], value.get_shape()[-1]])
        deconv = tf.nn.conv2d_transpose(value, weights, output_shape, strides=strides)
        biases = bias('bias', [output_shape[-1]])

        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, weights, biases
        else:
            return deconv
        
        

#卷积层
def conv2d(value, output_dim, k_h = 5, k_w = 5, strides = [1,2,2,1], name = 'conv2d'):

    with tf.variable_scope(name):

        weights = weight('weight', [k_h, k_w, value.get_shape()[-1], output_dim])
        conv = tf.nn.conv2d(value, weights, strides = strides, padding='SAME')
        biases = bias('bias', [output_dim])

        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv
    
    

# 把约束条件串联到 feature map
def conv_cond_concat(value, cond, name = 'concat'):

    #拉直张量
    value_shape = value.get_shape().as_list()
    cond_shape = cond.get_shape().as_list()

    with tf.variable_scope(name):
        return tf.concat([value,
                             cond * tf.ones(value_shape[0:3] +
                                            cond_shape[3:])], 3)
    
    
    
    
# Batch Normalization 层 
def batch_norm_layer(value, is_train = True, name = 'batch_norm'):

    with tf.variable_scope(name) as scope:
        if is_train:
            return batch_norm(value, decay=0.9, epsilon=1e-5, scale = True,
                              is_training=is_train,
                              updates_collections=None, scope=scope)
        else:
            return batch_norm(value, decay=0.9, epsilon=1e-5, scale=True,
                              is_training=is_train, reuse=True,
                              updates_collections=None, scope = scope)




