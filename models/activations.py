import tensorflow as tf
from tensorflow.python.framework import ops

def selu(x):
    with ops.name_scope('elu'):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
    return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))

def lrelu(x, alpha=.03):
    return tf.maximum(alpha*x, x)
