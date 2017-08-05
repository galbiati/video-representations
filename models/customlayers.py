import tensorflow as tf
import numpy as np
L = tf.layers

def dense_reshape(incoming, units, **kwargs):
    shape = incoming.get_shape().as_list()
    newdim = np.prod(shape[1:])
    reshaped = tf.reshape(incoming, (-1, newdim))
    return L.dense(reshaped, units=units, **kwargs)

def invert_layer(input, invlayer_in, inv_layer_out):
    return tf.gradient(inv_layer_out, inv_layer_in, input)
