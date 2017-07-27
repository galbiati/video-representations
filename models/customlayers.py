import tensorflow as tf

def dense_reshape(incoming, units, **kwargs):
    shape = incoming.get_shape().as_list()
    newdim = np.prod(shape[1:])
    reshaped = tf.reshape(incoming, (-1, newdim))
    return L.dense(reshaped, units=units, **kwargs)
