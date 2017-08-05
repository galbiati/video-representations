import tensorflow as tf
L = tf.layers

from models.activations import *
from models.customlayers import *

# note: you've hardcoded the sequence length here; come back to figure out
#       how to make it smarter (ie, move OUTSIDE function)

def encoder(image):
    input_layer = tf.reshape(image, (-1, 60, 80, 3))

    conv1 = L.conv2d(
        input_layer, name='conv1',
        filters=16, kernel_size=3, activation=lrelu
    )

    conv2 = L.conv2d(
        conv1, name='conv2',
        filters=32, kernel_size=5, activation=lrelu
    )

    shape_ = conv2.get_shape().as_list()
    newdim = shape_[1] * shape_[2] * shape_[3]
    print(shape_, newdim)

    dense1 = dense_reshape(conv2, name='dense1', units=512, activation=lrelu)

    return tf.reshape(dense1, (-1, 64, 512))    # hardcoded seq length


def decoder(encoded):
    encoded_reshaped = tf.reshape(encoded, (-1, 512))

    dense1 = L.dense(encoded_reshaped, units=127872, activation=lrelu, name='dense1')

    dense1_reshaped = tf.reshape(dense1, (-1, 54, 74, 32))

    deconv3 = L.conv2d_transpose(
        dense1_reshaped, name='deconv3',
        filters=32, kernel_size=5, activation=lrelu,
    )

    deconv4 = L.conv2d_transpose(
        deconv3, name='deconv4',
        filters=3, kernel_size=3, activation=None,
    )

    deconv4_reshaped = tf.transpose(deconv4, perm=(0, 3, 1, 2))

    return deconv4_reshaped

lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=256)
