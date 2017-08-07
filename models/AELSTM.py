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
        filters=16, kernel_size=3, activation=selu
    )

    conv2 = L.conv2d(
        conv1, name='conv2',
        filters=32, kernel_size=3, activation=selu
    )

    conv3 = L.conv2d(
        conv2, name='conv3',
        filters=32, kernel_size=5, activation=selu
    )

    conv4 = L.conv2d(
        conv3, name='conv4',
        filters=64, kernel_size=5, activation=selu
    )
    shape_ = conv4.get_shape().as_list()
    newdim = shape_[1] * shape_[2] * shape_[3]
    print(shape_, newdim)

    dense1 = dense_reshape(conv4, name='dense1', units=1024, activation=tf.nn.tanh)

    return tf.reshape(dense1, (-1, 64, 1024))    # hardcoded seq length


def decoder(encoded):
    encoded_reshaped = tf.reshape(encoded, (-1, 1024))

    dense1 = L.dense(encoded_reshaped, units=208896, activation=selu, name='dense1')

    dense1_reshaped = tf.reshape(dense1, (-1, 48, 68, 64))

    deconv1 = L.conv2d_transpose(
        dense1_reshaped, name='deconv1',
        filters=32, kernel_size=5, activation=selu
    )

    deconv2 = L.conv2d_transpose(
        deconv1, name='deconv2',
        filters=32, kernel_size=5, activation=selu
    )

    deconv3 = L.conv2d_transpose(
        deconv2, name='deconv3',
        filters=32, kernel_size=3, activation=selu,
    )

    deconv4 = L.conv2d_transpose(
        deconv3, name='deconv4',
        filters=3, kernel_size=3, activation=lrelu,
    )

    deconv4_reshaped = tf.transpose(deconv4, perm=(0, 3, 1, 2))

    return deconv4_reshaped

lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=1024)
