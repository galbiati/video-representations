import tensorflow as tf
L = tf.layers

from activations import *
from customlayers import *

def encoder(image):
    input_layer = tf.reshape(image, (-1, 60, 80, 3))

    conv1 = L.conv2d(
        input_layer, filters=16, kernel_size=3, activation=selu
    )

    conv2 = L.conv2d(
        conv1, filters=16, kernel_size=3, activation=selu
    )

    conv3 = L.conv2d(
        conv2, filters=16, kernel_size=5, activation=selu
    )

    conv4 = L.conv2d(
        conv3, filters=32, kernel_size=5, activation=selu
    )

    shape = conv4.get_shape().as_list()
    print(shape)
    newdim = shape[1] * shape[2] * shape[3]
    print(newdim)
    conv4_flat = tf.reshape(conv4, (-1, newdim))
    print(conv4_flat.shape)

    dense1 = dense_reshape(conv4, units=1024, activation=selu)

    return dense1

def decoder(encoded):
    dense1 = L.dense(encoded, units=104448, activation=selu)
    dense1_reshaped = tf.reshape(dense1, (-1, 48, 68, 32))

    deconv1 = L.conv2d_transpose(
        dense1_reshaped, filters=16, kernel_size=5, activation=selu
    )

    deconv2 = L.conv2d_transpose(
        deconv1, filters=16, kernel_size=5, activation=selu
    )

    deconv3 = L.conv2d_transpose(
        deconv2, filters=16, kernel_size=3, activation=selu
    )

    deconv4 = L.conv2d_transpose(
        deconv3, filters=3, kernel_size=3, activation=lrelu
    )

    deconv3_reshaped = tf.transpose(deconv4, perm=(0, 3, 1, 2))

    return deconv3_reshaped
