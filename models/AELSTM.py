import tensorflow as tf
import numpy as np
L = tf.layers

from models.activations import *
from models.customlayers import *

def encoder(image):
    """
    Creates an encoder connected to the input tensor

    Args:
    --------
    :image must be a rank 4 tensor

    Outputs:
    --------
    :dense1 is a rank 2 tensor from the final layer of the encoder
    """

    conv1 = L.conv2d(
        image, name='conv1',
        filters=16, kernel_size=3, activation=lrelu
    )

    conv2 = L.conv2d(
        conv1, name='conv2',
        filters=32, kernel_size=3, activation=lrelu
    )

    conv3 = L.conv2d(
        conv2, name='conv3',
        filters=32, kernel_size=5, activation=lrelu
    )

    conv4 = L.conv2d(
        conv3, name='conv4',
        filters=64, kernel_size=5, activation=lrelu
    )
    ## The below lines are useful for debugging when modifying architecture
    # shape_ = conv4.get_shape().as_list()
    # newdim = shape_[1] * shape_[2] * shape_[3]
    # print(shape_, newdim)

    dense1 = dense_reshape(conv4, name='dense1', units=2048, activation=lrelu)
    return dense1

def tied_decoder(encoded):
    """
    Variable scope should be same as encoder with reuse=True and corresponding names
    Adapted from https://github.com/pkmital/tensorflow_tutorials/blob/master/python/09_convolutional_autoencoder.py

    There's a lot of hard-coding here :/
    """
    W = dict()
    W['dense1'] = tf.transpose(tf.get_variable('dense1/kernel'))
    w_shape = W['dense1'].get_shape().as_list()
    default_graph = tf.get_default_graph()
    current_input = encoded
    current_shape = encoded.get_shape().as_list()
    b = tf.Variable(tf.zeros([w_shape[-1]]), name='dense1/bias/decoder')

    dense1 = tf.add(tf.matmul(current_input, W['dense1']), b)
    conv_output = default_graph.get_tensor_by_name('encoder/conv4/convolution:0')
    batch, xdim, ydim, chans = conv_output.get_shape().as_list()
    current_input = tf.reshape(dense1, (-1, xdim, ydim, chans))

    for i in range(4):
        layer_name = 'conv{}'.format(4-i)
        var = tf.get_variable('{}/kernel'.format(layer_name))
        w = var
        W[layer_name] = w
        w_shape = w.get_shape().as_list()

        current_shape = current_input.get_shape().as_list()

        if i < 3:
            conv_output = default_graph.get_tensor_by_name('encoder/conv{}/convolution:0'.format(3-i))
        else:
            conv_output = default_graph.get_tensor_by_name('encoder/Reshape:0')

        output_shape = conv_output.get_shape().as_list()
        b = tf.Variable(tf.zeros([w_shape[2]]), name='{}/bias/decoder'.format(layer_name))
        kernel_shape = tf.stack(output_shape)
        output = lrelu(tf.add(
            tf.nn.conv2d_transpose(
                current_input, w, kernel_shape, strides=[1, 1, 1, 1], padding='VALID'
            ), b
        ))
        current_input = output

    return current_input


def decoder(encoded):
    """
    Creates a decoder that reconstructs an image from a latent vector
    using transposed convolutions.

    Args:
    -------
    :encoded is a batch of latent vectors (rank 2 tensor)

    Outputs:
    :deconv4 is a batch of images (rank 4 tensor)
    """
    dense1 = L.dense(encoded, units=208896, activation=selu, name='dense1')

    dense1_reshaped = tf.reshape(dense1, (-1, 48, 68, 64))

    deconv4 = L.conv2d_transpose(
        dense1_reshaped, name='conv4',
        filters=32, kernel_size=5, activation=lrelu
    )

    deconv3 = L.conv2d_transpose(
        deconv4, name='conv3',
        filters=32, kernel_size=5, activation=lrelu
    )

    deconv2 = L.conv2d_transpose(
        deconv3, name='conv2',
        filters=32, kernel_size=3, activation=lrelu,
    )

    deconv1 = L.conv2d_transpose(
        deconv2, name='conv1',
        filters=3, kernel_size=3, activation=lrelu,
    )

    return deconv1

lstm_cell = PTLSTMCell(num_units=2048, activation=lambda x: 1.05*tf.nn.tanh(x))
