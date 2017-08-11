import tensorflow as tf
import numpy as np
L = tf.layers

from models.activations import *
from models.customlayers import *

# def make_filter(size, n_channels, n_filters):
#     shape = [size, size, n_channels, n_filters]
#     lim = 1 / np.sqrt(n_channels)
#     initial = tf.random_uniform(shape, -lim, lim)
#     return tf.Variable(initial)
#
#
# def tied_encoder(image):
#     """
#     Adapted from https://github.com/pkmital/tensorflow_tutorials/blob/master/python/09_convolutional_autoencoder.py
#     """
#     n_filters = [16, 32, 32, 64]
#     filter_sizes = (3, 3, 5, 5)
#     latent_size = 1024
#
#     encoder_ = []
#     shapes_ = []
#
#     current_input = image
#     for l, nfil in enumerate(n_filters[1:]):
#         current_input_shape = current_input.get_shape().as_list()
#         n_channels = current_input_shape[3]
#         shapes_.append(current_input_shape)
#         W = make_filter(filter_sizes[l], n_channels, nfil)
#         b = tf.Variable(tf.zeros([nfil]))
#         encoder.append(W)
#         output = lrelu(
#             tf.add(current_input, W, strides=[1, 1, 1, 1], padding='VALID', b)
#         )
#         current_input = output
#
#     # reshape and pass to dense layer
#     current_input_shape = current_input.get_shape().as_list()
#     shapes_.append(current_input_shape)
#     batchsize = current_input_shape[0]
#     new_dim = np.prod(current_input_shape[1:])
#     reshaped = tf.reshape(current_input, [batchsize, new_dim])
#
#     W_shape = [new_dim, latent_size]
#     lim = 1 / np.sqrt(latent_size)
#     initial = tf.random_uniform(W_shape, -lim, lim)
#     W = tf.Variable(initial)
#     b = tf.Variable(tf.zeros([latent_size]))
#
#     z = lrelu(tf.add(tf.matmul(reshaped, W), b))
#
#     return z, encoder_[::-1], shapes_[::-1]
#
# def tied_decoder(encoded, encoder_, shapes_):
#     for l, shape in enumerate(shapes_[1:]):       # backward pass over shapes
#         W = encoder[-(1+l)]
#         b = tf.Variable

def tied_decoder(encoded):
    """
    Variable scope should be same as encoder with reuse=True and corresponding names

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
        print(w_shape, current_shape)
        b = tf.Variable(tf.zeros([w_shape[2]]), name='{}/bias/decoder'.format(layer_name))
        kernel_shape = tf.stack(output_shape)
        output = lrelu(tf.add(
            tf.nn.conv2d_transpose(
                current_input, w, kernel_shape, strides=[1, 1, 1, 1], padding='VALID'
            ), b
        ))
        current_input = output

    return current_input

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

    dense1 = dense_reshape(conv4, name='dense1', units=1024, activation=lrelu)
    return dense1


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

lstm_cell = PTLSTMCell(num_units=1024, activation=lambda x: 1.05*tf.nn.tanh(x))
