import tensorflow as tf
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
    ## The below lines are useful for debugging when modifying architecture
    # shape_ = conv4.get_shape().as_list()
    # newdim = shape_[1] * shape_[2] * shape_[3]
    # print(shape_, newdim)

    dense1 = dense_reshape(conv4, name='dense1', units=1024, activation=selu)
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
        filters=32, kernel_size=5, activation=selu
    )

    deconv3 = L.conv2d_transpose(
        deconv4, name='conv3',
        filters=32, kernel_size=5, activation=selu
    )

    deconv2 = L.conv2d_transpose(
        deconv3, name='conv2',
        filters=32, kernel_size=3, activation=selu,
    )

    deconv1 = L.conv2d_transpose(
        deconv2, name='conv1',
        filters=3, kernel_size=3, activation=selu,
    )

    return deconv1

lstm_cell = PTLSTMCell(num_units=1024)
