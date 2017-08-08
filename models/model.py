import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl as tfrnn

class Model(object):
    # add batchsize and sequence length to model object
    def __init__(self, encoder, cell, decoder, batchsize, seqlen):
        """batchsize is necessary for reshaping"""
        self.encoder = encoder
        self.cell = cell
        self.decoder = decoder
        self.batchsize = batchsize
        self.seqlen = seqlen
        self.stacked_shape = (batchsize*seqlen, 60, 80, 3)
        self.sequence_shape = (batchsize, seqlen, 60, 80, 3)

    def stack(self, tensor):
        """
        Stacks batches and sequences to
        transform tensor to be shape (batchsize*seqlen, ...)
        """
        shape = tensor.get_shape().as_list()
        if shape[0] == self.batchsize:
            ndims = len(shape)
            new_shape = [self.batchsize*self.seqlen] + [shape[i] for i in range(2, ndims)]
            return tf.reshape(tensor, tuple(new_shape))
        else:
            return tensor

    def unstack(self, tensor):
        """
        Unstacks batches and sequences to transform tensor to be shape
        (batchsize, seqlen, ...)
        """
        shape = tensor.get_shape().as_list()
        if shape[0] == self.batchsize * self.seqlen:
            ndims = len(shape)
            new_shape = [self.batchsize, self.seqlen] + [shape[i] for i in range(1, ndims)]
            return tf.reshape(tensor, tuple(new_shape))
        else:
            return tensor

    def build(self, inputs, reuse=False):
        """
        Build the model on provided inputs

        Pass reuse=True to build the model with shared weights
        (eg for validation)
        """
        inputs = self.stack(inputs)
        with tf.variable_scope('encoder', reuse=reuse):
            encoded = self.encoder(inputs)
            self.latent_size = encoded.get_shape().as_list()[-1]
            # encoded = tf.reshape(encoded, (self.batchsize, self.seqlen, self.latent_size))
            encoded = self.unstack(encoded)

        with tf.variable_scope('lstm', reuse=reuse):
            # initialize hidden state with ones instead of zeros to ensure pass-through at start
            initial_state = tfrnn.LSTMStateTuple(
                tf.ones((self.batchsize, self.latent_size)),
                tf.zeros((self.batchsize, self.latent_size))
            )
            transitioned = tf.nn.dynamic_rnn(
                self.cell, encoded,
                initial_state = initial_state,
                sequence_length=[self.seqlen]*self.batchsize,
                dtype=tf.float32, swap_memory=True,
            )

            transitioned_ = self.unstack(transitioned[0])

        with tf.variable_scope('decoder', reuse=reuse):
            decoded = self.decoder(transitioned_)
            decoded = self.unstack(decoded)

        return encoded, transitioned[0], decoded

    def build_target_encoder(self, targets, reuse=True):
        """
        For LSTM-Encoder training, we need encodings of targets as well
        using same weights as encoder
        """
        targets = self.stack(targets)
        with tf.variable_scope('encoder', reuse=reuse):
            targeted = self.encoder(targets)
            targeted = self.unstack(targeted)

        return targeted
