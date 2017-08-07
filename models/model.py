import tensorflow as tf

class Model(object):
    def __init__(self, encoder, cell, decoder):
        self.encoder = encoder
        self.cell = cell
        self.decoder = decoder

    def build_full(self, inputs, batchsize):
        """Batchsize is necessary for reshaping"""
        # TODO: you've hardcoded 512; make responsive to input shapes

        inputs_ = tf.reshape(inputs, (-1, 60, 80, 3))
        with tf.variable_scope('encoder'):
            encoded = self.encoder(inputs_)
            encoded = tf.reshape(encoded, (batchsize, -1, 1024))

        with tf.variable_scope('lstm'):
            transitioned = tf.nn.dynamic_rnn(
                self.cell, encoded,
                sequence_length=[64]*batchsize,
                dtype=tf.float32, swap_memory=True
            )
            transitioned_ = tf.reshape(transitioned[0], (-1, 1024))

        with tf.variable_scope('decoder'):
            decoded = self.decoder(transitioned_)

        return encoded, transitioned, decoded

    def build_no_rnn(self, inputs, batchsize):
        inputs_ = tf.reshape(inputs, (-1, 60, 80, 3))

        with tf.variable_scope('encoder'):
            encoded = self.encoder(inputs_)

        with tf.variable_scope('decoder'):
            decoded = self.decoder(encoded)
            print(decoded.get_shape().as_list())

        return encoded, decoded
