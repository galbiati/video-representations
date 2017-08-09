import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl as tfrnn

class Model(object):
    """
    Model is a wrapper for the full LSTM Encoder model.

    // Future: modify to allow variable-length sequences using clever slicing,
                rather than reshaping
    // Future: can model be made into a context manager? might be a handy way
                to manage graphs

    __init__ args:
    :encoder is a function that returns a batch of image encodings (rank 2 tensor)
    :cell is a recurrent neural network cell that can be passed to tf.nn.rnn_cell.dynamic_rnn
    :decoder is a function that returns a batch of decoded images (rank 4 tensor)
    :batchsize is the size of batches (necessary for proper reshaping)
    :seqlen is the length of sequences (necessary for proper reshaping)
    """

    def __init__(self, encoder, cell, decoder, batchsize, seqlen):
        self.encoder = encoder
        self.cell = cell
        self.decoder = decoder
        self.batchsize = batchsize
        self.seqlen = seqlen
        self.stacked_shape = (batchsize*seqlen, 60, 80, 3)
        self.sequence_shape = (batchsize, seqlen, 60, 80, 3)

    def stack(self, tensor):
        """
        Stacks batches and sequences to transform tensor to be shape (batchsize*seqlen, ...)
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
        Unstacks batches and sequences to transform tensor to be shape (batchsize, seqlen, ...)
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

        Args:
        -------
        :inputs is a tensor of shape (batchsize, sequence length, x, y, channels)
        :reuse can be used to share weights across graph for (eg) validation

        Outputs:
        -------
        :encoded is a (batchsize, sequence length, latent size) tensor of image encodings
        :transitioned[0] is a (batchsize, sequence length, latent size) tensor of predicted next-frame encodings
        :decoded is a (batchsize, sequence length, x, y, channels) tensor of predicted frames
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

            transitioned_ = self.stack(transitioned[0])

        with tf.variable_scope('decoder', reuse=reuse):     # sshfs may have gummed change; this might need to be 'encoder' with reuse=True
            decoded = self.decoder(transitioned_)
            decoded = self.unstack(decoded)

        return encoded, transitioned[0], decoded

class AEModel(Model):
    """
    Encoder and decoder without intermediate LSTM cell
    """
    def __init__(self, encoder, decoder, batchsize, seqlen):
        # should probably rewrite Model base class that doesn't assume LSTM cell
        super(AEModel, self).__init__(encoder, None, decoder, batchsize, seqlen)

    def build(self, inputs, reuse=False):
        inputs = self.stack(inputs)
        with tf.variablce_scope('encoder', reuse=reuse):
            encoded = self.encoder(inputs)
            self.latent_size = encoded.get_shape().as_list()[-1]

        with tf.variable_scope('decoder', reuse=reuse):
            decoded = self.decoder(encoded)
            decoded = self.unstack(decoded)

        return encoded, decoded
