import tensorflow as tf
from tensorflow.python.framework import ops
from model import Model

class ATWModel(Model):
    """
    ATWModel implements a variant on the E-T-D model from model.Model()

    Instead of doing next frame prediction, ATW attempts to encode the entire
    sequence, then reproduce the video from only the final latent vectors.

    __init__ args:
    :encoder is a function that returns a batch of image encodings (rank 2 tensor)
    :cell is a recurrent neural network cell that can be passed to tf.nn.rnn_cell.dynamic_rnn
    :decoder is a function that returns a batch of decoded images (rank 4 tensor)
    :latent_size is the size of the latent space
    :activation is the activation function for the LSTM cells
    :batchsize is the size of batches (necessary for proper reshaping)
    :seqlen is the length of sequences (necessary for proper reshaping)
    """

    def __init__(self, encoder, cell, decoder,
                 latent_size, activation,
                 batchsize, seqlen):

        self.latent_size = latent_size
        self.encoder = lambda inputs: encoder(inputs, latent_size=latent_size)
        self.cell_fw = cell(num_units=latent_size, activation=activation)
        self.cell_bw = cell(num_units=latent_size, activation=activation)
        self.decoder = decoder
        self.batchsize = batchsize
        self.seqlen = seqlen
        self.stacked_shape = (batchsize*seqlen, 60, 80, 3)
        self.sequence_shape = (batchsize, seqlen, 60, 80, 3)

    def build(self, inputs, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse):
            inputs = self.stack(inputs)

            encoded = self.encoder(inputs)
            encoded = self.unstack(encoded)

        with tf.variable_scope('lstm', reuse=reuse):
            # initialize hidden state with ones instead of zeros to ensure pass-through at start
            initial_state = tfrnn.LSTMStateTuple(
                tf.ones((self.batchsize, self.latent_size)),
                tf.zeros((self.batchsize, self.latent_size))
            )

            # encoder pass
            _, seeds = tf.nn.dynamic_rnn(
                self.cell_fw, encoded,
                initial_state=initial_state,
                sequence_length=[self.seqlen]*self.batchsize,
                dtype=tf.float32, swap_memory=True,
            )

            # decoder pass
            def rnn_step(next_tuple, next_elem):
                input, state = next_tuple
                output, next_state = self.cell_fw(input, state)
                return (output, next_state)

            state = seeds
            next_input = state[1]
            elems = np.arange(self.seqlen)

            outputs, states = tf.scan(
                rnn_step, elems, (next_input, state),
                swap_memory=True
            )
            
            transitioned = tf.transpose(outputs, (1, 0, 2))

            transitioned_ = self.stack(transitioned)

        with tf.variable_scope('encoder', reuse=True):
            decoded = self.decoder(transitioned_)
            decoded = self.unstack(decoded)

        return encoded, transitioned, decoded
