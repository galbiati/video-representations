import tensorflow as tf
import numpy as np
from models.activations import *
from tensorflow.python.ops import rnn_cell_impl as tfrnn
L = tf.layers

## Convenience functions ##

def dense_reshape(incoming, units, **kwargs):
    """
    Convenience function for making a fc layer on top of feature maps

    Args:
    ------
    :incoming should be a set of feature maps (rank 4 tensor)
    :units should be the number of hidden units in the fully connected layer
    :kwargs are passed through to tf.layers.dense

    Outputs:
    --------
    :tf.layers.dense output (rank 2 tensor)
    """
    shape = incoming.get_shape().as_list()
    newdim = np.prod(shape[1:])
    reshaped = tf.reshape(incoming, (-1, newdim))
    return L.dense(reshaped, units=units, **kwargs)


## Tying encoder-decoder ##

def invert_layer(input, inv_layer_out, inv_layer_in):
    """
    Inverts a layer by using its gradient;
    see Lasagne.layers.InverseLayer for a similar implementation in Theano

    This currently does not work, but remains a point of interest. The gradient
    may simply need to be transposed; I will return to this when there is time.

    Args:
    -------
    :input is the input to the current layer
    :inv_layer_out is the output of the layer to be reversed
    :inv_layer_in is the input to the layer to be reversed

    Outputs:
    --------
    :the dot product of the input with the inverting gradient
    """
    return tf.tensordot(input, tf.gradients(inv_layer_out, inv_layer_in), 1)

## Custom RNN cells ##

class PTLSTMCell(tfrnn.RNNCell):

    """
    A "pass-through" LSTM cell that does not project input or pass it through
    a sigmoid squeeze.

    Based on LSTMCell implementation in tensorflow/python/ops/rnn_cell_impl.py,
    but removes many of the bells and whistles to expose core cell logic.

    Only works with LSTMSTateTuples!

    Best used by passing ones rather than zeros as initial state at
    tf.nn.rnn_cell.dynamic_rnn!

    __init__ args:
    ------
    :num_units is the size of the LSTM internal state. Mathematically,
                this must equal the size of the input vector.
    :initializer sets starting values for weights
    :forget_bias is the default bias on the forget gates
    :activation is nonlinearity for CELL STATE
    :reuse is reuse
    """
    def __init__(self, num_units,
                initializer=None, forget_bias=1.0,
                activation=None, reuse=None):

        super(PTLSTMCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._initializer = initializer
        self._forget_bias = forget_bias
        self._activation = activation or tf.nn.tanh # activation for STATE, not output
        self._state_size = (tfrnn.LSTMStateTuple(num_units, num_units))
        self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def call(self, inputs, state):
        """
        Performs PTLSTM calculation:
        1) concat inputs and previous output
        2) project to three hidden states of num_units
        3) apply forget to hidden state
        4) update hidden state by gated projection
        5) multiply inputs pointwise with hidden state after activation

        The logic is to have LSTM state encode motion/change information that
        modifies input, which should be very similar to output.

        Args:
        -------
        :inputs is a batch of sequences of encoded images (rank 3 tensor)
        :state is an LSTMStateTuple object wrapping
                the previous input and hidden state
        """
        input_size = inputs.get_shape().with_rank(2)[1]

        # assert the things
        assert input_size.value is not None
        assert inputs.get_shape().as_list()[-1] == self._num_units
        assert isinstance(state, tfrnn.LSTMStateTuple)

        c_prev, m_prev = state

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope, initializer=self._initializer):
            lstm_matrix = tfrnn._linear([inputs, m_prev], 3*self._num_units, bias=True)
            i, j, f = tf.split(lstm_matrix, 3, axis=1)

            c = tf.nn.sigmoid(f + self._forget_bias) * c_prev # forget from hidden state
            c = c + tf.nn.sigmoid(i) * self._activation(j) # update hidden state with gated input
            m = inputs * 1.1 * tf.nn.tanh(c)  # straight multiply inputs by hidden state filter

        new_state = tfrnn.LSTMStateTuple(c, m)  # must sometimes wrap as tuple?
        return m, new_state
