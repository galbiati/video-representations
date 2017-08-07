import tensorflow as tf
import numpy as np
from models.activations import *
from tensorflow.python.ops import rnn_cell_impl as tfrnn
L = tf.layers

def dense_reshape(incoming, units, **kwargs):
    shape = incoming.get_shape().as_list()
    newdim = np.prod(shape[1:])
    reshaped = tf.reshape(incoming, (-1, newdim))
    return L.dense(reshaped, units=units, **kwargs)

def invert_layer(input, invlayer_in, inv_layer_out):
    return tf.gradient(inv_layer_out, inv_layer_in, input)

class LSTMCell(tfrnn.RNNCell):
    # modified from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell_impl.py
    # wanted to pass through input more transparently
    """
    Instead of internally projecting input, pass it through unmodified except
    by hidden state. Project instead to three gating functions, then directly
    multiply by hidden state
    """
    def __init__(self, num_units,
                initializer=None, forget_bias=1.0, state_is_tuple=True,
                activation=None, reuse=None):

        super(LSTMCell, self).__init__(reuse=reuse)
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
            num_proj = self._num_units

            c_prev = tf.slice(state, [0, 0], [-1, self._num_units])
            m_prev = tf.slice(state, [0, self._num_units], [-1, num_proj])

            dtype = inputs.dtype
            input_size = inputs.get_shape().with_rank(2)[1]

            assert input_size.value is not None

            scope = tf.get_variable_scope()
            with tf.variable_scope(scope, initializer=self._initializer):
                lstm_matrix = tfrnn._linear([inputs, m_prev], 3*self._num_units, bias=True)
                i, j, f = tf.split(lstm_matrix, 3, axis=1)

                c = tf.nn.sigmoid(f + self._forget_bias) * c_prev # forget from hidden state
                c = c + tf.nn.sigmoid(i) * self._activation(j) # update hidden state with gated input
                m = inputs * self._activation(c)

            new_state = (LSTMStateTuple(c, m))
            return m, new_state
