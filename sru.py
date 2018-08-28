import tensorflow as tf
from tensorflow.python.util import nest


class SimpleSRUCell(tf.contrib.rnn.RNNCell):
    """Implements a simple distribution based recurrent unit that keeps moving
    averages of the mean map embeddings of features of inputs.
    """

    def __init__(self, num_stats, mavg_alphas, output_dims, recur_dims,
                 summarize=True, learn_alphas=False, linear_out=False,
                 include_input=False, activation=tf.nn.relu):
        self._num_stats = num_stats
        self._output_dims = output_dims
        self._recur_dims = recur_dims
        if learn_alphas:
            init_logit_alphas = -tf.log(1.0/mavg_alphas-1)
            logit_alphas = tf.get_variable(
                'logit_alphas', initializer=init_logit_alphas
            )
            self._mavg_alphas = tf.reshape(tf.sigmoid(logit_alphas), [1, -1, 1])
        else:
            self._mavg_alphas = tf.reshape(mavg_alphas, [1, -1, 1])
        self._nalphas = int(mavg_alphas.get_shape()[0])
        self._summarize = summarize
        self._linear_out = linear_out
        self._activation = activation
        self._include_input = include_input

    @property
    def state_size(self):
        return int(self._nalphas * self._num_stats)

    @property
    def output_size(self):
        return self._output_dims

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # Make statistics on input.
            if self._recur_dims > 0:
                recur_output = self._activation(_linear(
                    state, self._recur_dims, True, scope='recur_feats'
                ), name='recur_feats_act')
                stats = self._activation(_linear(
                    [inputs, recur_output], self._num_stats, True, scope='stats'
                ), name='stats_act')
            else:
                stats = self._activation(_linear(
                    inputs, self._num_stats, True, scope='stats'
                ), name='stats_act')
            # Compute moving averages of statistics for the state.
            with tf.variable_scope('out_state'):
                state_tensor = tf.reshape(
                    state, [-1, self._nalphas, self._num_stats], 'state_tensor'
                )
                stats_tensor = tf.reshape(
                    stats, [-1, 1, self._num_stats], 'stats_tensor'
                )
                out_state = tf.reshape(self._mavg_alphas*state_tensor +
                                       (1-self._mavg_alphas)*stats_tensor,
                                       [-1, self.state_size], 'out_state')
            # Compute the output.
            if self._include_input:
                output_vars = [out_state, inputs]
            else:
                output_vars = out_state
            output = _linear(
                output_vars, self._output_dims, True, scope='output'
            )
            if not self._linear_out:
                output = self._activation(output, name='output_act')
            return (output, out_state)


# No longer publicly expose function in tensorflow.
def _linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".

    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError(
                "Linear is expecting 2D arguments: %s" %
                str(shapes))
        if not shape[1]:
            raise ValueError(
                "Linear expects shape[1] of arguments: %s" %
                str(shapes))
        else:
            total_arg_size += shape[1]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable(
            "Matrix", [total_arg_size, output_size], dtype=dtype)
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(args, 1), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [output_size],
            dtype=dtype,
            initializer=tf.constant_initializer(bias_start, dtype=dtype)
        )
    return res + bias_term
