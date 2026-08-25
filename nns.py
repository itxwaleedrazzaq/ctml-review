'''
A part of this code is taken from https://github.com/mlech26l/ode-lstms for testing purposes.
'''
import os
# os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
import numpy as np
from typing import Optional
from einops import rearrange
import math
from ncps.tf import LTCCell, CfC
from ncps.wirings import AutoNCP
from keras_nac.layers import FLUID, NAC


class CTRNNCell(tf.keras.layers.Layer):
    def __init__(self, units, method, num_unfolds=None, tau=1, **kwargs):
        self.fixed_step_methods = {
            "euler": self.euler,
            "heun": self.heun,
            "rk4": self.rk4,
        }
        allowed_methods = ["euler", "heun", "rk4", "dopri5"]
        if not method in allowed_methods:
            raise ValueError(
                "Unknown ODE solver '{}', expected one of '{}'".format(
                    method, allowed_methods
                )
            )
        if method in self.fixed_step_methods.keys() and num_unfolds is None:
            raise ValueError(
                "Fixed-step ODE solver requires argument 'num_unfolds' to be specified!"
            )
        self.units = units
        self.state_size = units
        self.num_unfolds = num_unfolds
        self.method = method
        self.tau = tau
        super(CTRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_dim = input_shape[0][-1]

        self.kernel = self.add_weight(
            shape=(input_dim, self.units), initializer="glorot_uniform", name="kernel"
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer="orthogonal",
            name="recurrent_kernel",
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.Zeros(),
            name="bias",
        )

        self.scale = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(1.0),
            name="scale",
        )
        if self.method == "dopri5":
            # Only load tfp packge if it is really needed
            import tensorflow_probability as tfp

            # We don't need the most precise solver to speed up training
            self.solver = tfp.math.ode.DormandPrince(
                rtol=0.01,
                atol=1e-04,
                first_step_size=0.01,
                safety_factor=0.8,
                min_step_size_factor=0.1,
                max_step_size_factor=10.0,
                max_num_steps=None,
                make_adjoint_solver_fn=None,
                validate_args=False,
                name="dormand_prince",
            )
        self.built = True

    def call(self, inputs, states):
        hidden_state = states[0]
        elapsed = 1.0
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            elapsed = inputs[1]
            inputs = inputs[0]

        if self.method == "dopri5":
            # Only load tfp packge if it is really needed
            import tensorflow_probability as tfp

            if not type(elapsed) == float:
                batch_dim = tf.shape(elapsed)[0]
                elapsed = tf.reshape(elapsed, [batch_dim])

                idx = tf.argsort(elapsed)
                solution_times = tf.gather(elapsed, idx)
            else:
                solution_times = elapsed
            hidden_state = states[0]
            res = self.solver.solve(
                ode_fn=self.dfdt_wrapped,
                initial_time=0,
                initial_state=hidden_state,
                solution_times=solution_times,  # tfp.math.ode.ChosenBySolver(elapsed),
                constants={"input": inputs},
            )
            if not type(elapsed) == float:
                i2 = tf.stack([idx, tf.range(batch_dim)], axis=1)
                hidden_state = tf.gather_nd(res.states, i2)
            else:
                hidden_state = res.states[-1]
        else:
            delta_t = elapsed / self.num_unfolds
            method = self.fixed_step_methods[self.method]
            for i in range(self.num_unfolds):
                hidden_state = method(inputs, hidden_state, delta_t)
        return hidden_state, [hidden_state]

    def dfdt_wrapped(self, t, y, **constants):
        inputs = constants["input"]
        hidden_state = y
        return self.dfdt(inputs, hidden_state)

    def dfdt(self, inputs, hidden_state):
        h_in = tf.matmul(inputs, self.kernel)
        h_rec = tf.matmul(hidden_state, self.recurrent_kernel)
        dh_in = self.scale * tf.nn.tanh(h_in + h_rec + self.bias)
        if self.tau > 0:
            dh = dh_in - hidden_state * self.tau
        else:
            dh = dh_in
        return dh

    def euler(self, inputs, hidden_state, delta_t):
        dy = self.dfdt(inputs, hidden_state)
        return hidden_state + delta_t * dy

    def heun(self, inputs, hidden_state, delta_t):
        k1 = self.dfdt(inputs, hidden_state)
        k2 = self.dfdt(inputs, hidden_state + delta_t * k1)
        return hidden_state + delta_t * 0.5 * (k1 + k2)

    def rk4(self, inputs, hidden_state, delta_t):
        k1 = self.dfdt(inputs, hidden_state)
        k2 = self.dfdt(inputs, hidden_state + k1 * delta_t * 0.5)
        k3 = self.dfdt(inputs, hidden_state + k2 * delta_t * 0.5)
        k4 = self.dfdt(inputs, hidden_state + k3 * delta_t)

        return hidden_state + delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


class LSTMCell(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = (units, units)
        self.initializer = "glorot_uniform"
        self.recurrent_initializer = "orthogonal"
        super(LSTMCell, self).__init__(**kwargs)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return (
            tf.zeros([batch_size, self.units], dtype=tf.float32),
            tf.zeros([batch_size, self.units], dtype=tf.float32),
        )

    def build(self, input_shape):
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_shape = (input_shape[0][-1] + input_shape[1][-1],)

        self.input_kernel = self.add_weight(
            shape=(input_shape[-1], 4 * self.units),
            initializer=self.initializer,
            name="input_kernel",
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, 4 * self.units),
            initializer=self.recurrent_initializer,
            name="recurrent_kernel",
        )
        self.bias = self.add_weight(
            shape=(4 * self.units),
            initializer=tf.keras.initializers.Zeros(),
            name="bias",
        )

        self.built = True

    def call(self, inputs, states):
        cell_state, output_state = states
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            inputs = tf.concat([inputs[0], inputs[1]], axis=-1)

        z = (
            tf.matmul(inputs, self.input_kernel)
            + tf.matmul(output_state, self.recurrent_kernel)
            + self.bias
        )
        i, ig, fg, og = tf.split(z, 4, axis=-1)

        input_activation = tf.nn.tanh(i)
        input_gate = tf.nn.sigmoid(ig)
        forget_gate = tf.nn.sigmoid(fg + 1.0)
        output_gate = tf.nn.sigmoid(og)

        new_cell = cell_state * forget_gate + input_activation * input_gate
        output_state = tf.nn.tanh(new_cell) * output_gate

        return output_state, [new_cell, output_state]


class ODELSTM(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = (units, units)
        self.initializer = "glorot_uniform"
        self.recurrent_initializer = "orthogonal"
        self.ctrnn = CTRNNCell(self.units, num_unfolds=4, method="euler")
        super(ODELSTM, self).__init__(**kwargs)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return (
            tf.zeros([batch_size, self.units], dtype=tf.float32),
            tf.zeros([batch_size, self.units], dtype=tf.float32),
        )

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_dim = input_shape[0][-1]

        self.ctrnn.build([self.units])
        self.input_kernel = self.add_weight(
            shape=(input_dim, 4 * self.units),
            initializer=self.initializer,
            name="input_kernel",
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, 4 * self.units),
            initializer=self.recurrent_initializer,
            name="recurrent_kernel",
        )
        self.bias = self.add_weight(
            shape=(4 * self.units,),
            initializer=tf.keras.initializers.Zeros(),
            name="bias",
        )

        self.built = True

    def call(self, inputs, states):
        cell_state, ode_state = states
        elapsed = 1.0
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            elapsed = inputs[1]
            inputs = inputs[0]

        z = (
            tf.matmul(inputs, self.input_kernel)
            + tf.matmul(ode_state, self.recurrent_kernel)
            + self.bias
        )
        i, ig, fg, og = tf.split(z, 4, axis=-1)

        input_activation = tf.nn.tanh(i)
        input_gate = tf.nn.sigmoid(ig)
        forget_gate = tf.nn.sigmoid(fg + 3.0)
        output_gate = tf.nn.sigmoid(og)

        new_cell = cell_state * forget_gate + input_activation * input_gate
        ode_input = tf.nn.tanh(new_cell) * output_gate

        ode_output, new_ode_state = self.ctrnn.call([ode_input, elapsed], [ode_state])

        return ode_output, [new_cell, new_ode_state[0]]


class CTGRU(tf.keras.layers.Layer):
    # https://arxiv.org/abs/1710.04110
    def __init__(self, units, M=8, **kwargs):
        self.units = units
        self.M = M
        self.state_size = units * self.M

        # Pre-computed tau table (as recommended in paper)
        self.ln_tau_table = np.empty(self.M)
        self.tau_table = np.empty(self.M)
        tau = 1.0
        for i in range(self.M):
            self.ln_tau_table[i] = np.log(tau)
            self.tau_table[i] = tau
            tau = tau * (10.0 ** 0.5)

        super(CTGRU, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_dim = input_shape[0][-1]

        self.retrieval_layer = tf.keras.layers.Dense(
            self.units * self.M, activation=None
        )
        self.detect_layer = tf.keras.layers.Dense(self.units, activation="tanh")
        self.update_layer = tf.keras.layers.Dense(self.units * self.M, activation=None)
        self.built = True

    def call(self, inputs, states):
        elapsed = 1.0
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            elapsed = inputs[1]
            inputs = inputs[0]

        batch_dim = tf.shape(inputs)[0]

        # States is actually 2D
        h_hat = tf.reshape(states[0], [batch_dim, self.units, self.M])
        h = tf.reduce_sum(h_hat, axis=2)
        states = None  # Set state to None, to avoid misuses (bugs) in the code below

        # Retrieval
        fused_input = tf.concat([inputs, h], axis=-1)
        ln_tau_r = self.retrieval_layer(fused_input)
        ln_tau_r = tf.reshape(ln_tau_r, shape=[batch_dim, self.units, self.M])
        sf_input_r = -tf.square(ln_tau_r - self.ln_tau_table)
        rki = tf.nn.softmax(logits=sf_input_r, axis=2)

        q_input = tf.reduce_sum(rki * h_hat, axis=2)
        reset_value = tf.concat([inputs, q_input], axis=1)
        qk = self.detect_layer(reset_value)
        qk = tf.reshape(qk, [batch_dim, self.units, 1])  # in order to broadcast

        ln_tau_s = self.update_layer(fused_input)
        ln_tau_s = tf.reshape(ln_tau_s, shape=[batch_dim, self.units, self.M])
        sf_input_s = -tf.square(ln_tau_s - self.ln_tau_table)
        ski = tf.nn.softmax(logits=sf_input_s, axis=2)

        # Now the elapsed time enters the state update
        base_term = (1 - ski) * h_hat + ski * qk
        exp_term = tf.exp(-elapsed / self.tau_table)
        exp_term = tf.reshape(exp_term, [batch_dim, 1, self.M])
        h_hat_next = base_term * exp_term

        # Compute new state
        h_next = tf.reduce_sum(h_hat_next, axis=2)
        h_hat_next_flat = tf.reshape(h_hat_next, shape=[batch_dim, self.units * self.M])
        return h_next, [h_hat_next_flat]


class VanillaRNN(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units

        super(VanillaRNN, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_dim = input_shape[0][-1]

        self._layer = tf.keras.layers.Dense(self.units, activation="tanh")
        self._out_layer = tf.keras.layers.Dense(self.units, activation=None)
        self._tau = self.add_weight(
            "tau",
            shape=(self.units),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(0.1),
        )
        self.built = True

    def call(self, inputs, states):
        elapsed = 1.0
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            elapsed = inputs[1]
            inputs = inputs[0]

        fused_input = tf.concat([inputs, states[0]], axis=-1)
        new_states = self._out_layer(self._layer(fused_input)) - elapsed * self._tau

        return new_states, [new_states]


class BidirectionalRNN(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = (units, units, units)

        self.ctrnn = CTRNNCell(self.units, num_unfolds=4, method="euler")
        self.lstm = LSTMCell(units=self.units)

        super(BidirectionalRNN, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_dim = input_shape[0][-1]
        self._out_layer = tf.keras.layers.Dense(self.units, activation=None)
        fused_dim = ((input_dim + self.units,), (1,))
        self.lstm.build(fused_dim)
        self.ctrnn.build(fused_dim)
        self.built = True

    def call(self, inputs, states):
        elapsed = 1.0
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            elapsed = inputs[1]
            inputs = inputs[0]

        lstm_state = [states[0], states[1]]
        lstm_input = [tf.concat([inputs, states[2]], axis=-1), elapsed]
        ctrnn_state = [states[2]]
        ctrnn_input = [tf.concat([inputs, states[1]], axis=-1), elapsed]

        lstm_out, new_lstm_states = self.lstm.call(lstm_input, lstm_state)
        ctrnn_out, new_ctrnn_state = self.ctrnn.call(ctrnn_input, ctrnn_state)

        fused_output = lstm_out + ctrnn_out
        return (
            fused_output,
            [new_lstm_states[0], new_lstm_states[1], new_ctrnn_state[0]],
        )


class GRUD(tf.keras.layers.Layer):
    # Implemented according to
    # https://www.nature.com/articles/s41598-018-24271-9.pdf
    # without the masking

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(GRUD, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_dim = input_shape[0][-1]

        self._reset_gate = tf.keras.layers.Dense(
            self.units, activation="sigmoid", kernel_initializer="glorot_uniform"
        )
        self._detect_signal = tf.keras.layers.Dense(
            self.units, activation="tanh", kernel_initializer="glorot_uniform"
        )
        self._update_gate = tf.keras.layers.Dense(
            self.units, activation="sigmoid", kernel_initializer="glorot_uniform"
        )
        self._d_gate = tf.keras.layers.Dense(
            self.units, activation="relu", kernel_initializer="glorot_uniform"
        )

        self.built = True

    def call(self, inputs, states):
        elapsed = 1.0
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            elapsed = inputs[1]
            inputs = inputs[0]

        dt = self._d_gate(elapsed)
        gamma = tf.exp(-dt)
        h_hat = states[0] * gamma

        fused_input = tf.concat([inputs, h_hat], axis=-1)
        rt = self._reset_gate(fused_input)
        zt = self._update_gate(fused_input)

        reset_value = tf.concat([inputs, rt * h_hat], axis=-1)
        h_tilde = self._detect_signal(reset_value)

        # Compute new state
        ht = zt * h_hat + (1.0 - zt) * h_tilde

        return ht, [ht]


class PhasedLSTM(tf.keras.layers.Layer):
    # Implemented according to
    # https://papers.nips.cc/paper/6310-phased-lstm-accelerating-recurrent-network-training-for-long-or-event-based-sequences.pdf

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = (units, units)
        self.initializer = "glorot_uniform"
        self.recurrent_initializer = "orthogonal"
        super(PhasedLSTM, self).__init__(**kwargs)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return (
            tf.zeros([batch_size, self.units], dtype=tf.float32),
            tf.zeros([batch_size, self.units], dtype=tf.float32),
        )

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_dim = input_shape[0][-1]

        self.input_kernel = self.add_weight(
            shape=(input_dim, 4 * self.units),
            initializer=self.initializer,
            name="input_kernel",
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, 4 * self.units),
            initializer=self.recurrent_initializer,
            name="recurrent_kernel",
        )
        self.bias = self.add_weight(
            shape=(4 * self.units,),  # <-- FIX
            initializer=tf.keras.initializers.Zeros(),
            name="bias",
        )
        self.tau = self.add_weight(
            shape=(1,), initializer=tf.keras.initializers.Zeros(), name="tau"
        )
        self.ron = self.add_weight(
            shape=(1,), initializer=tf.keras.initializers.Zeros(), name="ron"
        )
        self.s = self.add_weight(
            shape=(1,), initializer=tf.keras.initializers.Zeros(), name="s"
        )

        self.built = True

    def call(self, inputs, states):
        cell_state, hidden_state = states
        elapsed = 1.0
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            elapsed = inputs[1]
            inputs = inputs[0]

        # Leaky constant taken fromt he paper
        alpha = 0.001
        # Make sure these values are positive
        tau = tf.nn.softplus(self.tau)
        s = tf.nn.softplus(self.s)
        ron = tf.nn.softplus(self.ron)

        phit = tf.math.mod(elapsed - s, tau) / tau
        kt = tf.where(
            tf.less(phit, 0.5 * ron),
            2 * phit * ron,
            tf.where(tf.less(phit, ron), 2.0 - 2 * phit / ron, alpha * phit),
        )

        z = (
            tf.matmul(inputs, self.input_kernel)
            + tf.matmul(hidden_state, self.recurrent_kernel)
            + self.bias
        )
        i, ig, fg, og = tf.split(z, 4, axis=-1)

        input_activation = tf.nn.tanh(i)
        input_gate = tf.nn.sigmoid(ig)
        forget_gate = tf.nn.sigmoid(fg + 1.0)
        output_gate = tf.nn.sigmoid(og)

        c_tilde = cell_state * forget_gate + input_activation * input_gate
        c = kt * c_tilde + (1.0 - kt) * cell_state

        h_tilde = tf.nn.tanh(c_tilde) * output_gate
        h = kt * h_tilde + (1.0 - kt) * hidden_state

        return h, [c, h]


class GRUODE(tf.keras.layers.Layer):
    # Implemented according to
    # https://arxiv.org/pdf/1905.12374.pdf
    # without the Bayesian stuff

    def __init__(self, units, num_unfolds=4, **kwargs):
        self.units = units
        self.num_unfolds = num_unfolds
        self.state_size = units
        super(GRUODE, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_dim = input_shape[0][-1]
        self._reset_gate = tf.keras.layers.Dense(
            self.units,
            activation="sigmoid",
            bias_initializer=tf.constant_initializer(1),
        )
        self._detect_signal = tf.keras.layers.Dense(self.units, activation="tanh")
        self._update_gate = tf.keras.layers.Dense(self.units, activation="sigmoid")

        self.built = True

    def _dh_dt(self, inputs, states):
        fused_input = tf.concat([inputs, states], axis=-1)
        rt = self._reset_gate(fused_input)
        zt = self._update_gate(fused_input)

        reset_value = tf.concat([inputs, rt * states], axis=-1)
        gt = self._detect_signal(reset_value)

        # Compute new state
        dhdt = (1.0 - zt) * (gt - states)
        return dhdt

    def euler(self, inputs, hidden_state, delta_t):
        dy = self._dh_dt(inputs, hidden_state)
        return hidden_state + delta_t * dy

    def call(self, inputs, states):
        elapsed = 1.0
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            elapsed = inputs[1]
            inputs = inputs[0]

        delta_t = elapsed / self.num_unfolds
        hidden_state = states[0]
        for i in range(self.num_unfolds):
            hidden_state = self.euler(inputs, hidden_state, delta_t)
        return hidden_state, [hidden_state]

        return ht, [ht]


class NODE(tf.keras.layers.Layer):
    # Neural ODE (https://arxiv.org/abs/1806.07366) as an RNN cell.
    # The hidden state evolves according to dh/dt = f(h, x), where f is a small
    # MLP, integrated over the elapsed time with a fixed-step Euler scheme.
    def __init__(self, units, hidden_dim=None, num_unfolds=4, **kwargs):
        self.units = units
        self.hidden_dim = hidden_dim or 4 * units
        self.num_unfolds = num_unfolds
        self.state_size = units
        super(NODE, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_dim = input_shape[0][-1]

        # Dynamics f([x; h]) -> dh/dt
        self._dynamics = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.hidden_dim, activation="tanh", name="dynamics_hidden"
                ),
                tf.keras.layers.Dense(self.units, activation=None, name="dynamics_out"),
            ]
        )
        self.built = True

    def _dh_dt(self, inputs, hidden_state):
        fused_input = tf.concat([inputs, hidden_state], axis=-1)
        return self._dynamics(fused_input)

    def euler(self, inputs, hidden_state, delta_t):
        dy = self._dh_dt(inputs, hidden_state)
        return hidden_state + delta_t * dy

    def call(self, inputs, states):
        hidden_state = states[0]
        elapsed = 1.0
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            elapsed = inputs[1]
            inputs = inputs[0]

        delta_t = elapsed / self.num_unfolds
        for i in range(self.num_unfolds):
            hidden_state = self.euler(inputs, hidden_state, delta_t)
        return hidden_state, [hidden_state]


class HawkLSTMCell(tf.keras.layers.Layer):
    # https://papers.nips.cc/paper/7252-the-neural-hawkes-process-a-neurally-self-modulating-multivariate-point-process.pdf
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = (units, units, units)  # state is a tripple
        self.initializer = "glorot_uniform"
        self.recurrent_initializer = "orthogonal"
        super(HawkLSTMCell, self).__init__(**kwargs)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return (
            tf.zeros([batch_size, self.units], dtype=tf.float32),
            tf.zeros([batch_size, self.units], dtype=tf.float32),
            tf.zeros([batch_size, self.units], dtype=tf.float32),
        )

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_dim = input_shape[0][-1]
        self.input_kernel = self.add_weight(
            shape=(input_dim, 7 * self.units),
            initializer=self.initializer,
            name="input_kernel",
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, 7 * self.units),
            initializer=self.recurrent_initializer,
            name="recurrent_kernel",
        )
        self.bias = self.add_weight(
            shape=(7 * self.units),
            initializer=tf.keras.initializers.Zeros(),
            name="bias",
        )

        self.built = True

    def call(self, inputs, states):
        c, c_bar, h = states
        k = inputs[0]  # Is the input
        delta_t = inputs[1]  # is the elapsed time

        z = (
            tf.matmul(k, self.input_kernel)
            + tf.matmul(h, self.recurrent_kernel)
            + self.bias
        )
        i, ig, fg, og, ig_bar, fg_bar, d = tf.split(z, 7, axis=-1)

        input_activation = tf.nn.tanh(i)
        input_gate = tf.nn.sigmoid(ig)
        input_gate_bar = tf.nn.sigmoid(ig_bar)
        forget_gate = tf.nn.sigmoid(fg)
        forget_gate_bar = tf.nn.sigmoid(fg_bar)
        output_gate = tf.nn.sigmoid(og)
        delta_gate = tf.nn.softplus(d)

        new_c = c * forget_gate + input_activation * input_gate
        new_c_bar = c_bar * forget_gate_bar + input_activation * input_gate_bar

        c_t = new_c_bar + (new_c - new_c_bar) * tf.exp(-delta_gate * delta_t)
        output_state = tf.nn.tanh(c_t) * output_gate

        return output_state, [new_c, new_c_bar, output_state]



class SPDATransformer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()

        self.embed_dim = embed_dim

        self.input_proj = tf.keras.layers.Dense(embed_dim)

        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads

        self.query_dense = tf.keras.layers.Dense(embed_dim)
        self.key_dense = tf.keras.layers.Dense(embed_dim)
        self.value_dense = tf.keras.layers.Dense(embed_dim)
        self.combine_heads = tf.keras.layers.Dense(embed_dim)

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]

        x = self.input_proj(inputs)

        # ---- Attention ----
        q = self.query_dense(x)
        k = self.key_dense(x)
        v = self.value_dense(x)

        def split_heads(t):
            t = tf.reshape(t, (batch_size, -1, self.num_heads, self.projection_dim))
            return tf.transpose(t, perm=[0, 2, 1, 3])

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        score = tf.matmul(q, k, transpose_b=True)
        scaled = score / tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32))
        weights = tf.nn.softmax(scaled, axis=-1)

        attention = tf.matmul(weights, v)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        attn_output = self.combine_heads(concat_attention)

        attn_output = self.dropout1(attn_output, training=training)

        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        return self.layernorm2(out1 + ffn_output)


#implemented based on the ODE-Transformer paper: https://arxiv.org/abs/2310.05573
class ODEformer(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, num_heads=4, ff_dim=None, n_steps=3, step_size=0.25, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim or 4 * hidden_dim
        self.n_steps = n_steps
        self.step_size = step_size
        self.dropout_rate = dropout

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if input_dim != self.hidden_dim:
            # add projection to match dimensions
            self.input_proj = tf.keras.layers.Dense(self.hidden_dim)
        else:
            self.input_proj = tf.identity

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.hidden_dim // self.num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.ff_dim, activation='relu'),
            tf.keras.layers.Dense(self.hidden_dim),
        ])
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=False):
        y = self.input_proj(inputs) if callable(self.input_proj) else self.input_proj(inputs)
        for _ in range(self.n_steps):
            attn_out = self.mha(y, y, y, training=training)
            y1 = self.norm1(y + self.dropout(attn_out, training=training))
            ffn_out = self.ffn(y1, training=training)
            dy = self.norm2(y1 + self.dropout(ffn_out, training=training))
            y = y + self.step_size * dy
        return y


#implemented based on https://arxiv.org/abs/2101.10318
class mTAN(tf.keras.layers.Layer):
    def __init__(self, hidden_dim=128, num_heads=8, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout_rate = dropout_rate

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.q_dense = tf.keras.layers.Dense(hidden_dim, use_bias=False)
        self.k_dense = tf.keras.layers.Dense(hidden_dim, use_bias=False)
        self.v_dense = tf.keras.layers.Dense(hidden_dim, use_bias=False)

        # Time encoding g(Δt)
        self.time_dense = tf.keras.layers.Dense(hidden_dim, activation='tanh', use_bias=True)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.out_dense = tf.keras.layers.Dense(hidden_dim)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Will be created in build() if needed
        self.input_proj = None

    def build(self, input_shape):
        # input_shape = (B, T, D+1)
        input_dim = input_shape[-1] - 1  # exclude time channel

        # Automatic projection for residual connection
        if input_dim != self.hidden_dim:
            self.input_proj = tf.keras.layers.Dense(self.hidden_dim, use_bias=False)

        super().build(input_shape)

    def call(self, inputs, mask=None, training=None):
        """
        inputs: (B, T, D+1)
        mask: (B, T), boolean
        """
        x = inputs[:, :, :-1]      # (B, T, D)
        t = inputs[:, :, -1:]      # (B, T, 1)

        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        # === Relative time differences: δt_{i,j} ===
        # Relative time differences
        t_i = tf.expand_dims(t, axis=2)   # (B, T, 1, 1)
        t_j = tf.expand_dims(t, axis=1)   # (B, 1, T, 1)
        delta_t = (t_i - t_j)[..., 0]     # (B, T, T)

        # Add last dimension for Dense
        delta_t_exp = tf.expand_dims(delta_t, axis=-1)  # (B, T, T, 1)
        time_enc = self.time_dense(delta_t_exp)         # (B, T, T, H)

        # Time encoding g(Δt)

        # === Standard Q, K, V projections ===
        Q = self.q_dense(x)  # (B, T, H)
        K = self.k_dense(x)
        V = self.v_dense(x)

        # Reshape for multi-head attention
        Q = tf.reshape(Q, [B, T, self.num_heads, self.head_dim])
        K = tf.reshape(K, [B, T, self.num_heads, self.head_dim])
        V = tf.reshape(V, [B, T, self.num_heads, self.head_dim])

        Q = tf.transpose(Q, [0, 2, 1, 3])  # (B, h, T, d)
        K = tf.transpose(K, [0, 2, 1, 3])
        V = tf.transpose(V, [0, 2, 1, 3])

        # Expand time encoding for heads
        time_enc = tf.reshape(time_enc, [B, T, T, self.num_heads, self.head_dim])
        time_enc = tf.transpose(time_enc, [0, 3, 1, 2, 4])  # (B, h, T, T, d)

        # K + g(Δt)
        K_time = K[:, :, tf.newaxis, :, :] + time_enc  # (B, h, T, T, d)

        # Compute attention scores via elementwise multiply & sum
        Q_exp = tf.expand_dims(Q, 3)  # (B, h, T, 1, d)
        attn_scores = tf.reduce_sum(Q_exp * K_time, axis=-1)  # (B, h, T, T)
        attn_scores *= self.scale

        # === Masking ===
        if mask is not None:
            # Save original for pooling later
            orig_mask = mask

            mask = tf.cast(mask, tf.float32)
            mask = tf.reshape(mask, (B, 1, 1, T))  # broadcast to (B, h, T, T)
            attn_scores += (1.0 - mask) * -1e9
        else:
            orig_mask = None

        # Softmax
        attn_weights = tf.nn.softmax(attn_scores, axis=-1)
        attn_weights = self.dropout(attn_weights, training=training)

        # Weighted sum
        attn_out = tf.matmul(attn_weights, V)  # (B, h, T, d)

        # Restore shape
        attn_out = tf.transpose(attn_out, [0, 2, 1, 3])  # (B, T, h, d)
        attn_out = tf.reshape(attn_out, [B, T, self.hidden_dim])

        # Output projection
        out = self.out_dense(attn_out)

        # === Residual connection with automatic projection ===
        residual = x
        if self.input_proj is not None:
            residual = self.input_proj(residual)

        out = self.norm(residual + out)

        # === Pooled representation ===
        if orig_mask is not None:
            m = tf.cast(orig_mask, out.dtype)
            m = tf.expand_dims(m, -1)
            summed = tf.reduce_sum(out * m, axis=1)
            context = summed / (tf.reduce_sum(m, axis=1) + 1e-8)
        else:
            context = tf.reduce_mean(out, axis=1)

        return context

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
        })
        return config

#implemented based on https://arxiv.org/abs/2402.10635
class ContiFormer(tf.keras.layers.Layer):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        ff_dim: Optional[int] = None,
        n_steps: int = 4,
        total_time: float = 1.0,
        learn_time: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim or max(4 * dim, dim)
        self.n_steps = int(n_steps)
        self.total_time = float(total_time)
        self.learn_time = bool(learn_time)
        self.dropout_rate = float(dropout)

        # Layers used by the dynamics function f(y, t)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.dim // self.num_heads)
        self.norm_attn = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_ff = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.ff_dim, activation="relu"),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.dim),
        ])
        self.final_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        # Time projection: project scalar time -> vector (to condition attention/MLP)
        self.time_proj = tf.keras.layers.Dense(self.dim, activation=None)

    def build(self, input_shape):
        # create trainable raw deltas if requested
        if self.learn_time:
            # initialize raw_deltas so that after softplus and normalization they are approx uniform
            init_val = tf.constant(1.0 / float(self.n_steps), dtype=tf.float32)
            # raw_deltas shape: (n_steps,)
            self._raw_deltas = self.add_weight(
                name="raw_time_deltas",
                shape=(self.n_steps,),
                initializer=tf.keras.initializers.Constant(init_val.numpy() if hasattr(init_val, "numpy") else 1.0 / self.n_steps),
                trainable=True,
                dtype=tf.float32,
            )
        else:
            self._raw_deltas = None
        super().build(input_shape)

    def get_time_deltas(self):
        """Return positive time deltas that sum to total_time.

        If learn_time is enabled, transform raw parameters with softplus and normalize.
        Otherwise return uniform deltas.
        """
        if self.learn_time:
            # ensure strictly positive deltas
            pos = tf.nn.softplus(self._raw_deltas)
            # normalize to total_time
            normalized = pos / tf.reduce_sum(pos + 1e-12) * self.total_time
            return normalized
        else:
            return tf.fill((self.n_steps,), tf.constant(self.total_time / float(self.n_steps), dtype=tf.float32))

    def compute_time_embeddings(self):
        """Compute a (n_steps, dim) time embedding matrix.

        We compute cumulative times t_k = cumsum(deltas) and project t_k through a small
        dense layer to get per-step time conditioning vectors.
        """
        deltas = self.get_time_deltas()  # (n_steps,)
        times = tf.cumsum(deltas)  # (n_steps,)
        # project times to dim -- produce (n_steps, dim)
        times_expanded = tf.expand_dims(times, axis=-1)  # (n_steps, 1)
        time_embs = self.time_proj(times_expanded)  # (n_steps, dim)
        return time_embs, deltas

    def dynamics(self, y, t_emb, attention_mask=None, training=None):
        """Dynamics function f(y, t): uses attention + small MLP to produce derivative-like output.

        y: (batch, seq_len, dim)
        t_emb: (batch, seq_len, dim) or broadcastable
        """
        # condition by adding time embedding
        z = y + t_emb
        # self-attention
        attn_out = self.mha(query=z, value=z, key=z, attention_mask=attention_mask, training=training)
        attn_out = self.dropout(attn_out, training=training)
        # residual + norm
        attn_res = self.norm_attn(y + attn_out)
        # feed-forward
        ff_out = self.ffn(attn_res, training=training)
        ff_res = self.norm_ff(attn_res + ff_out)
        # derivative candidate
        return ff_res

    def call(self, inputs, attention_mask=None, training=None):
        """Run Euler integration internally and return final states.

        inputs: (batch, seq_len, dim)
        attention_mask: optional mask compatible with tf.keras.layers.MultiHeadAttention
        """
        x = tf.convert_to_tensor(inputs)
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # compute time embeddings and deltas
        time_embs, deltas = self.compute_time_embeddings()  # time_embs: (n_steps, dim), deltas: (n_steps,)

        # expand time embeddings to (n_steps, batch, seq_len, dim) by broadcasting
        # We'll add per-step t_emb to y at each integration step.
        # Expand to (n_steps, 1, 1, dim) so it can broadcast with (batch, seq_len, dim)
        t_embs = tf.reshape(time_embs, (self.n_steps, 1, 1, self.dim))

        # initial state y0 = inputs
        y = tf.identity(x)

        # Euler integration loop
        for k in range(self.n_steps):
            # dt_k is scalar
            dt_k = deltas[k]
            # step-specific time embedding broadcasted to (batch, seq_len, dim)
            t_emb_k = tf.broadcast_to(t_embs[k], (batch_size, seq_len, self.dim))
            # compute derivative
            f_val = self.dynamics(y, t_emb_k, attention_mask=attention_mask, training=training)
            # Euler update: y <- y + dt * f
            # ensure dt broadcastable and cast consistent
            dt_k_cast = tf.cast(dt_k, y.dtype)
            y = y + dt_k_cast * f_val

        # optional final normalization
        out = self.final_norm(y)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "n_steps": self.n_steps,
            "total_time": self.total_time,
            "learn_time": self.learn_time,
            "dropout": self.dropout_rate,
        })
        return config





# implemented based on https://papers.nips.cc/paper_files/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
class DeepState(tf.keras.layers.Layer):
    def __init__(self, dim=64, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def build(self, input_shape):
        self.dim = input_shape[-1]

        self.in_proj = tf.keras.layers.Dense(self.dim)

        self.log_A = self.add_weight(
            shape=(self.dim,),
            initializer=tf.keras.initializers.Constant(-1.0),
            trainable=True,
            name="log_A",
        )
        self.B = self.add_weight(
            shape=(self.dim,),
            initializer="random_normal",
            trainable=True,
            name="B",
        )
        self.C = self.add_weight(
            shape=(self.dim, self.dim),
            initializer="random_normal",
            trainable=True,
            name="C",
        )
        self.D = self.add_weight(
            shape=(self.dim,),
            initializer="zeros",
            trainable=True,
            name="D",
        )

        super().build(input_shape)

    def call(self, x):
        # x: (B, T, dim)
        B = tf.shape(x)[0]

        u = self.in_proj(x)  # (B, T, dim)
        A = -tf.exp(self.log_A)

        def step(h, u_t):
            return h * A + u_t * self.B

        h0 = tf.zeros((B, self.dim), dtype=x.dtype)

        h_seq = tf.scan(
            step,
            tf.transpose(u, (1, 0, 2)),
            initializer=h0,
        )

        h_seq = tf.transpose(h_seq, (1, 0, 2))  # (B, T, dim)

        y = tf.einsum("bts,sd->btd", h_seq, self.C)  # (B, T, dim)

        return y + tf.einsum("btd,d->btd", x, self.D)


# implemented based on https://github.com/srush/annotated-s4/blob/main/s4/s4.py for testing purposes
def make_DPLR_HiPPO(N):
    n = np.arange(N)
    alpha = 0.5
    P = np.sqrt(n + alpha)
    A = P[:, None] * P[None, :]
    A = np.tril(A) - np.diag(n)
    A = -A
    B = np.sqrt(2 * n + 1.0)

    # Cast to complex64
    A = A.astype(np.complex64)
    P = P.astype(np.complex64)
    B = B.astype(np.complex64)

    S = A + np.outer(P, P)
    S = -1j * S

    Lambda_imag, V = np.linalg.eigh(S)
    Lambda_real = np.zeros(N, dtype=np.float32)
    P = V.conj().T @ P
    B = V.conj().T @ B
    Lambda = Lambda_real + 1j * Lambda_imag
    return Lambda, P.real, B.real


class S4(tf.keras.layers.Layer):
    def __init__(self, d_model, state_size=64, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.state_size = state_size

        # Assuming make_DPLR_HiPPO returns (Lambda, P, B) for HiPPO initialization
        Lambda, P, B = make_DPLR_HiPPO(state_size)

        Lambda_re_init = np.broadcast_to(Lambda.real, (d_model, state_size))
        Lambda_im_init = np.broadcast_to(Lambda.imag, (d_model, state_size))
        P_init = np.broadcast_to(P, (d_model, state_size))
        B_init = np.broadcast_to(B, (d_model, state_size))

        # strictly negative Lambda for stability
        self.Lambda_re = self.add_weight(
            name="Lambda_re",
            shape=(d_model, state_size),
            initializer=tf.constant_initializer(Lambda_re_init),
            trainable=True
        )
        self.Lambda_im = self.add_weight(
            name="Lambda_im",
            shape=(d_model, state_size),
            initializer=tf.constant_initializer(Lambda_im_init),
            trainable=True
        )

        self.P = self.add_weight(
            name="P",
            shape=(d_model, state_size),
            initializer=tf.constant_initializer(P_init),
            trainable=True
        )
        self.B = self.add_weight(
            name="B",
            shape=(d_model, state_size),
            initializer=tf.constant_initializer(B_init),
            trainable=True
        )

        self.D = self.add_weight(
            name="D",
            shape=(d_model,),
            initializer=tf.zeros_initializer(),  # zeros for stability
            trainable=True
        )

        self.log_step = self.add_weight(
            name="log_step",
            shape=(d_model,),
            initializer=tf.random_uniform_initializer(np.log(0.001), np.log(0.1)),
            trainable=True
        )

        std = 0.05  # reduced scale
        self.C_re = self.add_weight(
            name="C_re",
            shape=(d_model, state_size),
            initializer=tf.random_normal_initializer(stddev=std),
            trainable=True
        )
        self.C_im = self.add_weight(
            name="C_im",
            shape=(d_model, state_size),
            initializer=tf.random_normal_initializer(stddev=std),
            trainable=True
        )

    def build(self, input_shape):
        # Map the raw input features to d_model so the layer actually uses its
        # full d_model-shaped SSM weights and emits a (B, L, d_model) output.
        # Previously call() used u.shape[-1] (e.g. 7 for ETTm1) as the model
        # dim, which silently ignored d_model and only touched a prefix slice
        # of every weight.
        input_dim = input_shape[-1]
        if input_dim != self.d_model:
            self.input_proj = tf.keras.layers.Dense(self.d_model, use_bias=True, name="input_proj")
        else:
            self.input_proj = tf.identity
        super().build(input_shape)

    def call(self, u):
        batch_size, L = tf.unstack(tf.shape(u))[:2]

        # Project features -> d_model so weights are used in full.
        u = self.input_proj(u)  # (B, L, d_model)

        k = self.kernel(L)
        u = tf.transpose(u, [0, 2, 1])  # (B, d_model, L)

        pad_len = L
        u_pad = tf.pad(u, [[0, 0], [0, 0], [0, pad_len]])  # (B, d_model, 2L)
        k_pad = tf.pad(k, [[0, 0], [0, pad_len]])          # (d_model, 2L)

        u_fft = tf.signal.rfft(u_pad)
        k_fft = tf.signal.rfft(k_pad)
        out_fft = u_fft * k_fft[None, :, :]
        out = tf.signal.irfft(out_fft)
        out = out[:, :, :L]  # (B, d_model, L)

        # Skip connection
        out = out + u * self.D[:, None]

        return tf.transpose(out, [0, 2, 1])  # (B, L, d_model)

    def kernel(self, L):
        # strictly negative real part
        Lambda = tf.complex(-tf.nn.softplus(self.Lambda_re), self.Lambda_im)
        P = tf.complex(self.P, tf.zeros_like(self.P))
        B = tf.complex(self.B, tf.zeros_like(self.B))
        C = tf.complex(self.C_re, self.C_im)

        # step with clipping
        step = tf.exp(tf.clip_by_value(self.log_step, np.log(0.001), np.log(0.1)))
        step = tf.cast(step, tf.complex64)

        Lf = tf.cast(L, tf.float32)
        freq = tf.range(Lf) / Lf
        freq = tf.cast(freq, tf.complex64)
        omega = tf.exp(tf.complex(tf.constant(0.0, tf.float32), tf.constant(-2.0 * np.pi, tf.float32)) * freq)

        g = (2.0 / step[:, None]) * ((1.0 - omega) / (1.0 + omega))
        toeplitz = tf.cast(2.0 / (1.0 + omega), tf.complex64)

        def cauchy(v, g, Lambda):
            return tf.reduce_sum(
                v[:, :, None] / (g[:, None, :] - Lambda[:, :, None] + 1e-6),  # epsilon added
                axis=1
            )

        k00 = cauchy(C * B, g, Lambda)
        k01 = cauchy(C * P, g, Lambda)
        k10 = cauchy(P * B, g, Lambda)
        k11 = cauchy(P * P, g, Lambda)

        at_roots = toeplitz[None, :] * (
            k00 - k01 * (1.0 / (1.0 + k11 + 1e-6)) * k10
        )

        k = tf.signal.ifft(at_roots)
        k = tf.math.real(k)

        return k[:, :L]


# implemented based on https://arxiv.org/abs/2312.00752 (Mamba: Linear-Time Sequence
# Modeling with Selective State Spaces). A plain-TF (scan-based) selective-SSM block,
# structured the same way as the other layers in this file: it maps (B, L, D_in) -> (B, L, D).
class Mamba(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.dt_min = dt_min
        self.dt_max = dt_max

    def build(self, input_shape):
        input_dim = int(input_shape[-1])

        # Optional projection so the layer behaves like the others in this file
        # when the input feature dim differs from d_model (e.g. ETTm1 has 7 feats).
        if input_dim != self.d_model:
            self.input_proj = tf.keras.layers.Dense(
                self.d_model, use_bias=True, name="input_proj"
            )
        else:
            self.input_proj = tf.identity

        # Expand + projection & gate streams: x -> [proj, gate]
        self.in_proj = tf.keras.layers.Dense(self.d_inner * 2, use_bias=False, name="in_proj")

        # Depthwise causal conv for local convolution
        self.conv1d = tf.keras.layers.Conv1D(
            self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding="causal",
            use_bias=True,
            name="conv1d",
        )

        # Selective SSM parameters. dt, B, C are all input-dependent ("selective").
        self.x_proj = tf.keras.layers.Dense(
            self.dt_rank + 2 * self.d_state, use_bias=False, name="x_proj"
        )
        self.dt_proj = tf.keras.layers.Dense(self.d_inner, use_bias=True, name="dt_proj")

        # A matrix (diagonal, per (d_inner, d_state)). As in the reference
        # Mamba, A is parametrized as A = -exp(A_log) (strictly negative), so
        # the discretized decay factor deltaA = exp(dt * A) stays in (0, 1]
        # and the recurrence is stable for any sequence length.
        A = np.arange(1, self.d_state + 1, dtype=np.float32)
        A = np.tile(np.log(A), (self.d_inner, 1))  # (d_inner, d_state)
        self.A_log = self.add_weight(
            name="A_log",
            shape=(self.d_inner, self.d_state),
            initializer=tf.constant_initializer(A),
            trainable=True,
        )
        # D skip connection (per channel)
        self.D = self.add_weight(
            name="D",
            shape=(self.d_inner,),
            initializer=tf.ones_initializer(),
            trainable=True,
        )
        # dt bias is added post-projection (before softplus)
        self.dt_bias = self.add_weight(
            name="dt_bias",
            shape=(self.d_inner,),
            initializer=tf.constant_initializer(0.0),
            trainable=True,
        )

        self.out_proj = tf.keras.layers.Dense(self.d_model, use_bias=False, name="out_proj")
        super().build(input_shape)

    def selective_scan(self, u, delta, A, B, C):
        """Scalar selective scan on the time axis.

        u:     (B, L, d_inner)
        delta: (B, L, d_inner)  (already positive via softplus)
        A:     (d_inner, d_state)
        B, C:  (B, L, d_state)  (input dependent)
        Returns y: (B, L, d_inner)  (includes D * u skip added by caller)
        """
        d_inner = self.d_inner
        d_state = self.d_state
        batch = tf.shape(u)[0]
        L = tf.shape(u)[1]

        # Continuous discretization (diagonal A):
        #   A_bar = exp(dt * A)      (B, L, d_inner, d_state)
        #   B_bar = dt * B_t         (B, L, d_inner, d_state)
        deltaA = tf.exp(delta[..., None] * A)                       # (B,L,d_inner,d_state)
        deltaB = delta[..., None] * B[..., None, :]                 # (B,L,d_inner,d_state)
        Bxu = deltaB * u[..., None]                                 # (B,L,d_inner,d_state)

        # Orient for tf.scan over the time axis: (L, B, ...)
        dA_t = tf.transpose(deltaA, [1, 0, 2, 3])                   # (L,B,d_inner,d_state)
        Bx_t = tf.transpose(Bxu, [1, 0, 2, 3])                      # (L,B,d_inner,d_state)
        C_t = tf.transpose(C, [1, 0, 2])                            # (L,B,d_state)

        def step(prev, elem):
            h_prev = prev[0]
            y_prev = prev[1]  # unused; kept for uniform scan outputs
            dA, Bx, C = elem
            h = dA * h_prev + Bx                                    # (B,d_inner,d_state)
            y = tf.reduce_sum(C[:, None, :] * h, axis=-1)           # (B,d_inner)
            return (h, y)

        init = (tf.zeros((batch, d_inner, d_state)), tf.zeros((batch, d_inner)))
        _, y_seq = tf.scan(step, (dA_t, Bx_t, C_t), init)
        y_seq = tf.transpose(y_seq, [1, 0, 2])                      # (B, L, d_inner)
        return y_seq

    def call(self, inputs):
        # x: (B, L, D_in); input_proj is always callable (Dense or tf.identity)
        x = self.input_proj(inputs)
        batch = tf.shape(x)[0]
        L = tf.shape(x)[1]

        # 1) Input expansion + gating stream
        proj = self.in_proj(x)                          # (B, L, 2*d_inner)
        x, gate = tf.split(proj, 2, axis=-1)            # (B, L, d_inner) each

        # 2) Short 1D conv
        x = self.conv1d(x)                              # (B, L, d_inner) causal
        x = tf.nn.silu(x)

        # 3) Selective parameters from the (convolved) input
        dtB_C = self.x_proj(x)                          # (B, L, dt_rank + 2*d_state)
        dt_rank = self.dt_rank
        dt_proj_raw = dtB_C[..., :dt_rank]
        B = dtB_C[..., dt_rank:dt_rank + self.d_state]  # (B, L, d_state)
        C = dtB_C[..., dt_rank + self.d_state:]         # (B, L, d_state)

        # Per-timestep step size, projected to d_inner, then softplus -> positive.
        # Matches the reference: dt = softplus(dt_proj(x) + dt_bias), no clamp.
        dt = self.dt_proj(dt_proj_raw) + self.dt_bias   # (B, L, d_inner)
        dt = tf.nn.softplus(dt)

        # Strictly negative (stable) decay rates; same convention as reference.
        A = -tf.exp(self.A_log)                         # (d_inner, d_state)

        y = self.selective_scan(x, dt, A, B, C)         # (B, L, d_inner)
        y = y + self.D[None, None, :] * x               # skip connection inside SSM

        # 4) Gated output
        out = y * tf.nn.silu(gate)
        return self.out_proj(out)                       # (B, L, d_model)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "d_state": self.d_state,
            "d_conv": self.d_conv,
            "expand": self.expand,
            "dt_rank": self.dt_rank,
            "dt_min": self.dt_min,
            "dt_max": self.dt_max,
        })
        return config


#implemented based on https://arxiv.org/abs/2505.20666
class PDEAttention(tf.keras.layers.Layer):
    def __init__(self,
                 num_heads: int,
                 key_dim: int,
                 value_dim: int | None = None,
                 pde_type: str = 'diffusion',
                 nt: int = 4,
                 dt: float = 0.1,
                 alpha: float = 0.1,
                 beta: float = 0.02,   # reaction coefficient (ignored for diffusion)
                 c: float = 0.15,      # wave speed (ignored for diffusion/reaction)
                 dropout: float = 0.0,
                 use_bias: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim or key_dim
        self.pde_type = pde_type.lower()
        self.nt = nt
        self.dt = dt
        self.alpha = alpha
        self.beta = beta
        self.c = c
        self.dropout = tf.keras.layers.Dropout(dropout)

        if self.pde_type not in {'diffusion', 'reaction_diffusion', 'wave'}:
            raise ValueError("pde_type must be 'diffusion', 'reaction_diffusion' or 'wave'")

        # Linear projections (exactly as in standard MultiHeadAttention)
        self.query_dense = tf.keras.layers.Dense(
            num_heads * key_dim, use_bias=use_bias, name="query_dense")
        self.key_dense = tf.keras.layers.Dense(
            num_heads * key_dim, use_bias=use_bias, name="key_dense")
        self.value_dense = tf.keras.layers.Dense(
            num_heads * self.value_dim, use_bias=use_bias, name="value_dense")
        self.output_dense = tf.keras.layers.Dense(
            self.value_dim * num_heads, use_bias=use_bias, name="output_dense")

    def _split_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        """Split the last dimension into (num_heads, depth)."""
        depth = self.key_dim if x.shape[-1] == self.num_heads * self.key_dim else self.value_dim
        x = tf.reshape(x, (batch_size, -1, self.num_heads, depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (B, H, T, d)

    def _discrete_laplacian(self, x: tf.Tensor, axis: int = -1) -> tf.Tensor:
        left = tf.roll(x, shift=1, axis=axis)
        right = tf.roll(x, shift=-1, axis=axis)
        return left + right - 2.0 * x

    def _pde_evolve(self, A: tf.Tensor, training: bool) -> tf.Tensor:
        for _ in range(self.nt):
            if self.pde_type == 'diffusion':
                lap = self._discrete_laplacian(A, axis=-1)
                update = self.alpha * lap

            elif self.pde_type == 'reaction_diffusion':
                lap = self._discrete_laplacian(A, axis=-1)
                R = self.beta * A * (1.0 - A)
                update = self.alpha * lap + R

            elif self.pde_type == 'wave':
                if not hasattr(self, '_prev_A'):
                    self._prev_A = tf.identity(A)  # first step
                lap = self._discrete_laplacian(A, axis=-1)
                # Verlet-like update for wave: A_new = 2*A - A_prev + (c*dt)^2 * lap
                A_new = 2.0 * A - self._prev_A + (self.c * self.dt) ** 2 * lap
                self._prev_A = tf.identity(A)  # update for next step
                A = A_new
                continue  # skip the generic update below

            else:
                update = tf.zeros_like(A)

            A = A + self.dt * update
        A = tf.clip_by_value(A, 0.0, 1.0)
        return A

    def call(self,
             query: tf.Tensor,
             key: tf.Tensor | None = None,
             value: tf.Tensor | None = None,
             training: bool = False,
             return_attention_weights: bool = False,
             **kwargs) -> tf.Tensor:

        if key is None:
            key = query
        if value is None:
            value = query

        batch_size = tf.shape(query)[0]

        # Project to Q, K, V
        q = self.query_dense(query)   # (B, T, num_heads * key_dim)
        k = self.key_dense(key)
        v = self.value_dense(value)

        # Split into heads
        q = self._split_heads(q, batch_size)      # (B, H, Tq, d)
        k = self._split_heads(k, batch_size)
        v = self._split_heads(v, batch_size)

        scale = tf.math.sqrt(tf.cast(self.key_dim, tf.float32))
        attention_scores = tf.matmul(q, k, transpose_b=True) / scale   # (B, H, Tq, Tk)

        A = tf.nn.softmax(attention_scores, axis=-1)
        A = self._pde_evolve(A, training=training)
        A = self.dropout(A, training=training)
        attention_output = tf.matmul(A, v)   # (B, H, Tq, d)

        # Concatenate heads and project
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat = tf.reshape(attention_output, (batch_size, -1, self.num_heads * self.value_dim))
        output = self.output_dense(concat)

        if return_attention_weights:
            return output, A
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "pde_type": self.pde_type,
            "nt": self.nt,
            "dt": self.dt,
            "alpha": self.alpha,
            "beta": self.beta,
            "c": self.c,
            "dropout": self.dropout.rate,
        })
        return config


#implemented based on https://arxiv.org/abs/2501.18793

class OTTransformer(tf.keras.layers.Layer):
    def __init__(self,
                 num_heads: int = 8,
                 key_dim: int = 64,   # this will be d_model
                 ff_dim: int = 256,
                 num_steps: int = 8,
                 T: float = 1.0,
                 lambda_reg: float = 0.5,
                 dropout_rate: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)

        self.num_heads = num_heads
        self.d_model = key_dim   # treat this as model dimension
        self.ff_dim = ff_dim
        self.num_steps = num_steps
        self.T = T
        self.dt = T / num_steps
        self.lambda_reg = lambda_reg
        self.dropout_rate = dropout_rate

        self.input_proj = tf.keras.layers.Dense(self.d_model)

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.d_model // num_heads,  # per-head dim
            output_shape=self.d_model,
            dropout=dropout_rate,
            name="mha"
        )

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(self.d_model),
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def _compute_velocity(self, x, training):
        # Attention block
        attn = self.mha(x, x, x, training=training)
        x1 = self.layernorm1(x + attn)

        # FFN block
        ffn_out = self.dropout(self.ffn(x1), training=training)
        block_output = self.layernorm2(x1 + ffn_out)

        return block_output - x

    def call(self, x, training=False):
        current = self.input_proj(x)
        dt = tf.cast(self.dt, current.dtype)
        batch_size = tf.shape(current)[0]
        ot_integral = tf.zeros((batch_size,), dtype=current.dtype)

        for _ in range(self.num_steps):
            f = self._compute_velocity(current, training=training)

            current = current + dt * f

            if training:
                f_sq = tf.reduce_sum(tf.square(f), axis=[1, 2])
                dn = tf.cast(tf.shape(f)[1] * tf.shape(f)[2], current.dtype)
                ot_integral += (f_sq / dn) * dt

        if training:
            batch_ot = tf.reduce_mean(ot_integral)
            ot_loss = (self.lambda_reg / 2.0) * batch_ot
            self.add_loss(ot_loss)

        return current

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.d_model,
            "ff_dim": self.ff_dim,
            "num_steps": self.num_steps,
            "T": self.T,
            "lambda_reg": self.lambda_reg,
            "dropout_rate": self.dropout_rate,
        })
        return config