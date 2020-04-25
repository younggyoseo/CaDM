import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import DiagGaussianPdType, CategoricalPdType, MultiCategoricalPdType, BernoulliPdType

def make_pdtype(ac_space):
    from cadm import spaces as custom_spaces
    from gym import spaces
    if isinstance(ac_space, custom_spaces.Box):
        assert len(ac_space.shape) == 1
        return DiagGaussianPdType(ac_space.shape[0])
    elif isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        return DiagGaussianPdType(ac_space.shape[0])
    elif isinstance(ac_space, spaces.Discrete):
        return CategoricalPdType(ac_space.n)
    elif isinstance(ac_space, spaces.MultiDiscrete):
        return MultiCategoricalPdType(ac_space.nvec)
    elif isinstance(ac_space, spaces.MultiBinary):
        return BernoulliPdType(ac_space.n)
    else:
        raise NotImplementedError

class MlpCPPolicy(object):
    def __init__(self, sess, obs_dim, ac_space, nbatch, nsteps, hidden_size, cp_dim_output, reuse=False, n_layers=2): #pylint: disable=W0613

        if len(ac_space.shape) == 0:
            act_dim = ac_space.n
            discrete = True
        else:
            act_dim = ac_space.shape[0]
            discrete = False

        ob_shape = (None, obs_dim)
        con_shape = (None, cp_dim_output)

        X = tf.compat.v1.placeholder(tf.float32, ob_shape, name='Ob')
        context_X = tf.compat.v1.placeholder(tf.float32, con_shape, name='Con')
        
        policy_input = tf.concat([X, context_X], axis=1)
        value_input = tf.concat([X, context_X], axis=1)
        
        with tf.compat.v1.variable_scope("model", reuse=reuse):
            activ = tf.tanh
            h = policy_input
            for idx in range(n_layers):
                h = activ(fc(h, 'pi_fc{}'.format(idx), nh=hidden_size, init_scale=np.sqrt(2)))
            pi = fc(h, 'pi', act_dim, init_scale=0.01)
            h = value_input
            for idx in range(n_layers):
                h = activ(fc(h, 'vf_fc{}'.format(idx), nh=hidden_size, init_scale=np.sqrt(2)))
            vf = fc(h, 'vf', 1)[:,0]
            logstd = tf.compat.v1.get_variable(name="logstd", shape=[1, act_dim],
                initializer=tf.zeros_initializer())

        if discrete:
            pdparam = pi
        else:
            pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, con, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob, context_X: con})
            return a, v, self.initial_state, neglogp

        def value(ob, con, *_args, **_kwargs):
            return sess.run(vf, {X:ob, context_X: con})

        self.X = X
        self.context_X = context_X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
