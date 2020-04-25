from cadm.dynamics.core.layers import PlusCaDMEnsembleCEMMLP
from cadm.dynamics.core.layers import PureEnsembleContextPredictor
from collections import OrderedDict
import tensorflow as tf
import numpy as np
from cadm.utils.serializable import Serializable
from cadm.utils import tensor_utils
from cadm.logger import logger
import time
import joblib

class MLPEnsembleCEMDynamicsModel(Serializable):
    """
    Class for MLP continous dynamics model
    """

    _activations = {
        None: tf.identity,
        "relu": tf.nn.relu,
        "tanh": tf.tanh,
        "sigmoid": tf.sigmoid,
        "softmax": tf.nn.softmax,
        "swish": lambda x: x * tf.sigmoid(x)
    }

    def __init__(self,
                 name,
                 env,
                 hidden_sizes=(200, 200, 200, 200),
                 hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=None,
                 batch_size=128,
                 learning_rate=0.001,
                 normalize_input=True,
                 optimizer=tf.compat.v1.train.AdamOptimizer,
                 valid_split_ratio=0.2,
                 rolling_average_persitency=0.99,
                 n_forwards=30,
                 n_candidates=2500,
                 ensemble_size=5,
                 n_particles=20,
                 use_cem=False,
                 deterministic=False,
                 weight_decays=(0., 0., 0., 0., 0.),
                 weight_decay_coeff=0.0,
                 cp_hidden_sizes=(256, 128, 64),
                 context_weight_decays=(0., 0., 0., 0.),
                 context_out_dim=10,
                 context_hidden_nonlinearity=tf.nn.relu,
                 history_length=10,
                 future_length=10,
                 state_diff=False,
                 back_coeff=0.0,
                 ):

        Serializable.quick_init(self, locals())

        # Default Attributes
        self.env = env
        self.name = name
        self._dataset = None

        # Dynamics Model Attributes
        self.deterministic = deterministic

        # MPC Attributes
        self.n_forwards = n_forwards
        self.n_candidates = n_candidates
        self.use_cem = use_cem

        # Training Attributes
        self.weight_decays = weight_decays
        self.weight_decay_coeff = weight_decay_coeff
        self.normalization = None
        self.normalize_input = normalize_input
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.valid_split_ratio = valid_split_ratio
        self.rolling_average_persitency = rolling_average_persitency

        # PE-TS Attributes
        self.ensemble_size = ensemble_size
        self.n_particles = n_particles

        # CaDM Attributes
        self.cp_hidden_sizes = cp_hidden_sizes
        self.context_out_dim = context_out_dim
        self.history_length = history_length
        self.future_length = future_length
        self.context_weight_decays = context_weight_decays
        self.state_diff = state_diff
        self.back_coeff = back_coeff

        # Dimensionality of state and action space
        self.obs_space_dims = obs_space_dims = env.observation_space.shape[0]
        self.proc_obs_space_dims = proc_obs_space_dims = env.proc_observation_space_dims
        if len(env.action_space.shape) == 0:
            self.action_space_dims = action_space_dims = env.action_space.n
            self.discrete = True
        else:
            self.action_space_dims = action_space_dims = env.action_space.shape[0]
            self.discrete = False

        hidden_nonlinearity = self._activations[hidden_nonlinearity]
        output_nonlinearity = self._activations[output_nonlinearity]

        with tf.compat.v1.variable_scope(name):
            # placeholders
            self.obs_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, obs_space_dims))
            self.obs_next_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, obs_space_dims))
            self.act_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, action_space_dims))
            self.cp_obs_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, obs_space_dims*self.history_length))
            self.cp_act_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, action_space_dims*self.history_length))

            self.bs_obs_ph = tf.compat.v1.placeholder(tf.float32, shape=(ensemble_size, None, obs_space_dims))
            self.bs_obs_next_ph = tf.compat.v1.placeholder(tf.float32, shape=(ensemble_size, None, obs_space_dims))
            self.bs_act_ph = tf.compat.v1.placeholder(tf.float32, shape=(ensemble_size, None, action_space_dims))
            self.bs_delta_ph = tf.compat.v1.placeholder(tf.float32, shape=(ensemble_size, None, obs_space_dims))
            self.bs_back_delta_ph = tf.compat.v1.placeholder(tf.float32, shape=(ensemble_size, None, obs_space_dims))
            self.bs_cp_obs_ph = tf.compat.v1.placeholder(tf.float32, shape=(ensemble_size, None, obs_space_dims*self.history_length))
            self.bs_cp_act_ph = tf.compat.v1.placeholder(tf.float32, shape=(ensemble_size, None, action_space_dims*self.history_length))

            self.norm_obs_mean_ph = tf.compat.v1.placeholder(tf.float32, shape=(proc_obs_space_dims,))
            self.norm_obs_std_ph = tf.compat.v1.placeholder(tf.float32, shape=(proc_obs_space_dims,))
            self.norm_act_mean_ph = tf.compat.v1.placeholder(tf.float32, shape=(action_space_dims,))
            self.norm_act_std_ph = tf.compat.v1.placeholder(tf.float32, shape=(action_space_dims,))
            self.norm_delta_mean_ph = tf.compat.v1.placeholder(tf.float32, shape=(obs_space_dims,))
            self.norm_delta_std_ph = tf.compat.v1.placeholder(tf.float32, shape=(obs_space_dims,))
            self.norm_cp_obs_mean_ph = tf.compat.v1.placeholder(tf.float32, shape=(obs_space_dims*self.history_length,))
            self.norm_cp_obs_std_ph = tf.compat.v1.placeholder(tf.float32, shape=(obs_space_dims*self.history_length,))
            self.norm_cp_act_mean_ph = tf.compat.v1.placeholder(tf.float32, shape=(action_space_dims*self.history_length,))
            self.norm_cp_act_std_ph = tf.compat.v1.placeholder(tf.float32, shape=(action_space_dims*self.history_length,))
            self.norm_back_delta_mean_ph = tf.compat.v1.placeholder(tf.float32, shape=(obs_space_dims,))
            self.norm_back_delta_std_ph = tf.compat.v1.placeholder(tf.float32, shape=(obs_space_dims,))

            self.cem_init_mean_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, self.n_forwards, action_space_dims))
            self.cem_init_var_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, self.n_forwards, action_space_dims))

            # create MLP
            with tf.compat.v1.variable_scope('context_model'):
                cp = PureEnsembleContextPredictor(name,
                                                    output_dim=0,
                                                    input_dim=0,
                                                    context_dim=(obs_space_dims+action_space_dims)*self.history_length,
                                                    context_hidden_sizes=self.cp_hidden_sizes,
                                                    output_nonlinearity=output_nonlinearity,
                                                    ensemble_size=self.ensemble_size,
                                                    context_weight_decays=self.context_weight_decays,
                                                    bs_input_cp_obs_var=self.bs_cp_obs_ph,
                                                    bs_input_cp_act_var=self.bs_cp_act_ph,
                                                    norm_cp_obs_mean_var=self.norm_cp_obs_mean_ph,
                                                    norm_cp_obs_std_var=self.norm_cp_obs_std_ph,
                                                    norm_cp_act_mean_var=self.norm_cp_act_mean_ph,
                                                    norm_cp_act_std_var=self.norm_cp_act_std_ph,
                                                    context_out_dim=self.context_out_dim,
                                                    )
                self.bs_cp_var = cp.context_output_var

            with tf.compat.v1.variable_scope('ff_model'):
                mlp = PlusCaDMEnsembleCEMMLP(name,
                                             # Inputs
                                             input_dim=0,
                                             output_dim=obs_space_dims,
                                             hidden_sizes=hidden_sizes,
                                             hidden_nonlinearity=hidden_nonlinearity,
                                             output_nonlinearity=output_nonlinearity,
                                             input_obs_dim=obs_space_dims,
                                             input_act_dim=action_space_dims,
                                             input_obs_var=self.obs_ph,
                                             input_act_var=self.act_ph,
                                             bs_input_obs_var=self.bs_obs_ph,
                                             bs_input_act_var=self.bs_act_ph,
                                             # CaDM
                                             context_obs_var=self.cp_obs_ph,
                                             context_act_var=self.cp_act_ph,
                                             cp_forward=cp.forward,
                                             bs_input_cp_var=self.bs_cp_var,
                                             context_out_dim=self.context_out_dim,
                                             # PE-TS
                                             weight_decays=self.weight_decays,
                                             deterministic=self.deterministic,
                                             ensemble_size=self.ensemble_size,
                                             n_particles=self.n_particles,
                                             # Environments
                                             obs_preproc_fn=env.obs_preproc,
                                             obs_postproc_fn=env.obs_postproc,
                                             reward_fn=env.tf_reward_fn(),
                                             # Policy
                                             n_forwards=self.n_forwards,
                                             n_candidates=self.n_candidates,
                                             use_cem=self.use_cem,
                                             # Normalization
                                             cem_init_mean_var=self.cem_init_mean_ph,
                                             cem_init_var_var=self.cem_init_var_ph,
                                             norm_obs_mean_var=self.norm_obs_mean_ph,
                                             norm_obs_std_var=self.norm_obs_std_ph,
                                             norm_act_mean_var=self.norm_act_mean_ph,
                                             norm_act_std_var=self.norm_act_std_ph,
                                             norm_delta_mean_var=self.norm_delta_mean_ph,
                                             norm_delta_std_var=self.norm_delta_std_ph,
                                             norm_cp_obs_mean_var=self.norm_cp_obs_mean_ph,
                                             norm_cp_obs_std_var=self.norm_cp_obs_std_ph,
                                             norm_cp_act_mean_var=self.norm_cp_act_mean_ph,
                                             norm_cp_act_std_var=self.norm_cp_act_std_ph,
                                             norm_back_delta_mean_var=None,
                                             norm_back_delta_std_var=None,
                                             # Others
                                             discrete=self.discrete,
                                             build_policy_graph=True,
                                             )

            if self.back_coeff > 0.0:
                with tf.compat.v1.variable_scope('backward_model'):
                    back_mlp = PlusCaDMEnsembleCEMMLP(name,
                                                    # Inputs
                                                    input_dim=0,
                                                    output_dim=obs_space_dims,
                                                    hidden_sizes=hidden_sizes,
                                                    hidden_nonlinearity=hidden_nonlinearity,
                                                    output_nonlinearity=output_nonlinearity,
                                                    input_obs_dim=obs_space_dims,
                                                    input_act_dim=action_space_dims,
                                                    input_obs_var=self.obs_next_ph, ##
                                                    input_act_var=self.act_ph,
                                                    bs_input_obs_var=self.bs_obs_next_ph, ##
                                                    bs_input_act_var=self.bs_act_ph,
                                                    # CaDM
                                                    context_obs_var=self.cp_obs_ph,
                                                    context_act_var=self.cp_act_ph,
                                                    cp_forward=None, ##
                                                    bs_input_cp_var=self.bs_cp_var,
                                                    context_out_dim=self.context_out_dim,
                                                    # PE-TS
                                                    weight_decays=self.weight_decays,
                                                    deterministic=True, ##
                                                    ensemble_size=self.ensemble_size,
                                                    n_particles=self.n_particles,
                                                    # Environments
                                                    obs_preproc_fn=env.obs_preproc,
                                                    obs_postproc_fn=env.obs_postproc,
                                                    reward_fn=env.tf_reward_fn(),
                                                    # Policy
                                                    n_forwards=self.n_forwards,
                                                    n_candidates=self.n_candidates,
                                                    use_cem=self.use_cem,
                                                    # Normalization
                                                    cem_init_mean_var=self.cem_init_mean_ph,
                                                    cem_init_var_var=self.cem_init_var_ph,
                                                    norm_obs_mean_var=self.norm_obs_mean_ph,
                                                    norm_obs_std_var=self.norm_obs_std_ph,
                                                    norm_act_mean_var=self.norm_act_mean_ph,
                                                    norm_act_std_var=self.norm_act_std_ph,
                                                    norm_delta_mean_var=None, ##
                                                    norm_delta_std_var=None, ##
                                                    norm_cp_obs_mean_var=self.norm_cp_obs_mean_ph, ##
                                                    norm_cp_obs_std_var=self.norm_cp_obs_std_ph, ##
                                                    norm_cp_act_mean_var=self.norm_cp_act_mean_ph,
                                                    norm_cp_act_std_var=self.norm_cp_act_std_ph,
                                                    norm_back_delta_mean_var=self.norm_back_delta_std_ph, ##
                                                    norm_back_delta_std_var=self.norm_back_delta_std_ph, ##
                                                    # Others
                                                    discrete=self.discrete,
                                                    build_policy_graph=False, ##
                                                    )

            self.params = tf.compat.v1.trainable_variables()
            self.delta_pred = mlp.output_var 

            # 1. Forward Dynamics Prediction Loss
            # Outputs from Dynamics Model are normalized delta predictions
            mu, logvar = mlp.mu, mlp.logvar
            bs_normalized_delta = normalize(self.bs_delta_ph, self.norm_delta_mean_ph, self.norm_delta_std_ph)
            self.mse_loss = tf.reduce_sum(
                tf.reduce_mean(tf.reduce_mean(tf.square(mu - bs_normalized_delta), axis=-1), axis=-1))

            # 2. Backward Dynamics Prediction Loss
            if self.back_coeff > 0.0:
                back_mu = back_mlp.mu
                bs_normalized_back_delta = normalize(self.bs_back_delta_ph, self.norm_back_delta_mean_ph, self.norm_back_delta_std_ph)
                self.back_mse_loss = tf.reduce_sum(
                    tf.reduce_mean(tf.reduce_mean(tf.square(back_mu - bs_normalized_back_delta), axis=-1), axis=-1))

                self.back_l2_reg_loss = tf.reduce_sum(back_mlp.l2_regs)
            else:
                self.back_mse_loss = tf.constant(0.0)

            # 4. Weight Decay Regularization
            self.l2_reg_loss = tf.reduce_sum(mlp.l2_regs)
            self.context_l2_reg_loss = tf.reduce_sum(cp.l2_regs)

            l2_loss = self.l2_reg_loss + self.context_l2_reg_loss
            if self.back_coeff > 0.0:
                l2_loss += self.back_l2_reg_loss
            self.l2_loss = l2_loss

            if self.deterministic:
                recon_loss = self.mse_loss
                if self.back_coeff > 0.0:
                    recon_loss += self.back_coeff * self.back_mse_loss
                self.recon_loss = recon_loss
                self.loss = self.recon_loss + self.l2_loss * self.weight_decay_coeff
            else:
                invvar = tf.exp(-logvar)
                self.mu_loss = tf.reduce_sum(
                    tf.reduce_mean(tf.reduce_mean(tf.square(mu - bs_normalized_delta) * invvar, axis=-1), axis=-1))
                self.var_loss = tf.reduce_sum(
                    tf.reduce_mean(tf.reduce_mean(logvar, axis=-1), axis=-1))
                self.reg_loss = 0.01 * tf.reduce_sum(mlp.max_logvar) - 0.01 * tf.reduce_sum(mlp.min_logvar)

                recon_loss = self.mu_loss + self.var_loss
                if self.back_coeff > 0.0:
                    recon_loss += self.back_coeff * self.back_mse_loss
                self.recon_loss = recon_loss
                self.loss = self.recon_loss + self.reg_loss + self.l2_loss * self.weight_decay_coeff

            self.optimizer = optimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

            # tensor_utils
            self._get_cem_action = tensor_utils.compile_function([self.obs_ph,
                                                                  self.cp_obs_ph, self.cp_act_ph,
                                                                  self.norm_obs_mean_ph, self.norm_obs_std_ph,
                                                                  self.norm_act_mean_ph, self.norm_act_std_ph,
                                                                  self.norm_delta_mean_ph, self.norm_delta_std_ph,
                                                                  self.norm_cp_obs_mean_ph, self.norm_cp_obs_std_ph,
                                                                  self.norm_cp_act_mean_ph, self.norm_cp_act_std_ph,
                                                                  self.cem_init_mean_ph, self.cem_init_var_ph],
                                                                  mlp.optimal_action_var)

            self._get_rs_action = tensor_utils.compile_function([self.obs_ph,
                                                                 self.cp_obs_ph, self.cp_act_ph,
                                                                 self.norm_obs_mean_ph, self.norm_obs_std_ph,
                                                                 self.norm_act_mean_ph, self.norm_act_std_ph,
                                                                 self.norm_delta_mean_ph, self.norm_delta_std_ph,
                                                                 self.norm_cp_obs_mean_ph, self.norm_cp_obs_std_ph,
                                                                 self.norm_cp_act_mean_ph, self.norm_cp_act_std_ph,],
                                                                 mlp.optimal_action_var)

            self._get_context_pred = tensor_utils.compile_function([self.bs_cp_obs_ph, self.bs_cp_act_ph,
                                                                    self.norm_cp_obs_mean_ph, self.norm_cp_obs_std_ph,
                                                                    self.norm_cp_act_mean_ph, self.norm_cp_act_std_ph],
                                                                    self.bs_cp_var)  ## inference cp var

    def get_action(self, obs, cp_obs, cp_act, cem_init_mean=None, cem_init_var=None):
        (norm_obs_mean, norm_obs_std, norm_act_mean, norm_act_std, norm_delta_mean, norm_delta_std,
         norm_cp_obs_mean, norm_cp_obs_std, norm_cp_act_mean, norm_cp_act_std, _, _) = \
            self.get_normalization_stats()
        if cem_init_mean is not None:
            action = self._get_cem_action(obs,
                                          cp_obs, cp_act,
                                          norm_obs_mean, norm_obs_std,
                                          norm_act_mean, norm_act_std,
                                          norm_delta_mean, norm_delta_std,
                                          norm_cp_obs_mean, norm_cp_obs_std,
                                          norm_cp_act_mean, norm_cp_act_std,
                                          cem_init_mean, cem_init_var)
        else:
            action = self._get_rs_action(obs,
                                         cp_obs, cp_act,
                                         norm_obs_mean, norm_obs_std,
                                         norm_act_mean, norm_act_std,
                                         norm_delta_mean, norm_delta_std,
                                         norm_cp_obs_mean, norm_cp_obs_std,
                                         norm_cp_act_mean, norm_cp_act_std)
        if not self.discrete:
            action = np.minimum(np.maximum(action, -1.0), 1.0)
        return action

    def get_context_pred(self, cp_obs, cp_act):
        (_, _, _, _, _, _,
         norm_cp_obs_mean, norm_cp_obs_std, norm_cp_act_mean, norm_cp_act_std, _, _) = \
            self.get_normalization_stats()

        bs_cp_obs = np.tile(cp_obs, [self.ensemble_size, 1, 1])
        bs_cp_act = np.tile(cp_act, [self.ensemble_size, 1, 1])

        context = self._get_context_pred(bs_cp_obs, bs_cp_act,
                                         norm_cp_obs_mean, norm_cp_obs_std,
                                         norm_cp_act_mean, norm_cp_act_std)
        return context

    def fit(self, obs, act, obs_next, cp_obs, cp_act, future_bool, epochs=1000, compute_normalization=True,
            valid_split_ratio=None, rolling_average_persitency=None, verbose=False, log_tabular=False, max_logging=5000):

        assert obs.ndim == 2 and obs.shape[1] == self.obs_space_dims*self.future_length
        assert obs_next.ndim == 2 and obs_next.shape[1] == self.obs_space_dims*self.future_length
        assert act.ndim == 2 and act.shape[1] == self.action_space_dims*self.future_length
        assert cp_obs.ndim == 2 and cp_obs.shape[1] == (self.obs_space_dims*self.history_length)
        assert cp_act.ndim == 2 and cp_act.shape[1] == (self.action_space_dims*self.history_length)
        assert future_bool.ndim == 2 and future_bool.shape[1] == self.future_length

        if valid_split_ratio is None: valid_split_ratio = self.valid_split_ratio
        if rolling_average_persitency is None: rolling_average_persitency = self.rolling_average_persitency

        assert 1 > valid_split_ratio >= 0

        sess = tf.compat.v1.get_default_session()

        obs = obs.reshape(-1, self.obs_space_dims)
        obs_next = obs_next.reshape(-1, self.obs_space_dims)
        delta = self.env.targ_proc(obs, obs_next)
        back_delta = self.env.targ_proc(obs_next, obs)

        obs = obs.reshape(-1, self.future_length * self.obs_space_dims)
        obs_next = obs_next.reshape(-1, self.future_length * self.obs_space_dims)
        delta = delta.reshape(-1, self.future_length * self.obs_space_dims)
        back_delta = back_delta.reshape(-1, self.future_length * self.obs_space_dims)

        single_obs = obs[:, :self.obs_space_dims]
        single_act = act[:, :self.action_space_dims]
        single_delta = delta[:, :self.obs_space_dims]
        single_back_delta = back_delta[:, :self.obs_space_dims]

        if self._dataset is None:
            self._dataset = dict(obs=obs, act=act, delta=delta, 
                                 cp_obs=cp_obs, cp_act=cp_act, future_bool=future_bool,
                                 obs_next=obs_next, back_delta=back_delta,
                                 single_obs=single_obs, single_act=single_act,
                                 single_delta=single_delta, single_back_delta=single_back_delta)
        else:
            self._dataset['obs'] = np.concatenate([self._dataset['obs'], obs])
            self._dataset['act'] = np.concatenate([self._dataset['act'], act])
            self._dataset['delta'] = np.concatenate([self._dataset['delta'], delta])
            self._dataset['cp_obs'] = np.concatenate([self._dataset['cp_obs'], cp_obs])
            self._dataset['cp_act'] = np.concatenate([self._dataset['cp_act'], cp_act])
            self._dataset['future_bool'] = np.concatenate([self._dataset['future_bool'], future_bool])
            self._dataset['obs_next'] = np.concatenate([self._dataset['obs_next'], obs_next])
            self._dataset['back_delta'] = np.concatenate([self._dataset['back_delta'], back_delta])
            
            self._dataset['single_obs'] = np.concatenate([self._dataset['single_obs'], single_obs])
            self._dataset['single_act'] = np.concatenate([self._dataset['single_act'], single_act])
            self._dataset['single_delta'] = np.concatenate([self._dataset['single_delta'], single_delta])
            self._dataset['single_back_delta'] = np.concatenate([self._dataset['single_back_delta'], single_back_delta])

        self.compute_normalization(self._dataset['single_obs'],
                                   self._dataset['single_act'],
                                   self._dataset['single_delta'],
                                   self._dataset['cp_obs'],
                                   self._dataset['cp_act'],
                                   self._dataset['single_back_delta'])

        dataset_size = self._dataset['obs'].shape[0]
        n_valid_split = min(int(dataset_size * valid_split_ratio), max_logging)
        permutation = np.random.permutation(dataset_size)
        train_obs, valid_obs = self._dataset['obs'][permutation[n_valid_split:]], self._dataset['obs'][permutation[:n_valid_split]]
        train_act, valid_act = self._dataset['act'][permutation[n_valid_split:]], self._dataset['act'][permutation[:n_valid_split]]
        train_delta, valid_delta = self._dataset['delta'][permutation[n_valid_split:]], self._dataset['delta'][permutation[:n_valid_split]]
        train_cp_obs, valid_cp_obs = self._dataset['cp_obs'][permutation[n_valid_split:]], self._dataset['cp_obs'][permutation[:n_valid_split]]
        train_cp_act, valid_cp_act = self._dataset['cp_act'][permutation[n_valid_split:]], self._dataset['cp_act'][permutation[:n_valid_split]]
        train_obs_next, valid_obs_next = self._dataset['obs_next'][permutation[n_valid_split:]], self._dataset['obs_next'][permutation[:n_valid_split]]
        train_future_bool, valid_future_bool = self._dataset['future_bool'][permutation[n_valid_split:]], self._dataset['future_bool'][permutation[:n_valid_split]]
        train_back_delta, valid_back_delta = self._dataset['back_delta'][permutation[n_valid_split:]], self._dataset['back_delta'][permutation[:n_valid_split]]

        train_obs, train_act, train_delta, train_obs_next, train_back_delta, train_cp_obs, train_cp_act = \
            self._preprocess_inputs(train_obs, train_act, train_delta, train_cp_obs, train_cp_act, train_future_bool, train_obs_next, train_back_delta)
        if n_valid_split > 0:
            valid_obs, valid_act, valid_delta, valid_obs_next, valid_back_delta, valid_cp_obs, valid_cp_act = \
                self._preprocess_inputs(valid_obs, valid_act, valid_delta, valid_cp_obs, valid_cp_act, valid_future_bool, valid_obs_next, valid_back_delta)

        valid_loss_rolling_average = None
        epoch_times = []

        train_dataset_size = train_obs.shape[0]
        if self.ensemble_size > 1:
            bootstrap_idx = np.random.randint(0, train_dataset_size, size=(self.ensemble_size, train_dataset_size))
        else:
            bootstrap_idx = np.tile(np.arange(train_dataset_size, dtype='int32'), (self.ensemble_size, 1))

        valid_dataset_size = valid_obs.shape[0]
        valid_boostrap_idx = np.tile(np.arange(valid_dataset_size, dtype='int32'), (self.ensemble_size, 1))

        def shuffle_rows(arr):
            idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxs]

        """ ------- Looping over training epochs ------- """
        for epoch in range(epochs):

            # preparations for recording training stats
            mse_losses, back_mse_losses, recon_losses = [], [], []
            t0 = time.time()

            bootstrap_idx = shuffle_rows(bootstrap_idx)

            """ ------- Looping through the shuffled and batched dataset for one epoch -------"""
            for batch_num in range(int(np.ceil(bootstrap_idx.shape[-1] / self.batch_size))):
                batch_idxs = bootstrap_idx[:, batch_num * self.batch_size: (batch_num + 1) * self.batch_size]

                bootstrap_train_obs = train_obs[batch_idxs]
                bootstrap_train_act = train_act[batch_idxs]
                bootstrap_train_delta = train_delta[batch_idxs]
                bootstrap_train_obs_next = train_obs_next[batch_idxs]
                bootstrap_train_back_delta = train_back_delta[batch_idxs]
                bootstrap_train_cp_obs = train_cp_obs[batch_idxs]
                bootstrap_train_cp_act = train_cp_act[batch_idxs]

                feed_dict = self.get_feed_dict(bootstrap_train_obs,
                                               bootstrap_train_act,
                                               bootstrap_train_delta,
                                               bootstrap_train_obs_next,
                                               bootstrap_train_back_delta,
                                               bootstrap_train_cp_obs,
                                               bootstrap_train_cp_act)

                mse_loss, back_mse_loss, recon_loss, _ = sess.run([
                    self.mse_loss, self.back_mse_loss, self.recon_loss, self.train_op
                ], feed_dict=feed_dict)

                mse_losses.append(mse_loss)
                back_mse_losses.append(back_mse_loss)
                recon_losses.append(recon_loss)

            """ ------- Validation -------"""
            if n_valid_split > 0:
                bootstrap_valid_obs = valid_obs[valid_boostrap_idx]
                bootstrap_valid_act = valid_act[valid_boostrap_idx]
                bootstrap_valid_delta = valid_delta[valid_boostrap_idx]
                bootstrap_valid_obs_next = valid_obs_next[valid_boostrap_idx]
                bootstrap_valid_back_delta = valid_back_delta[valid_boostrap_idx]
                bootstrap_valid_cp_obs = valid_cp_obs[valid_boostrap_idx]
                bootstrap_valid_cp_act = valid_cp_act[valid_boostrap_idx]

                feed_dict = self.get_feed_dict(bootstrap_valid_obs,
                                               bootstrap_valid_act,
                                               bootstrap_valid_delta,
                                               bootstrap_valid_obs_next,
                                               bootstrap_valid_back_delta,
                                               bootstrap_valid_cp_obs,
                                               bootstrap_valid_cp_act)

                v_mse_loss, v_back_mse_loss, v_recon_loss = sess.run([
                    self.mse_loss, self.back_mse_loss, self.recon_loss,
                ], feed_dict=feed_dict)

                if verbose:
                    logger.log("Training DynamicsModel - finished epoch %i --"
                               "[Training] mse loss: %.4f  back mse loss: %.4f  recon loss:  %.4f "
                               "[Validation] mse loss: %.4f  back mse loss: %.4f  recon loss:  %.4f  epoch time: %.2f"
                                % (epoch, 
                                   np.mean(mse_losses), np.mean(back_mse_losses), np.mean(recon_losses),
                                   v_mse_loss, v_back_mse_loss, v_recon_loss, time.time() - t0))
                
                # Early Stopping with Validation Loss
                if valid_loss_rolling_average is None:
                    valid_loss_rolling_average = 1.5 * v_recon_loss  # set initial rolling to a higher value avoid too early stopping
                    valid_loss_rolling_average_prev = 2 * v_recon_loss
                    if v_recon_loss < 0:
                        valid_loss_rolling_average = v_recon_loss/1.5  # set initial rolling to a higher value avoid too early stopping
                        valid_loss_rolling_average_prev = v_recon_loss/2

                valid_loss_rolling_average = rolling_average_persitency*valid_loss_rolling_average \
                                                + (1.0-rolling_average_persitency)*v_recon_loss

                if valid_loss_rolling_average_prev < valid_loss_rolling_average:
                    logger.log('Stopping Training of Model since its valid_loss_rolling_average decreased')
                    break

            else:
                if verbose:
                    logger.log("Training DynamicsModel - finished epoch %i --"
                               "[Training] mse loss: %.4f  back mse loss: %.4f  recon loss: %.4f  epoch time: %.2f"
                                % (epoch, np.mean(mse_losses), np.mean(back_mse_losses), np.mean(recon_losses), time.time() - t0))

            valid_loss_rolling_average_prev = valid_loss_rolling_average

        """ ------- Tabular Logging ------- """
        if log_tabular:
            logger.logkv('AvgModelEpochTime', np.mean(epoch_times))
            logger.logkv('Epochs', epoch)

    def save(self, save_path):
        sess = tf.compat.v1.get_default_session()
        ps = sess.run(self.params)
        joblib.dump(ps, save_path)
        if self.normalization is not None:
            norm_save_path = save_path + '_norm_stats'
            joblib.dump(self.normalization, norm_save_path)

    def load(self, load_path):
        sess = tf.compat.v1.get_default_session()
        loaded_params = joblib.load(load_path)
        restores = []
        for p, loaded_p in zip(self.params, loaded_params):
            restores.append(p.assign(loaded_p))
        sess.run(restores)
        if self.normalize_input:
            norm_save_path = load_path + '_norm_stats'
            self.normalization = joblib.load(norm_save_path)

    def compute_normalization(self, obs, act, delta, cp_obs, cp_act, back_delta):
        assert obs.shape[0] == delta.shape[0] == act.shape[0]

        proc_obs = self.env.obs_preproc(obs)

        # store means and std in dict
        self.normalization = OrderedDict()
        self.normalization['obs'] = (np.mean(proc_obs, axis=0), np.std(proc_obs, axis=0))
        self.normalization['delta'] = (np.mean(delta, axis=0), np.std(delta, axis=0))
        self.normalization['act'] = (np.mean(act, axis=0), np.std(act, axis=0))
        self.normalization['cp_obs'] = (np.mean(cp_obs, axis=0), np.std(cp_obs, axis=0))
        self.normalization['cp_act'] = (np.mean(cp_act, axis=0), np.std(cp_act, axis=0))
        self.normalization['back_delta'] = (np.mean(back_delta, axis=0), np.std(back_delta, axis=0))

    def get_normalization_stats(self):
        if self.normalize_input:
            norm_obs_mean = self.normalization['obs'][0]
            norm_obs_std = self.normalization['obs'][1]
            norm_delta_mean = self.normalization['delta'][0]
            norm_delta_std = self.normalization['delta'][1]
            if self.discrete:
                norm_act_mean = np.zeros((self.action_space_dims,))
                norm_act_std = np.ones((self.action_space_dims,))
            else:
                norm_act_mean = self.normalization['act'][0]
                norm_act_std = self.normalization['act'][1]
            if self.state_diff:
                norm_cp_obs_mean = np.zeros((self.obs_space_dims*self.history_length,))
                norm_cp_obs_std = np.ones((self.obs_space_dims*self.history_length,))
            else:
                norm_cp_obs_mean = self.normalization['cp_obs'][0]
                norm_cp_obs_std = self.normalization['cp_obs'][1]
            if self.discrete:
                norm_cp_act_mean = np.zeros((self.action_space_dims*self.history_length,))
                norm_cp_act_std = np.ones((self.action_space_dims*self.history_length,))
            else:
                norm_cp_act_mean = self.normalization['cp_act'][0]
                norm_cp_act_std = self.normalization['cp_act'][1]
            norm_back_delta_mean = self.normalization['back_delta'][0]
            norm_back_delta_std = self.normalization['back_delta'][1]
        else:
            norm_obs_mean = np.zeros((self.proc_obs_space_dims,))
            norm_obs_std = np.ones((self.proc_obs_space_dims,))
            norm_act_mean = np.zeros((self.action_space_dims,))
            norm_act_std = np.ones((self.action_space_dims,))
            norm_delta_mean = np.zeros((self.obs_space_dims,))
            norm_delta_std = np.ones((self.obs_space_dims,))
            norm_cp_obs_mean = np.zeros((self.obs_space_dims*self.history_length,))
            norm_cp_obs_std = np.ones((self.obs_space_dims*self.history_length,))
            norm_cp_act_mean = np.zeros((self.action_space_dims*self.history_length,))
            norm_cp_act_std = np.ones((self.action_space_dims*self.history_length,))
            norm_back_delta_mean = np.zeros((self.obs_space_dims,))
            norm_back_delta_std = np.ones((self.obs_space_dims,))

        return (norm_obs_mean, norm_obs_std, norm_act_mean, norm_act_std, norm_delta_mean, norm_delta_std,
                norm_cp_obs_mean, norm_cp_obs_std, norm_cp_act_mean, norm_cp_act_std, norm_back_delta_mean, norm_back_delta_std)

    def get_feed_dict(self, obs, act, delta, obs_next, back_delta, cp_obs, cp_act):
        (norm_obs_mean, norm_obs_std, norm_act_mean, norm_act_std, norm_delta_mean, norm_delta_std,
         norm_cp_obs_mean, norm_cp_obs_std, norm_cp_act_mean, norm_cp_act_std, norm_back_delta_mean, norm_back_delta_std) = \
            self.get_normalization_stats()

        feed_dict = {
            self.bs_obs_ph: obs,
            self.bs_act_ph: act,
            self.bs_delta_ph: delta,
            self.bs_obs_next_ph: obs_next,
            self.bs_back_delta_ph: back_delta,
            self.bs_cp_obs_ph: cp_obs,
            self.bs_cp_act_ph: cp_act,
            self.norm_obs_mean_ph: norm_obs_mean,
            self.norm_obs_std_ph: norm_obs_std,
            self.norm_act_mean_ph: norm_act_mean,
            self.norm_act_std_ph: norm_act_std,
            self.norm_delta_mean_ph: norm_delta_mean,
            self.norm_delta_std_ph: norm_delta_std,
            self.norm_cp_obs_mean_ph: norm_cp_obs_mean,
            self.norm_cp_obs_std_ph: norm_cp_obs_std,
            self.norm_cp_act_mean_ph: norm_cp_act_mean,
            self.norm_cp_act_std_ph: norm_cp_act_std,
            self.norm_back_delta_mean_ph: norm_back_delta_mean,
            self.norm_back_delta_std_ph: norm_back_delta_std,
        }

        return feed_dict

    def _preprocess_inputs(self, obs, act, delta, cp_obs, cp_act, future_bool, obs_next, back_delta):
        _future_bool= future_bool.reshape(-1)
        _obs = obs.reshape((-1, self.obs_space_dims))
        _act = act.reshape((-1, self.action_space_dims))
        _delta = delta.reshape((-1, self.obs_space_dims))
        _obs_next = obs_next.reshape((-1, self.obs_space_dims))
        _back_delta = back_delta.reshape((-1, self.obs_space_dims))

        _cp_obs = np.tile(cp_obs, (1, self.future_length))
        _cp_obs = _cp_obs.reshape((-1, self.obs_space_dims*self.history_length))
        _cp_act = np.tile(cp_act, (1, self.future_length))
        _cp_act = _cp_act.reshape((-1, self.action_space_dims*self.history_length))

        _obs = _obs[_future_bool>0, :]
        _act = _act[_future_bool>0, :]
        _delta = _delta[_future_bool>0, :]
        _obs_next = _obs_next[_future_bool>0, :]
        _back_delta = _back_delta[_future_bool>0, :]
        _cp_obs = _cp_obs[_future_bool>0, :]
        _cp_act = _cp_act[_future_bool>0, :]
        return _obs, _act, _delta, _obs_next, _back_delta, _cp_obs, _cp_act

def normalize(data_array, mean, std):
    return (data_array - mean) / (std + 1e-10)

def denormalize(data_array, mean, std):
    return data_array * (std + 1e-10) + mean
