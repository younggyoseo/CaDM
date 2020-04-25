from cadm.dynamics.core.layers import PlusEnsembleCEMMLP
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
            self.act_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, action_space_dims))
            self.delta_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, obs_space_dims))

            self.bs_obs_ph = tf.compat.v1.placeholder(tf.float32, shape=(ensemble_size, None, obs_space_dims))
            self.bs_act_ph = tf.compat.v1.placeholder(tf.float32, shape=(ensemble_size, None, action_space_dims))
            self.bs_delta_ph = tf.compat.v1.placeholder(tf.float32, shape=(ensemble_size, None, obs_space_dims))

            self.norm_obs_mean_ph = tf.compat.v1.placeholder(tf.float32, shape=(proc_obs_space_dims,))
            self.norm_obs_std_ph = tf.compat.v1.placeholder(tf.float32, shape=(proc_obs_space_dims,))
            self.norm_act_mean_ph = tf.compat.v1.placeholder(tf.float32, shape=(action_space_dims,))
            self.norm_act_std_ph = tf.compat.v1.placeholder(tf.float32, shape=(action_space_dims,))
            self.norm_delta_mean_ph = tf.compat.v1.placeholder(tf.float32, shape=(obs_space_dims,))
            self.norm_delta_std_ph = tf.compat.v1.placeholder(tf.float32, shape=(obs_space_dims,))

            self.cem_init_mean_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, self.n_forwards, action_space_dims))
            self.cem_init_var_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, self.n_forwards, action_space_dims))

            # create MLP
            with tf.compat.v1.variable_scope('ff_model'):
                mlp = PlusEnsembleCEMMLP(name,
                                        input_dim=0,
                                        output_dim=obs_space_dims,
                                        hidden_sizes=hidden_sizes,
                                        hidden_nonlinearity=hidden_nonlinearity,
                                        output_nonlinearity=output_nonlinearity,
                                        input_obs_dim=obs_space_dims,
                                        input_act_dim=action_space_dims,
                                        input_obs_var=self.obs_ph,
                                        input_act_var=self.act_ph,
                                        n_forwards=self.n_forwards,
                                        reward_fn=env.tf_reward_fn(),
                                        n_candidates=self.n_candidates,
                                        discrete=self.discrete,
                                        bs_input_obs_var=self.bs_obs_ph,
                                        bs_input_act_var=self.bs_act_ph,
                                        ensemble_size=self.ensemble_size,
                                        n_particles=self.n_particles,
                                        norm_obs_mean_var=self.norm_obs_mean_ph,
                                        norm_obs_std_var=self.norm_obs_std_ph,
                                        norm_act_mean_var=self.norm_act_mean_ph,
                                        norm_act_std_var=self.norm_act_std_ph,
                                        norm_delta_mean_var=self.norm_delta_mean_ph,
                                        norm_delta_std_var=self.norm_delta_std_ph,
                                        obs_preproc_fn=env.obs_preproc,
                                        obs_postproc_fn=env.obs_postproc,
                                        use_cem=self.use_cem,
                                        cem_init_mean_var=self.cem_init_mean_ph,
                                        cem_init_var_var=self.cem_init_var_ph,
                                        deterministic=self.deterministic,
                                        weight_decays=self.weight_decays,
                                        build_policy_graph=True,
                                        )

                self.params = tf.compat.v1.trainable_variables()
            self.delta_pred = mlp.output_var 

            # Outputs from Dynamics Model are normalized delta predictions
            mu, logvar = mlp.mu, mlp.logvar
            bs_normalized_delta = normalize(self.bs_delta_ph, self.norm_delta_mean_ph, self.norm_delta_std_ph)
            
            self.mse_loss = tf.reduce_sum(
                tf.reduce_mean(tf.reduce_mean(tf.square(mu - bs_normalized_delta), axis=-1), axis=-1))
            self.l2_reg_loss = tf.reduce_sum(mlp.l2_regs) 

            if self.deterministic:
                self.recon_loss = self.mse_loss
                self.loss = self.mse_loss + self.l2_reg_loss * self.weight_decay_coeff
            else:
                invvar = tf.exp(-logvar)
                self.mu_loss = tf.reduce_sum(
                    tf.reduce_mean(tf.reduce_mean(tf.square(mu - bs_normalized_delta) * invvar, axis=-1), axis=-1))
                self.var_loss = tf.reduce_sum(
                    tf.reduce_mean(tf.reduce_mean(logvar, axis=-1), axis=-1))
                self.recon_loss = self.mu_loss + self.var_loss
                self.reg_loss = 0.01 * tf.reduce_sum(mlp.max_logvar) - 0.01 * tf.reduce_sum(mlp.min_logvar)
                self.loss = self.recon_loss + self.reg_loss + self.l2_reg_loss * self.weight_decay_coeff

            self.optimizer = optimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

            # tensor_utils
            self._get_cem_action = tensor_utils.compile_function([self.obs_ph,
                                                                  self.norm_obs_mean_ph, self.norm_obs_std_ph,
                                                                  self.norm_act_mean_ph, self.norm_act_std_ph,
                                                                  self.norm_delta_mean_ph, self.norm_delta_std_ph,
                                                                  self.cem_init_mean_ph, self.cem_init_var_ph],
                                                                  mlp.optimal_action_var)
            self._get_rs_action = tensor_utils.compile_function([self.obs_ph,
                                                                  self.norm_obs_mean_ph, self.norm_obs_std_ph,
                                                                  self.norm_act_mean_ph, self.norm_act_std_ph,
                                                                  self.norm_delta_mean_ph, self.norm_delta_std_ph],
                                                                  mlp.optimal_action_var)

            self._get_pred = tensor_utils.compile_function([self.bs_obs_ph, self.bs_act_ph,
                                                            self.norm_obs_mean_ph, self.norm_obs_std_ph,
                                                            self.norm_act_mean_ph, self.norm_act_std_ph,
                                                            self.norm_delta_mean_ph, self.norm_delta_std_ph],
                                                            [mlp.mu, mlp.logvar])

    def get_action(self, obs, cem_init_mean=None, cem_init_var=None):
        norm_obs_mean, norm_obs_std, norm_act_mean, norm_act_std, norm_delta_mean, norm_delta_std = \
            self.get_normalization_stats()
        if cem_init_mean is not None:
            action = self._get_cem_action(obs,
                                          norm_obs_mean, norm_obs_std,
                                          norm_act_mean, norm_act_std,
                                          norm_delta_mean, norm_delta_std,
                                          cem_init_mean, cem_init_var)
        else:
            action = self._get_rs_action(obs,
                                         norm_obs_mean, norm_obs_std,
                                         norm_act_mean, norm_act_std,
                                         norm_delta_mean, norm_delta_std)
        if not self.discrete:
            action = np.minimum(np.maximum(action, -1.0), 1.0)
        return action

    def fit(self, obs, act, obs_next, epochs=1000, compute_normalization=True,
            valid_split_ratio=None, rolling_average_persitency=None, verbose=False, log_tabular=False, max_logging=5000):

        assert obs.ndim == 2 and obs.shape[1] == self.obs_space_dims
        assert obs_next.ndim == 2 and obs_next.shape[1] == self.obs_space_dims
        assert act.ndim == 2 and act.shape[1] == self.action_space_dims

        if valid_split_ratio is None: valid_split_ratio = self.valid_split_ratio
        if rolling_average_persitency is None: rolling_average_persitency = self.rolling_average_persitency

        assert 1 > valid_split_ratio >= 0

        sess = tf.compat.v1.get_default_session()

        delta = self.env.targ_proc(obs, obs_next)

        if self._dataset is None:
            self._dataset = dict(obs=obs, act=act, delta=delta)
        else:
            self._dataset['obs'] = np.concatenate([self._dataset['obs'], obs])
            self._dataset['act'] = np.concatenate([self._dataset['act'], act])
            self._dataset['delta'] = np.concatenate([self._dataset['delta'], delta])

        self.compute_normalization(self._dataset['obs'], self._dataset['act'], self._dataset['delta'])

        dataset_size = self._dataset['obs'].shape[0]
        n_valid_split = min(int(dataset_size * valid_split_ratio), max_logging)
        permutation = np.random.permutation(dataset_size)
        train_obs, valid_obs = self._dataset['obs'][permutation[n_valid_split:]], self._dataset['obs'][permutation[:n_valid_split]]
        train_act, valid_act = self._dataset['act'][permutation[n_valid_split:]], self._dataset['act'][permutation[:n_valid_split]]
        train_delta, valid_delta = self._dataset['delta'][permutation[n_valid_split:]], self._dataset['delta'][permutation[:n_valid_split]]

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
            mse_losses, recon_losses = [], []
            t0 = time.time()

            bootstrap_idx = shuffle_rows(bootstrap_idx)

            """ ------- Looping through the shuffled and batched dataset for one epoch -------"""
            for batch_num in range(int(np.ceil(bootstrap_idx.shape[-1] / self.batch_size))):
                batch_idxs = bootstrap_idx[:, batch_num * self.batch_size: (batch_num + 1) * self.batch_size]
                bootstrap_train_obs = train_obs[batch_idxs]
                bootstrap_train_act = train_act[batch_idxs]
                bootstrap_train_delta = train_delta[batch_idxs]

                feed_dict = self.get_feed_dict(bootstrap_train_obs, bootstrap_train_act, bootstrap_train_delta)

                mse_loss, recon_loss, _ = sess.run([
                    self.mse_loss, self.recon_loss, self.train_op
                ], feed_dict=feed_dict)

                mse_losses.append(mse_loss)
                recon_losses.append(recon_loss)

            """ ------- Validation -------"""
            if n_valid_split > 0:
                bootstrap_valid_obs = valid_obs[valid_boostrap_idx]
                bootstrap_valid_act = valid_act[valid_boostrap_idx]
                bootstrap_valid_delta = valid_delta[valid_boostrap_idx]
                feed_dict = self.get_feed_dict(bootstrap_valid_obs, bootstrap_valid_act, bootstrap_valid_delta)

                v_mse_loss, v_recon_loss = sess.run([
                    self.mse_loss, self.recon_loss,
                ], feed_dict=feed_dict)

                if verbose:
                    logger.log("Training DynamicsModel - finished epoch %i --"
                                "[Training] mse loss: %.4f  recon loss:  %.4f [Validation] mse loss: %.4f  recon loss:  %.4f  epoch time: %.2f"
                                % (epoch, 
                                   np.mean(mse_losses), np.mean(recon_losses), v_mse_loss, v_recon_loss, time.time() - t0))
                
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
                                "[Training] mse loss: %.4f  recon loss: %.4f  epoch time: %.2f"
                                % (epoch, np.mean(mse_losses), np.mean(recon_losses), time.time() - t0))

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

    def compute_normalization(self, obs, act, delta):
        assert obs.shape[0] == delta.shape[0] == act.shape[0]
        proc_obs = self.env.obs_preproc(obs)

        # store means and std in dict
        self.normalization = OrderedDict()
        self.normalization['obs'] = (np.mean(proc_obs, axis=0), np.std(proc_obs, axis=0))
        self.normalization['delta'] = (np.mean(delta, axis=0), np.std(delta, axis=0))
        self.normalization['act'] = (np.mean(act, axis=0), np.std(act, axis=0))

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
        else:
            norm_obs_mean = np.zeros((self.proc_obs_space_dims,))
            norm_obs_std = np.ones((self.proc_obs_space_dims,))
            norm_act_mean = np.zeros((self.action_space_dims,))
            norm_act_std = np.ones((self.action_space_dims,))
            norm_delta_mean = np.zeros((self.obs_space_dims,))
            norm_delta_std = np.ones((self.obs_space_dims,))
        return norm_obs_mean, norm_obs_std, norm_act_mean, norm_act_std, norm_delta_mean, norm_delta_std
    
    def get_feed_dict(self, obs, act, delta):
        norm_obs_mean, norm_obs_std, norm_act_mean, norm_act_std, norm_delta_mean, norm_delta_std = \
            self.get_normalization_stats()
    
        feed_dict = {
            self.bs_obs_ph: obs,
            self.bs_act_ph: act,
            self.bs_delta_ph: delta,
            self.norm_obs_mean_ph: norm_obs_mean,
            self.norm_obs_std_ph: norm_obs_std,
            self.norm_act_mean_ph: norm_act_mean,
            self.norm_act_std_ph: norm_act_std,
            self.norm_delta_mean_ph: norm_delta_mean,
            self.norm_delta_std_ph: norm_delta_std,
        }
        return feed_dict

def normalize(data_array, mean, std):
    return (data_array - mean) / (std + 1e-10)

def denormalize(data_array, mean, std):
    return data_array * (std + 1e-10) + mean