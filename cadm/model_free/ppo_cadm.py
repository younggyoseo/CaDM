import os
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
from cadm.logger import logger
from collections import deque
from baselines.common import explained_variance
from cadm.samplers.vectorized_env_executor import ParallelEnvExecutor
from cadm.samplers.vectorized_env_executor import IterativeEnvExecutor
from cadm.envs.normalized_env import normalize

from cadm.envs import *
import os.path as osp
import itertools

class Model(object):
    def __init__(self, *, policy, proc_obs_dim, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, hidden_size, cp_dim_output, n_layers):
        
        sess = tf.compat.v1.get_default_session()

        act_model = policy(sess, proc_obs_dim, ac_space, nbatch_act, 1, hidden_size, cp_dim_output, reuse=False, n_layers=n_layers)
        train_model = policy(sess, proc_obs_dim, ac_space, nbatch_train, nsteps, hidden_size, cp_dim_output, reuse=True, n_layers=n_layers)

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.compat.v1.placeholder(tf.float32, [None])
        R = tf.compat.v1.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.compat.v1.placeholder(tf.float32, [None])
        OLDVPRED = tf.compat.v1.placeholder(tf.float32, [None])
        LR = tf.compat.v1.placeholder(tf.float32, [])
        CLIPRANGE = tf.compat.v1.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        with tf.compat.v1.variable_scope('model'):
            params = tf.compat.v1.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        def train(lr, cliprange, obs, contexts, returns, masks, actions, values, neglogpacs, states=None):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X:obs, train_model.context_X: contexts, A:actions, ADV:advs, R:returns, LR:lr,
                      CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)
            # If you want to load weights, also save/load observation scaling inside VecNormalize

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        tf.compat.v1.initializers.global_variables().run(session=sess) #pylint: disable=E1101

class Runner(object):

    def __init__(self, *, env, dynamics_model, vec_env, model, nsteps, gamma, lam, env_flag, normalize_flag,
                 history_length, state_diff):
        self.env = env
        self.dynamics_model = dynamics_model
        self.vec_env = vec_env
        self.model = model
        self.nenv = vec_env.num_envs
        self.obs = np.zeros((self.nenv,) + env.observation_space.shape, dtype=model.train_model.X.dtype.name)
        self.obs[:] = vec_env.reset()
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(self.nenv)]
        # We assume that observations are 1-d vector.
        # If we want to train with image observation, we should change this line
        self.obs_dim = env.observation_space.shape[0]
        self.proc_obs_dim = env.proc_observation_space_dims
        if len(env.action_space.shape) == 0:
            self.act_dim = env.action_space.n
            self.discrete = True
        else:
            self.act_dim = env.action_space.shape[0]
            self.discrete = False
            
        self.history_states = np.zeros((self.nenv, self.obs_dim * history_length))
        self.history_acts = np.zeros((self.nenv, self.act_dim * history_length))
        self.state_counts = [0] * self.nenv
        self.reward_list = [0] * self.nenv
        self.env_flag = env_flag
        self.normalize_flag = normalize_flag
        self.history_length = history_length
        self.state_diff = state_diff
        
        self.normalization_stats = {
            'obs_mean': np.zeros((self.proc_obs_dim,)),
            'obs_var': np.ones((self.proc_obs_dim,)),
        }
        self.obs_alpha = 0.0001

    def save(self, save_path):
        norm_save_path = save_path + '_norm_stats'
        joblib.dump(self.normalization_stats, norm_save_path)

    def normalize_data(self, proc_obs, update_stats=False):
        o_a = self.obs_alpha
        obs_mean = self.normalization_stats['obs_mean']
        obs_var = self.normalization_stats['obs_var']

        if update_stats:
            obs_mean = (1 - o_a) * obs_mean + o_a * np.mean(proc_obs, axis=0)
            obs_var = (1 - o_a) * obs_var + o_a * np.mean(np.square(proc_obs - obs_mean), axis=0)
            self.normalization_stats['obs_mean'] = obs_mean
            self.normalization_stats['obs_var'] = obs_var

        normalized_proc_obs = (proc_obs - obs_mean) / (np.sqrt(obs_var) + 1e-10)
        return normalized_proc_obs

    def extract_context(self, cp_obs, cp_act):
        if self.normalize_flag:
            context = self.dynamics_model.get_context_pred(cp_obs, cp_act)

        if context.ndim == 3:
            # context = context.reshape(context.shape[1], context.shape[2])
            context = context[0]
        return context

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, mb_contexts = [],[],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        for step_idx in range(self.nsteps):
            proc_obs = self.env.obs_preproc(self.obs)
            if self.normalize_flag:
                normalized_proc_obs = self.normalize_data(proc_obs, update_stats=True)
            else:
                normalized_proc_obs = np.copy(proc_obs)
            context = self.extract_context(self.history_states, self.history_acts)
            actions, values, self.states, neglogpacs = self.model.step(normalized_proc_obs, context, self.states, self.dones)
            mb_obs.append(normalized_proc_obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            mb_contexts.append(context)

            next_obs, rewards, self.dones, infos = self.vec_env.step(actions)
            self.dones = np.array(self.dones)

            if self.discrete:
                actions = np.eye(self.act_dim)[actions]

            # making a history buffer
            for idx in range(self.nenv):
                if self.state_counts[idx] < self.history_length:
                    if self.state_diff == 0:
                        self.history_states[idx][self.state_counts[idx]*self.obs_dim:(self.state_counts[idx]+1)*self.obs_dim] = self.obs[idx]
                    else:
                        self.history_states[idx][self.state_counts[idx]*self.obs_dim:(self.state_counts[idx]+1)*self.obs_dim] = next_obs[idx] - self.obs[idx]
                    self.history_acts[idx][self.state_counts[idx]*self.act_dim:(self.state_counts[idx]+1)*self.act_dim] = actions[idx]
                else:
                    self.history_states[idx][:-self.obs_dim:] = self.history_states[idx][self.obs_dim:]
                    if self.state_diff == 0:
                        self.history_states[idx][-self.obs_dim:] = self.obs[idx]
                    else:
                        self.history_states[idx][-self.obs_dim:] = next_obs[idx] - self.obs[idx]
                    self.history_acts[idx][:-self.act_dim:] = self.history_acts[idx][self.act_dim:]
                    self.history_acts[idx][-self.act_dim:] = actions[idx]
                    
            self.obs[:] = next_obs

            if (step_idx + 1) % self.nsteps == 0:
                context = self.extract_context(self.history_states, self.history_acts)
                # we should normalize input because next_obs is not normalized here
                proc_obs = self.env.obs_preproc(self.obs)
                if self.normalize_flag:
                    normalized_proc_obs = self.normalize_data(proc_obs, update_stats=False)
                else:
                    normalized_proc_obs = np.copy(proc_obs)
                last_values = self.model.value(normalized_proc_obs, context, self.states, self.dones)

            # if the running path is done, empty history
            for idx in range(self.nenv):
                if self.dones[idx]:
                    self.history_states[idx] = np.zeros((self.obs_dim * self.history_length))
                    self.history_acts[idx] = np.zeros((self.act_dim * self.history_length))
                    epinfos.append({'r': self.reward_list[idx], 'l': self.state_counts[idx]}) # a hack to calculate average rewards
                    self.state_counts[idx] = 0
                    self.reward_list[idx] = 0
                else:
                    self.state_counts[idx] += 1
                    self.reward_list[idx] += rewards[idx]

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)

        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_contexts = np.asarray(mb_contexts, dtype=np.float32)

        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        # mb_returns = (mb_returns - mb_returns.mean()) / (mb_returns.std() + 1e-8)

        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_contexts)), mb_states, epinfos)

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def constfn(val):
    def f(_):
        return val
    return f


def learn(*, policy, dynamics_model, env, nsteps, total_timesteps, ent_coef, lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            history_length=10, state_diff=1, load_path='', n_layers=2,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, n_parallel=1, num_rollouts=1, max_path_length=200, seed=0,
            hidden_size=512, test_range=[], num_test=4, total_test=20,
            test_interval=0, env_flag='pendulum', normalize_flag=0,
            no_test_flag=False, only_test_flag=False, cp_dim_output=10,):

    f_test_list = []
    for i in range(0, num_test):
        file_name = '%s/test_c%d.txt'%(logger.get_dir(), i)
        f_test = open(file_name, 'w+')
        f_test_list.append(f_test)
        
    file_name = '%s/test_tot.txt'%(logger.get_dir())
    f_test_tot = open(file_name, 'w+')
    
    file_name = '%s/train.txt'%(logger.get_dir())
    f_train = open(file_name, 'w+')

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    if n_parallel > 1:
        vec_env = ParallelEnvExecutor(env, n_parallel, num_rollouts, max_path_length)
    else:
        vec_env = IterativeEnvExecutor(env, num_rollouts, max_path_length)

    nenvs = vec_env.num_envs
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    obs_dim = env.observation_space.shape[0]
    proc_obs_dim = env.proc_observation_space_dims
    if len(env.action_space.shape) == 0:
        act_dim = env.action_space.n
        discrete = True
    else:
        act_dim = env.action_space.shape[0]
        discrete = False

    make_model = lambda : Model(policy=policy, proc_obs_dim=proc_obs_dim, ac_space=ac_space, nbatch_act=nenvs, 
                                nbatch_train=nbatch_train, nsteps=nsteps, ent_coef=ent_coef, 
                                vf_coef=vf_coef, max_grad_norm=max_grad_norm, hidden_size=hidden_size,
                                cp_dim_output=cp_dim_output, n_layers=n_layers)
    
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    runner = Runner(env=env, dynamics_model=dynamics_model, vec_env=vec_env, model=model, nsteps=nsteps, gamma=gamma, lam=lam, env_flag=env_flag,
                    normalize_flag=normalize_flag, history_length=history_length, state_diff=state_diff)

    if load_path:
        dynamics_model.load(load_path)
        logger.log("Successfully loaded parameters from {}".format(load_path))
    else:
        logger.log("Failed to load parameters from {}".format(load_path))

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    nupdates = total_timesteps//nbatch
    
    test_env_list = []
    if env_flag == 'cartpole':
        env_cls = RandomCartPole_Force_Length
    elif env_flag == 'pendulum':
        env_cls = RandomPendulumAll
    elif env_flag == 'halfcheetah':
        env_cls = HalfCheetahEnv
    elif env_flag == 'cripple_halfcheetah':
        env_cls = CrippleHalfCheetahEnv
    elif env_flag == 'ant':
        env_cls = AntEnv
    elif env_flag == 'slim_humanoid':
        env_cls = SlimHumanoidEnv

    train_env = env_cls()
    train_env.seed(0)
    train_env = normalize(train_env)
    for i in range(0, num_test):
        test_env = env_cls(test_range[i][0], test_range[i][1])
        test_env.seed(0)
        test_env = normalize(test_env)
        vec_test_env = ParallelEnvExecutor(test_env, n_parallel, 10, max_path_length)
        test_env_list.append(vec_test_env)
        
    if n_parallel > 1:
        vec_train_env = ParallelEnvExecutor(train_env, n_parallel, 10, max_path_length)
    else:
        vec_train_env = IterativeEnvExecutor(train_env, 10, max_path_length)
    
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        obs, returns, masks, actions, values, neglogpacs, contexts, states, epinfos = runner.run() #pylint: disable=E0632
        epinfobuf.extend(epinfos)
        mblossvals = []
        if states is None: # nonrecurrent version
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, contexts, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        else: # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, contexts, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('epminrew', safemin([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('epmaxrew', safemax([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
            runner.save(savepath)

        if not no_test_flag:
            # TEST
            if test_interval and update % test_interval == 0 and logger.get_dir():
                train_reward = context_pred_rollout_multi(vec_env=vec_train_env,
                                                          env=train_env,
                                                          obs_dim=obs_dim,
                                                          act_dim=act_dim,
                                                          discrete=discrete,
                                                          model=model,
                                                          history_length=history_length,
                                                          state_diff=state_diff,
                                                          test_total=total_test,
                                                          runner=runner)
                    
                print("train reward: " + str(train_reward))
                f_train.write("{}\n".format(train_reward))
                f_train.flush()
                os.fsync(f_train.fileno())
                
                total_test_reward = 0.0
                for i in range(0, num_test):
                    test_reward = context_pred_rollout_multi(vec_env=test_env_list[i],
                                                             env=test_env,
                                                             obs_dim=obs_dim,
                                                             act_dim=act_dim,
                                                             discrete=discrete,
                                                             model=model,
                                                             history_length=history_length,
                                                             state_diff=state_diff,
                                                             test_total=total_test,
                                                             runner=runner)
                        
                    print("test c" + str(i) +" reward: " + str(test_reward))
                    f_test_list[i].write("{}\n".format(test_reward))
                    f_test_list[i].flush()
                    os.fsync(f_test_list[i].fileno())
                    total_test_reward += test_reward

                f_test_tot.write("{}\n".format(total_test_reward))
                f_test_tot.flush()
                os.fsync(f_test_tot.fileno())
    for i in range(0, num_test):
        f_test_list[i].close()
        
    f_test_tot.close()
    f_train.close()
    logger.log("Training finished")

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def safemin(xs):
    return np.nan if len(xs) == 0 else np.min(xs)

def safemax(xs):
    return np.nan if len(xs) == 0 else np.max(xs)

def context_pred_rollout_multi(vec_env, env, obs_dim, act_dim, discrete, model, history_length, state_diff, test_total, runner):
    num_envs = vec_env.num_envs
    history_states = np.zeros((num_envs, obs_dim * history_length))
    history_acts = np.zeros((num_envs, act_dim * history_length))
    state_counts = [0] * num_envs
    
    n_test = 0
    total_reward_list = []
    test_reward_list = np.zeros(num_envs)

    obses = vec_env.reset()
    states = model.initial_state
    dones = [False for _ in range(num_envs)]

    while n_test < test_total:
        obses = np.array(obses)
        proc_obses = env.obs_preproc(obses)
        if runner.normalize_flag:
            normalized_proc_obses = runner.normalize_data(proc_obses, update_stats=False)
        else:
            normalized_proc_obses = proc_obses
        context = runner.extract_context(history_states, history_acts)
        actions, _, states, _ = model.step(normalized_proc_obses, context, states, dones)
        next_obses, rewards, dones, _ = vec_env.step(actions)

        for idx, obs, action, reward, done in zip(itertools.count(), obses, actions, rewards, dones):
            if discrete:
                action = np.eye(act_dim)[action]
            else:
                if action.ndim == 0:
                    action = np.expand_dims(action, 0)
            test_reward_list[idx] += reward

            if state_counts[idx] < history_length:                
                if state_diff == 0:
                    history_states[idx][state_counts[idx]*obs_dim:(state_counts[idx]+1)*obs_dim] = obs
                else:
                    history_states[idx][state_counts[idx]*obs_dim:(state_counts[idx]+1)*obs_dim] = next_obses[idx] - obs
                history_acts[idx][state_counts[idx]*act_dim:(state_counts[idx]+1)*act_dim] = action
            else:                                                                                 
                history_states[idx][:-obs_dim] = history_states[idx][obs_dim:]
                if state_diff == 0:
                    history_states[idx][-obs_dim:] = obs                 
                else:
                    history_states[idx][-obs_dim:] = next_obses[idx] - obs
                history_acts[idx][:-act_dim] = history_acts[idx][act_dim:]
                history_acts[idx][-act_dim:] = action

            if done:
                n_test += 1
                total_reward_list.append(test_reward_list[idx])
                test_reward_list[idx] = 0
                history_states[idx] = np.zeros((obs_dim * history_length))
                history_acts[idx] = np.zeros((act_dim * history_length))
                state_counts[idx] = 0
            else:
                state_counts[idx] += 1
        obses = next_obses
    return np.average(total_reward_list)