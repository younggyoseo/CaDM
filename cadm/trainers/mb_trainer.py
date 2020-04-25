import tensorflow as tf
import time
from cadm.logger import logger
from cadm.samplers.utils import rollout_multi
from cadm.samplers.utils import context_rollout_multi
from cadm.envs.normalized_env import normalize
from cadm.envs import * 
from cadm.samplers.vectorized_env_executor import ParallelEnvExecutor

import os
import os.path as osp
import numpy as np
from tensorboardX import SummaryWriter

class Trainer(object):
    """
    Training script for Learning to Adapt

    Args:
        env (Env) :
        sampler (Sampler) :
        sample_processor (SampleProcessor) :
        policy (Policy) :
        n_itr (int) : Number of iterations to train for
        start_itr (int) : Number of iterations policy has already trained for, if reloading
        initial_random_sampled (bool) : Whether or not to collect random samples in the first iteration
        dynamics_model_max_epochs (int): Number of epochs to train the dynamics model
        sess (tf.compat.v1.Session) : current tf session (if we loaded policy, for example)
    """
    def __init__(
            self,
            env,
            env_flag,
            sampler,
            sample_processor,
            policy,
            dynamics_model,
            n_itr,
            writer,
            start_itr=0,
            adapt_batch_size=None,
            initial_random_samples=True,
            num_rollouts=10,
            dynamics_model_max_epochs=200,
            test_max_epochs=200,
            sess=None,
            context=False,
            num_test=4,
            test_range=[[1.0, 2.0], [3.0, 4.0], [16.0, 17.0], [18.0, 19.0]],
            total_test=20,
            no_test_flag=False,
            only_test_flag=False,
            use_cem=False,
            horizon=0,
            test_num_rollouts=10,
            test_n_parallel=2,
            history_length=10,
            state_diff=False,
            ):

        # Environment Attirubtes
        self.env = env
        self.env_flag = env_flag

        # Sampler Attributes
        self.sampler = sampler
        self.sample_processor = sample_processor

        # Dynamics Model Attributes
        self.dynamics_model = dynamics_model

        # Policy Attributes
        self.policy = policy
        self.use_cem = use_cem
        self.horizon = horizon

        # Algorithm Attributes
        self.context = context

        # CaDM Attributes
        self.context = context
        self.history_length = history_length
        self.state_diff = state_diff

        # Training Attributes
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.num_rollouts = num_rollouts
        self.dynamics_model_max_epochs = dynamics_model_max_epochs
        self.test_max_epochs = test_max_epochs
        self.initial_random_samples = initial_random_samples

        # Testing Attributes
        self.no_test = no_test_flag
        self.only_test = only_test_flag
        self.total_test = total_test
        self.num_test = num_test
        self.test_range = test_range
        self.writer = writer
        self.test_num_rollouts = test_num_rollouts
        self.test_n_parallel = test_n_parallel

        if sess is None:
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.compat.v1.Session(config=config)
        self.sess = sess

    def train(self):
        """
        Collects data and trains the dynamics model
        """
        f_test_list = []
        for i in range(0, self.num_test):
            file_name = '%s/test_c%d.txt'%(logger.get_dir(), i)
            f_test = open(file_name, 'w+')
            f_test_list.append(f_test)

        file_name = '%s/test_tot.txt'%(logger.get_dir())
        f_test_tot = open(file_name, 'w+')

        file_name = '%s/train.txt'%(logger.get_dir())
        f_train = open(file_name, 'w+')

        itr_times = []
        t0 = time.time()

        test_env_list = []

        if self.env_flag == 'cartpole':
            env_cls = RandomCartPole_Force_Length
        elif self.env_flag == 'pendulum':
            env_cls = RandomPendulumAll
        elif self.env_flag == 'halfcheetah':
            env_cls = HalfCheetahEnv
        elif self.env_flag == 'cripple_halfcheetah':
            env_cls = CrippleHalfCheetahEnv
        elif self.env_flag == 'ant':
            env_cls = AntEnv
        elif self.env_flag == 'slim_humanoid':
            env_cls = SlimHumanoidEnv
        else:
            raise ValueError(self.env_flag)
        
        train_env = env_cls()
        train_env.seed(0)
        train_env = normalize(train_env)
        for i in range(0, self.num_test):
            test_env = env_cls(self.test_range[i][0], self.test_range[i][1])
            test_env.seed(0)
            test_env = normalize(test_env)
            vec_test_env = ParallelEnvExecutor(test_env, self.test_n_parallel, self.test_num_rollouts, self.test_max_epochs)
            test_env_list.append(vec_test_env)

        if len(train_env.action_space.shape) == 0:
            act_dim = train_env.action_space.n
            discrete = True
        else:
            act_dim = train_env.action_space.shape[0]
            discrete = False
        
        with self.sess.as_default() as sess:

            sess.run(tf.compat.v1.initializers.global_variables())

            start_time = time.time()
            for itr in range(self.start_itr, self.n_itr):
                if not self.only_test:
                    itr_start_time = time.time()
                    logger.log("\n ---------------- Iteration %d ----------------" % itr)

                    time_env_sampling_start = time.time()

                    if self.initial_random_samples and itr == 0:
                        logger.log("Obtaining random samples from the environment...")
                        env_paths = self.sampler.obtain_samples(log=True, random=True, log_prefix='')
                    else:
                        logger.log("Obtaining samples from the environment using the policy...")
                        env_paths = self.sampler.obtain_samples(log=True, log_prefix='')

                    logger.record_tabular('Time-EnvSampling', time.time() - time_env_sampling_start)

                    ''' -------------- Process the samples ----------------'''
                    logger.log("Processing environment samples...")

                    time_env_samp_proc = time.time()
                    samples_data = self.sample_processor.process_samples(env_paths, log=True, itr=itr)
                    logger.record_tabular('Time-EnvSampleProc', time.time() - time_env_samp_proc)

                    ''' --------------- Fit the dynamics model --------------- '''

                    time_fit_start = time.time()

                    logger.log("Training dynamics model for %i epochs ..." % (self.dynamics_model_max_epochs))
                    if self.context:
                        self.dynamics_model.fit(samples_data['concat_obs'],
                                                samples_data['concat_act'],
                                                samples_data['concat_next_obs'],
                                                samples_data['cp_observations'],
                                                samples_data['cp_actions'],
                                                samples_data['concat_bool'],
                                                epochs=self.dynamics_model_max_epochs,
                                                verbose=True,
                                                log_tabular=True)
                    else:
                        self.dynamics_model.fit(samples_data['observations'],
                                                samples_data['actions'],
                                                samples_data['next_observations'],
                                                epochs=self.dynamics_model_max_epochs,
                                                verbose=True,
                                                log_tabular=True)

                    logger.record_tabular('Time-ModelFit', time.time() - time_fit_start)

                    """ ------------------- Logging --------------------------"""
                    logger.logkv('Itr', itr)
                    logger.logkv('n_timesteps', self.sampler.total_timesteps_sampled)

                    logger.logkv('Time', time.time() - start_time)
                    logger.logkv('ItrTime', time.time() - itr_start_time)

                    logger.log("Saving snapshot...")
                    params = self.get_itr_snapshot(itr)
                    self.log_diagnostics(env_paths, '')
                    logger.save_itr_params(itr, params)
                    print(logger.get_dir())
                    checkdir = osp.join(logger.get_dir(), 'checkpoints')
                    os.makedirs(checkdir, exist_ok=True)
                    savepath = osp.join(checkdir, 'params_epoch_{}'.format(itr))
                    self.dynamics_model.save(savepath)
                    logger.log("Saved")
                    
                    logger.dumpkvs()
                else:
                    logger.log("Test - {}/{} iterations".format(itr+1, self.n_itr))
                    checkdir = osp.join(logger.get_dir(), 'checkpoints')
                    loadpath = osp.join(checkdir, 'params_epoch_{}'.format(itr))
                    self.dynamics_model.load(loadpath)
                    logger.log("Succesfully loaded parameters from {}".format(loadpath))
                    if itr != 0:
                        itr_times.append(time.time() - t0)
                        avg_itr_time = np.mean(itr_times)
                        eta = avg_itr_time * (self.n_itr - itr) / 60.
                        logger.log("Test - {}/{} iterations | ETA: {:.2f} mins".format(itr+1, self.n_itr, eta))
                        t0 = time.time()

                if self.no_test:
                    print('no test')
                else:
                    if itr % 1 == 0 or itr == self.n_itr-1:
                        if self.context:
                            rollout = context_rollout_multi
                        else:
                            rollout = rollout_multi

                        total_test_reward = 0.0
                        for i in range(0, self.num_test):
                            test_reward = rollout(vec_env=test_env_list[i], 
                                                  policy=self.policy, 
                                                  discrete=discrete,
                                                  num_rollouts=self.test_num_rollouts, 
                                                  test_total=self.total_test,
                                                  act_dim=act_dim,
                                                  use_cem=self.use_cem,
                                                  horizon=self.horizon,
                                                  context=self.context,
                                                  history_length=self.history_length,
                                                  state_diff=self.state_diff)
                            
                            print("test c" + str(i) +" reward: " + str(test_reward))
                            f_test_list[i].write("{}\n".format(test_reward))
                            f_test_list[i].flush()
                            os.fsync(f_test_list[i].fileno())
                            self.writer.add_scalar("test/c{}".format(i), test_reward, itr)
                            total_test_reward += test_reward / self.num_test

                        f_test_tot.write("{}\n".format(total_test_reward))
                        f_test_tot.flush()
                        os.fsync(f_test_tot.fileno())
                        self.writer.add_scalar("test/total_test", total_test_reward, itr)
                            
                if itr == 1:
                    sess.graph.finalize()
        
        for i in range(0, self.num_test):
            f_test_list[i].close()
            
        f_test_tot.close()
        f_train.close()
        logger.log("Training finished")
        self.sess.close()

    def get_itr_snapshot(self, itr):
        """
        Gets the current policy, env, and dynamics model for storage
        """
        return dict(itr=itr, policy=self.policy, env=self.env, dynamics_model=self.dynamics_model)

    def log_diagnostics(self, paths, prefix):
        self.env.log_diagnostics(paths, prefix)
        self.policy.log_diagnostics(paths, prefix)
