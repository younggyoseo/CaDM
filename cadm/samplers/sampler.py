from cadm.samplers.base import BaseSampler
from cadm.samplers.vectorized_env_executor import ParallelEnvExecutor, IterativeEnvExecutor
from cadm.logger import logger
from cadm.utils import utils
from pyprind import ProgBar
import numpy as np
import time
import itertools


class Sampler(BaseSampler):

    def __init__(
            self,
            env,
            policy,
            num_rollouts,
            max_path_length,
            n_parallel=1,
            random_flag=False,
            use_cem=False,
            horizon=None,
            context=False,
            state_diff=False,
            history_length=10,
    ):
        super(Sampler, self).__init__(env, policy, n_parallel, max_path_length)

        self.total_samples = num_rollouts * max_path_length
        self.n_parallel = n_parallel
        self.total_timesteps_sampled = 0
        self.temp_env= env
        self.random_flag = random_flag
        self.context = context
        self.state_diff = state_diff
        self.history_length = history_length
        if len(env.action_space.shape) == 0:
            self.act_dim = env.action_space.n
        else:
            self.act_dim = env.action_space.shape[0]
        
        # setup vectorized environment
        if self.n_parallel > 1:
            self.vec_env = ParallelEnvExecutor(env, n_parallel, num_rollouts, self.max_path_length)
        else:
            self.vec_env = IterativeEnvExecutor(env, num_rollouts, self.max_path_length)
        
        # setup cem mean/var for each rollout
        self.use_cem = use_cem
        self.horizon = horizon
        if self.use_cem:
            self.prev_sol = np.tile(0., [num_rollouts, horizon, self.act_dim])
            self.init_var = np.tile(np.square(2) / 16, [num_rollouts, horizon, self.act_dim])
    
    def reset_cem(self, idx):
        # setup cem mean/var for each rollout
        self.prev_sol[idx] = 0.

    def update_tasks(self):
        pass

    def obtain_samples(self, log=False, log_prefix='', random=False):
        """
        Collect batch_size trajectories from each task

        Args:
            log (boolean): whether to log sampling times
            log_prefix (str) : prefix for logger
            random (boolean): whether the actions are random

        Returns:
            (list): A list of dicts with the samples
        """

        # initial setup / preparation
        paths = []

        n_samples = 0
        num_envs = self.vec_env.num_envs
        running_paths = [_get_empty_running_paths_dict() for _ in range(num_envs)]

        pbar = ProgBar(self.total_samples)
        policy_time, env_time = 0, 0

        policy = self.policy
        if self.use_cem:
            for i in range(num_envs):
                self.reset_cem(i)

        # initial reset of meta_envs
        obses = np.asarray(self.vec_env.reset())
        state_counts = [0] * self.vec_env.num_envs

        # history
        self.obs_dim = obses.shape[1]
        history_state = np.zeros((obses.shape[0], self.obs_dim*self.history_length))
        history_act = np.zeros((obses.shape[0], self.act_dim*self.history_length))

        while n_samples < self.total_samples:

            # execute policy
            t = time.time()
            if random:
                actions = np.stack([self.env.action_space.sample() for _ in range(num_envs)], axis=0)
                agent_infos = {}
            else:
                if self.use_cem:
                    if self.context:
                        cem_solutions, agent_infos = policy.get_actions(obses,
                                                                        init_mean=self.prev_sol,
                                                                        init_var=self.init_var,
                                                                        cp_obs=history_state,
                                                                        cp_act=history_act)
                    else:
                        cem_solutions, agent_infos = policy.get_actions(obses, 
                                                                        init_mean=self.prev_sol,
                                                                        init_var=self.init_var)
                    self.prev_sol[:, :-1] = cem_solutions[:, 1:].copy()
                    self.prev_sol[:, -1:] = 0.
                    actions = cem_solutions[:, 0].copy()
                else:
                    if self.context:
                        actions, agent_infos = policy.get_actions(obses,
                                                                  cp_obs=history_state,
                                                                  cp_act=history_act)
                    else:
                        actions, agent_infos = policy.get_actions(obses)
                if len(self.env.action_space.shape) == 0:
                    actions = actions.reshape(-1)

            policy_time += time.time() - t

            # step environments
            t = time.time()
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)
            env_time += time.time() - t

            #  stack agent_infos and if no infos were provided (--> None) create empty dicts
            agent_infos, env_infos = self._handle_info_dicts(agent_infos, env_infos)

            new_samples = 0
            for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                    rewards, env_infos, agent_infos,
                                                                                    dones):
                if len(self.env.action_space.shape) == 0:
                    action = np.eye(self.act_dim)[action]
                else:
                    if action.ndim == 0:
                        action = np.expand_dims(action, 0)
                assert action.ndim == 1, (action, action.shape)

                # append new samples to running paths
                if isinstance(reward, np.ndarray):
                    reward = reward[0]
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["dones"].append(done)
                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["agent_infos"].append(agent_info)
                running_paths[idx]["cp_obs"].append(history_state[idx].copy())
                running_paths[idx]["cp_act"].append(history_act[idx].copy())

                # making a history buffer
                if state_counts[idx] < self.history_length:
                    if self.state_diff:
                        history_state[idx][state_counts[idx]*self.obs_dim:(state_counts[idx]+1)*self.obs_dim] = next_obses[idx] - observation
                    else:
                        history_state[idx][state_counts[idx]*self.obs_dim:(state_counts[idx]+1)*self.obs_dim] = observation
                    history_act[idx][state_counts[idx]*self.act_dim:(state_counts[idx]+1)*self.act_dim] = action
                else:
                    history_state[idx][:-self.obs_dim] = history_state[idx][self.obs_dim:]
                    if self.state_diff:
                        history_state[idx][-self.obs_dim:] = next_obses[idx] - observation    
                    else:
                        history_state[idx][-self.obs_dim:] = observation                
                    history_act[idx][:-self.act_dim] = history_act[idx][self.act_dim:]
                    history_act[idx][-self.act_dim:] = action

                # if running path is done, add it to paths and empty the running path
                if done:
                    paths.append(dict(
                        observations=np.asarray(running_paths[idx]["observations"]),
                        actions=np.asarray(running_paths[idx]["actions"]),
                        rewards=np.asarray(running_paths[idx]["rewards"]),
                        dones=np.asarray(running_paths[idx]["dones"]),
                        env_infos=utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                        agent_infos=utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                        cp_obs=np.asarray(running_paths[idx]["cp_obs"]),
                        cp_act=np.asarray(running_paths[idx]["cp_act"]),
                    ))
                    new_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = _get_empty_running_paths_dict()
                    if not random and self.use_cem:
                        self.reset_cem(idx)

                    state_counts[idx] = 0
                    history_state[idx] = np.zeros((self.obs_dim*self.history_length))
                    history_act[idx] = np.zeros((self.act_dim*self.history_length))
                else:
                    state_counts[idx] += 1
            pbar.update(self.vec_env.num_envs)
            n_samples += new_samples
            obses = next_obses
        pbar.stop()

        self.total_timesteps_sampled += self.total_samples
        if log:
            logger.logkv(log_prefix + "PolicyExecTime", policy_time)
            logger.logkv(log_prefix + "EnvExecTime", env_time)

        return paths

    def _handle_info_dicts(self, agent_infos, env_infos):
        if not env_infos:
            env_infos = [dict() for _ in range(self.vec_env.num_envs)]
        if not agent_infos:
            agent_infos = [dict() for _ in range(self.vec_env.num_envs)]
        return agent_infos, env_infos


def _get_empty_running_paths_dict():
    return dict(observations=[], actions=[], rewards=[], dones=[], env_infos=[], agent_infos=[],
                cp_obs=[], cp_act=[])