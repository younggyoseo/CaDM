from cadm.policies.base import Policy
from cadm.utils.serializable import Serializable
import numpy as np


class MPCController(Policy, Serializable):
    def __init__(
            self,
            name,
            env,
            dynamics_model,
            reward_model=None,
            discount=1,
            use_cem=False,
            n_candidates=1024,
            horizon=10,
            num_rollouts=10,
            context=False,
    ):
        self.dynamics_model = dynamics_model
        self.reward_model = reward_model
        self.discount = discount
        self.n_candidates = n_candidates
        self.horizon = horizon
        self.use_cem = use_cem
        self.env = env
        self.context = context

        self.unwrapped_env = env
        while hasattr(self.unwrapped_env, 'wrapped_env'):
            self.unwrapped_env = self.unwrapped_env.wrapped_env

        # make sure that enc has reward function
        assert hasattr(self.unwrapped_env, 'reward'), "env must have a reward function"

        Serializable.quick_init(self, locals())
        super(MPCController, self).__init__(env=env)

    @property
    def vectorized(self):
        return True

    def get_action(self, observation, init_mean=None, init_var=None):
        if observation.ndim == 1:
            observation = observation[None]

        if self.use_cem:
            action = self.get_cem_gpu_action(observation, init_mean, init_var)
        else:
            # override with random shooting on GPU
            action = self.get_rs_gpu_action(observation)

        return action, dict()

    def get_actions(self, observations, cp_obs=None, cp_act=None, init_mean=None, init_var=None):
        if self.context:
            if self.use_cem:
                actions = self.get_cem_gpu_action(observations, init_mean, init_var, cp_obs, cp_act)
            else:
                # override with random shooting on GPU
                actions = self.get_rs_gpu_action(observations, cp_obs, cp_act)
        else: 
            if self.use_cem:
                actions = self.get_cem_gpu_action(observations, init_mean, init_var)
            else:
                # override with random shooting on GPU
                actions = self.get_rs_gpu_action(observations)

        return actions, dict()

    def get_random_action(self, n):
        if len(self.unwrapped_env.action_space.shape) == 0:
            return np.random.randint(self.unwrapped_env.action_space.n, size=n)
        else:
            return np.random.uniform(low=self.action_space.low,
                                     high=self.action_space.high, size=(n,) + self.action_space.low.shape)

    def get_rs_gpu_action(self, observations, cp_obs=None, cp_act=None):
        if self.context:
            action = self.dynamics_model.get_action(observations, cp_obs, cp_act)
        else:
            action = self.dynamics_model.get_action(observations)
        return action
    
    def get_cem_gpu_action(self, observations, init_mean, init_var, cp_obs=None, cp_act=None,):
        if self.context:
            action = self.dynamics_model.get_action(observations, cp_obs, cp_act, init_mean, init_var)
        else:
            action = self.dynamics_model.get_action(observations, init_mean, init_var)
        return action
