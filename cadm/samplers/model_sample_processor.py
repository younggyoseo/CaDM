from cadm.samplers.base import SampleProcessor
from cadm.utils import tensor_utils
import numpy as np
import operator

class ModelSampleProcessor(SampleProcessor):
    def __init__(
            self,
            discount=0.99,
            max_path_length=200,
            recurrent=False,
            context=False,
            writer=None,
            future_length=10,
    ):

        self.discount = discount
        self.max_path_length = max_path_length
        self.recurrent = recurrent
        self.context = context
        self.writer = writer
        self.future_length = future_length
        
    def process_samples(self, paths, log=False, log_prefix='', itr=None):
        """ Compared with the standard Sampler, ModelBaseSampler.process_samples provides 3 additional data fields
                - observations_dynamics
                - next_observations_dynamics
                - actions_dynamics
            since the dynamics model needs (obs, act, next_obs) for training, observations_dynamics and actions_dynamics
            skip the last step of a path while next_observations_dynamics skips the first step of a path
        """

        assert len(paths) > 0
        recurrent = self.recurrent

        # compute discounted rewards - > returns
        returns = []
        for _, path in enumerate(paths):
            path["returns"] = tensor_utils.discount_cumsum(path["rewards"], self.discount)
            returns.append(path["returns"])

        # 8) log statistics if desired
        self._log_path_stats(paths, log=log, log_prefix=log_prefix, writer=self.writer, itr=itr)
        observations_dynamics = tensor_utils.concat_tensor_list([path["observations"][:-1] for path in paths], recurrent)
        next_observations_dynamics = tensor_utils.concat_tensor_list([path["observations"][1:] for path in paths], recurrent)
        actions_dynamics = tensor_utils.concat_tensor_list([path["actions"][:-1] for path in paths], recurrent)
        timesteps_dynamics = tensor_utils.concat_tensor_list([np.arange((len(path["observations"]) - 1)) for path in paths])

        rewards = tensor_utils.concat_tensor_list([path["rewards"][:-1] for path in paths], recurrent)
        returns = tensor_utils.concat_tensor_list([path["returns"] for path in paths], recurrent)

        if self.context:
            obs_dim = paths[0]["observations"].shape[1]
            act_dim = paths[0]["actions"].shape[1]
            cp_obs_dim = paths[0]["cp_obs"].shape[1]
            cp_act_dim = paths[0]["cp_act"].shape[1]

            concat_obs_list, concat_act_list, concat_next_obs_list, concat_bool_list = [], [], [], []
            for path in paths:
                path_len = path["observations"].shape[0]
                remainder = 0
                if path_len < self.future_length + 1:
                    remainder = self.future_length + 1 - path_len
                    path["observations"] = np.concatenate([path["observations"], np.zeros((remainder, obs_dim))], axis=0)
                    path["actions"] = np.concatenate([path["actions"], np.zeros((remainder, act_dim))], axis=0)
                    path["cp_obs"] = np.concatenate([path["cp_obs"], np.zeros((remainder, cp_obs_dim))], axis=0)
                    path["cp_act"] = np.concatenate([path["cp_act"], np.zeros((remainder, cp_act_dim))], axis=0)

                concat_bool = np.ones((path["observations"][:-1].shape[0], self.future_length))
                for i in range(self.future_length):
                    if i == 0:
                        concat_obs = path["observations"][:-1]
                        concat_act = path["actions"][:-1]
                        concat_next_obs = path["observations"][1:]
                        temp_next_act = path["actions"][1:]
                    else:
                        temp_next_obs = np.concatenate([path["observations"][1+i:], np.zeros((i, obs_dim))], axis=0)
                        concat_obs = np.concatenate([concat_obs, concat_next_obs[:, -obs_dim:]], axis=1)
                        concat_next_obs = np.concatenate([concat_next_obs, temp_next_obs], axis=1)
                        
                        concat_act = np.concatenate([concat_act, temp_next_act], axis=1)
                        temp_next_act = np.concatenate([path["actions"][1+i:], np.zeros((i, act_dim))], axis=0)

                    start_idx = max(i - remainder, 0)
                    concat_bool[-i][start_idx:] = 0
                        
                concat_obs_list.append(concat_obs)
                concat_act_list.append(concat_act)
                concat_next_obs_list.append(concat_next_obs)
                concat_bool_list.append(concat_bool)
            concat_next_obs_list = tensor_utils.concat_tensor_list(concat_next_obs_list, recurrent)
            concat_obs_list = tensor_utils.concat_tensor_list(concat_obs_list, recurrent)
            concat_act_list = tensor_utils.concat_tensor_list(concat_act_list, recurrent)
            concat_bool_list = tensor_utils.concat_tensor_list(concat_bool_list, recurrent)

            cp_observations = tensor_utils.concat_tensor_list([path["cp_obs"][:-1] for path in paths], recurrent)
            cp_actions = tensor_utils.concat_tensor_list([path["cp_act"][:-1] for path in paths], recurrent)

            samples_data = dict(
                observations=observations_dynamics,
                next_observations=next_observations_dynamics,
                actions=actions_dynamics,
                timesteps=timesteps_dynamics,
                rewards=rewards,
                returns=returns, 
                cp_observations=cp_observations,
                cp_actions=cp_actions,
                concat_next_obs=concat_next_obs_list,
                concat_obs=concat_obs_list,
                concat_act=concat_act_list,
                concat_bool=concat_bool_list,
            )
        else:
            samples_data = dict(
                observations=observations_dynamics,
                next_observations=next_observations_dynamics,
                actions=actions_dynamics,
                timesteps=timesteps_dynamics,
                rewards=rewards,
                returns=returns,
            )

        return samples_data
