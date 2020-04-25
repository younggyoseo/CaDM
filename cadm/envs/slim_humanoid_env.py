import numpy as np
import tensorflow as tf
from gym.envs.mujoco import mujoco_env
from gym import utils
import os


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class SlimHumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    # For SlimHumanoid, we followed the implementation from
    # https://github.com/WilsonWangTHU/mbbl/blob/master/mbbl/env/gym_env/humanoid.py
    def __init__(self, mass_scale_set=[0.8, 0.9, 1.0, 1.15, 1.25], damping_scale_set=[0.8, 0.9, 1.0, 1.15, 1.25]):
        self.prev_pos = None
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/humanoid.xml' % dir_path, 5)

        self.original_mass = np.copy(self.model.body_mass)
        self.original_damping = np.copy(self.model.dof_damping)

        self.mass_scale_set = mass_scale_set
        self.damping_scale_set = damping_scale_set

        utils.EzPickle.__init__(self, mass_scale_set, damping_scale_set)
    
    def _set_observation_space(self, observation):
        super(SlimHumanoidEnv, self)._set_observation_space(observation)
        proc_observation = self.obs_preproc(observation[None])
        self.proc_observation_space_dims = proc_observation.shape[-1]

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat])

    def obs_preproc(self, obs):
        return obs

    def obs_postproc(self, obs, pred):
        return obs + pred

    def targ_proc(self, obs, next_obs):
        return next_obs - obs

    def step(self, a):
        old_obs = np.copy(self._get_obs())
        self.do_simulation(a, self.frame_skip)
        data = self.sim.data
        lin_vel_cost = 0.25 / 0.015 * old_obs[..., 22]
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = 0.
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        alive_bonus = 5.0 * (1 - float(done))
        done = False
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        pos_before = mass_center(self.model, self.sim)
        self.prev_pos = np.copy(pos_before)

        random_index = self.np_random.randint(len(self.mass_scale_set))
        self.mass_scale = self.mass_scale_set[random_index]

        random_index = self.np_random.randint(len(self.damping_scale_set))
        self.damping_scale = self.damping_scale_set[random_index]

        self.change_env()

        return self._get_obs()

    def reward(self, obs, action, next_obs):
        ctrl = action

        lin_vel_cost = 0.25 / 0.015 * obs[..., 22]
        quad_ctrl_cost = 0.1 * np.sum(np.square(ctrl), axis=-1)
        quad_impact_cost = 0.

        done = (obs[..., 1] < 1.0) | (obs[..., 1] > 2.0)
        alive_bonus = 5.0 * -1 * done

        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus

        return reward

    def tf_reward_fn(self):
        def _thunk(obs, act, next_obs):
            ctrl = act

            # lin_vel_cost = 1.25 * obs[..., 0]
            lin_vel_cost = 0.25 / 0.015 * obs[..., 22]
            quad_ctrl_cost = 0.1 * tf.reduce_sum(tf.square(ctrl), axis=-1)
            quad_impact_cost = 0.

            alive_bonus = 5.0 * tf.cast(
                tf.logical_and(tf.greater(obs[..., 1], 1.0),
                               tf.less(obs[..., 1], 2.0)),
                dtype=tf.float32)

            reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
            return reward
        return _thunk
    
    def change_env(self):
        mass = np.copy(self.original_mass)
        damping = np.copy(self.original_damping)
        mass *= self.mass_scale
        damping *= self.damping_scale

        self.model.body_mass[:] = mass
        self.model.dof_damping[:] = damping

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20
    
    def get_sim_parameters(self):
        return np.array([self.mass_scale, self.damping_scale])

    def num_modifiable_parameters(self):
        return 2

    def log_diagnostics(self, paths, prefix):
        return