import os
import numpy as np
from gym import utils, error
from gym.envs.mujoco import mujoco_env

DEFAULT_CAMERA_CONFIG = {
    'distance': 5.0,
}

class HandEnvRot0(mujoco_env.MujocoEnv, utils.EzPickle):
    """Environment for ball with No-tactile information"""
    def __init__(self,
                 xml_file='xml/ball/MOhand_ball_no_tactile.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.0,
                 reset_noise_scale=0.1,
                 max_reward=0.0,
                 i=-1,
                 nn=0,
                 exclude_current_positions_from_observation=False,
                 rgb_rendering_tracking=True):

        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._max_reward = max_reward
        self._ij = i
        self.face_id = nn
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        """Calculate control cost."""
        return self._ctrl_cost_weight * np.sum(np.square(action))

    def step(self, action):
        """Step through the environment."""
        y_position_before = self.sim.data.qpos[9]
        z_position_before = self.sim.data.qpos[8]
        self.do_simulation(action, self.frame_skip)
        y_position_after = self.sim.data.qpos[9]
        z_position_after = self.sim.data.qpos[8]

        y_velocity = (y_position_after - y_position_before) / self.dt
        z_velocity = (z_position_after - z_position_before) / self.dt
        ctrl_cost = self.control_cost(action)

        # defining different curriculum
        if self._ij > 999999:   
            rot_weight = 1
            lift_weight = 1
        else:
            rot_weight = 1 
            lift_weight = 1 
        
        forward_reward = rot_weight * (self._forward_reward_weight * y_velocity) - lift_weight * (95 * abs(z_position_after - 0.06))
        height = z_position_before
        height_reward = -(95 * abs(z_position_after - 0.06))
        rotation_reward = self._forward_reward_weight * y_velocity
        degree_pos = y_position_after
      
        observation = self.get_obs()
        self._ij += 1 
        current_reward = forward_reward - ctrl_cost
        reward = current_reward

        done = False
        info = {
            'y_position': y_position_after,
            'y_velocity': y_velocity,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'height_reward': height_reward
        }

        return observation, reward, done, info, height, rotation_reward, height_reward, degree_pos 

    def get_obs(self):
        """Get observation from the environment."""
        position_tmp = self.sim.data.qpos.flat.copy()
        velocity_tmp = self.sim.data.qvel.flat.copy()

        position = position_tmp[:-3]
        velocity = velocity_tmp[:-3]

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()           
        return observation

    def reset_model(self):
        """Reset the environment model."""
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(self.model.nv)

        self.set_state(qpos, qvel)
        observation = self.get_obs()
        return observation

    def viewer_setup(self):
        """Setup the viewer."""
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

class HandEnvRot1(mujoco_env.MujocoEnv, utils.EzPickle):
    """Environment for ball with 3D-force"""
    def __init__(self,
                 xml_file='xml/ball/MOhand_ball_3D_force.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.0,
                 reset_noise_scale=0.1,
                 max_reward=0.0,
                 i=-1,
                 nn=0,
                 exclude_current_positions_from_observation=False,
                 rgb_rendering_tracking=True):

        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._max_reward = max_reward
        self._ij = i
        self.face_id = nn
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        """Calculate control cost."""
        return self._ctrl_cost_weight * np.sum(np.square(action))

    def step(self, action):
        """Step through the environment."""
        y_position_before = self.sim.data.qpos[9]
        z_position_before = self.sim.data.qpos[8]
        self.do_simulation(action, self.frame_skip)
        y_position_after = self.sim.data.qpos[9]
        z_position_after = self.sim.data.qpos[8]

        y_velocity = (y_position_after - y_position_before) / self.dt
        z_velocity = (z_position_after - z_position_before) / self.dt
        ctrl_cost = self.control_cost(action)

        if self._ij > 999999:
            rot_weight = 1
            lift_weight = 1
        else:
            rot_weight = 1
            lift_weight = 1

        forward_reward = rot_weight * (self._forward_reward_weight * y_velocity) - lift_weight * (95 * abs(z_position_after - 0.06))
        height = z_position_before
        height_reward = -(95 * abs(z_position_after - 0.06))
        rotation_reward = self._forward_reward_weight * y_velocity
        degree_pos = y_position_after      

        self._ij += 1  

        observation = self.get_obs()
        current_reward = forward_reward - ctrl_cost
        reward = current_reward

        done = False
        info = {
            'y_position': y_position_after,
            'y_velocity': y_velocity,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'height_reward': height_reward
        }

        return observation, reward, done, info, height, rotation_reward, height_reward, degree_pos

    def get_obs(self):
        """Get observation from the environment."""
        position_tmp = self.sim.data.qpos.flat.copy()
        velocity_tmp = self.sim.data.qvel.flat.copy()
        sensor = self.sim.data.sensordata.copy()

        position = position_tmp[:-3]
        velocity = velocity_tmp[:-3]

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity, sensor)).ravel()
                         
        return observation

    def reset_model(self):
        """Reset the environment model."""
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(self.model.nv)

        self.set_state(qpos, qvel)

        observation = self.get_obs()
        return observation

    def viewer_setup(self):
        """Setup the viewer."""
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
