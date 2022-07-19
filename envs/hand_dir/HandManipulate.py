import os
import numpy as np
from gym import utils, error
from gym.envs.mujoco import mujoco_env


DEFAULT_CAMERA_CONFIG = {
    'distance': 5.0,
}

class HandEnvRot0(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='ball/SWoStiffness/MOhand_ball_sti0sen0_servos.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1*0,
                 reset_noise_scale=0.1,
                 max_reward=0.0,
                 max1=1.0,
                 max2=1.0,
                 i=-1,
                 posstd=np.zeros([2001000, 1]),
                 volstd=np.zeros([2001000, 14]),
                 obs_mean = np.zeros(20),
                 obs_var=np.ones(20),
                 nn=0,
                 exclude_current_positions_from_observation=False,
                 rgb_rendering_tracking=True):

        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale
        
        self._max_reward= max_reward

        self._max_1= max1
        self._max_2= max2
        self._posstd=posstd
        self._volstd=volstd
        self._ij=i
        self.face_id=nn

        self._obs_mean = obs_mean
        self._obs_var = obs_var
       
         
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        mujoco_env.MujocoEnv.__init__(self, xml_file,5, rgb_rendering_tracking=rgb_rendering_tracking)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        y_position_before = self.sim.data.qpos[9]
        z_position_before = self.sim.data.qpos[8]
        self.do_simulation(action, self.frame_skip)
        y_position_after = self.sim.data.qpos[9]
        z_position_after = self.sim.data.qpos[8]


        y_velocity = ((y_position_after - y_position_before)
                      / self.dt)
    
        z_velocity = ((z_position_after - z_position_before)
                      / self.dt)


        ctrl_cost = self.control_cost(action)



        if self._ij>999999:   

          rot_weight=1
          lift_weight=1

        else:

          rot_weight=1 
          lift_weight=1 
        

        # lift_weight=1

        # rot_weight=1

        
        forward_reward = rot_weight*(self._forward_reward_weight * y_velocity) - lift_weight*(95*abs(z_position_after-0.06))
        
        height=z_position_before

        height_reward=-(95*abs(z_position_after-0.06))

        rotation_reward=(self._forward_reward_weight * y_velocity)

        degree_pos= y_position_after
        

        observation = self.get_obs()

        self._ij+=1 
        self._posstd[self._ij,:]=z_position_before
        self._volstd[self._ij,:]=observation
        

        current_reward = forward_reward - ctrl_cost
        
        reward=current_reward
        
    
        done = False
        info = {
            'y_position': y_position_after,
            'y_velocity': y_velocity,

            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'height_reward':height_reward
        }

        return observation, reward, done, info, height, rotation_reward,height_reward,degree_pos 

    def get_obs(self):
        position_tmp = self.sim.data.qpos.flat.copy()
        velocity_tmp = self.sim.data.qvel.flat.copy()


        position=position_tmp[:-3]
        velocity=velocity_tmp[:-3]

        # print(position.shape,velocity.shape)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
                         
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self.get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

   # def get_sensor_sensordata(self):
    #    return self.data.sensordata

    

class HandEnvRot1(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='ball/SWoStiffness/MOhand_ball_sti0sen1_servos.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1*0,
                 reset_noise_scale=0.1,
                 max_reward=0.0,
                 max1=1.0,
                 max2=1.0,
                 max3=1.0,
                 i=-1,
                 posstd=np.zeros([2001000, 1]),
                 volstd=np.zeros([2001000, 7]),
                 sensortd=np.zeros([9500000, 3]),
                 obs_mean = np.zeros(23),
                 obs_var=np.ones(23),
                 nn=0,
                 exclude_current_positions_from_observation=False,
                 rgb_rendering_tracking=True):

        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._max_reward= max_reward

        self._max_1= max1
        self._max_2= max2
        self._max_3= max3
        self._posstd=posstd
        self._volstd=volstd
        self._sensord=sensortd
        self._ij=i
        self.face_id=nn

        self._obs_mean = obs_mean
        self._obs_var = obs_var
        

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5, rgb_rendering_tracking=rgb_rendering_tracking)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):

        y_position_before = self.sim.data.qpos[9]
        z_position_before = self.sim.data.qpos[8]
        self.do_simulation(action, self.frame_skip)
        y_position_after = self.sim.data.qpos[9]
        z_position_after = self.sim.data.qpos[8]


        y_velocity = ((y_position_after - y_position_before)
                      / self.dt)

    
        z_velocity = ((z_position_after - z_position_before)
                      / self.dt)

       
      
        ctrl_cost = self.control_cost(action)

        if self._ij>999999:

          rot_weight=1
          lift_weight=1

        else:

          rot_weight=1
          lift_weight=1


        forward_reward = rot_weight*(self._forward_reward_weight * y_velocity) - lift_weight*(95*abs(z_position_after-0.06))
        
    
        height=z_position_before

        height_reward=-(95*abs(z_position_after-0.06))

        rotation_reward=(self._forward_reward_weight * y_velocity)

        degree_pos= y_position_after      


        self._ij+=1  
        self._posstd[self._ij]=z_position_before
        
      
        self._volstd[self._ij]=action
        
  

        observation = self.get_obs()

        current_reward = forward_reward - ctrl_cost
        
        reward=current_reward

        done = False
        info = {
            'y_position': y_position_after,
            'y_velocity': y_velocity,

            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'height_reward':height_reward
        }

        return observation, reward, done, info, height, rotation_reward,height_reward,degree_pos

    def get_obs(self):
        position_tmp = self.sim.data.qpos.flat.copy()
        velocity_tmp = self.sim.data.qvel.flat.copy()
        sensor = self.sim.data.sensordata.copy()


        position=position_tmp[:-3]
        velocity=velocity_tmp[:-3]

        # position=position_tmp
        # velocity=velocity_tmp

        if self._exclude_current_positions_from_observation:
            position = position[1:]


        observation = np.concatenate((position, velocity, sensor)).ravel()
                         
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self.get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    # def get_sensor_sensordata(self):
    #     return self.data.sensordata


class HandEnvRot2(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='ball/SWoStiffness/MOhand_ball_sti0sen2_servos.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1*0,
                 reset_noise_scale=0.1,
                 max_reward=0.0,
                 max1=1.0,
                 max2=1.0,
                 max3=1.0,
                 i=-1,
                 nn=0,
                 posstd=np.zeros([2001000, 10]),
                 volstd=np.zeros([2001000, 10]),
                 sensortd=np.zeros([9500000, 9]),
                 exclude_current_positions_from_observation=False,
                 rgb_rendering_tracking=True):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale
        
        self._max_reward= max_reward
        self._max_1= max1
        self._max_2= max2
        self._max_3= max3
        self._posstd=posstd
        self._volstd=volstd
        self._sensord=sensortd
        self._ij=i
        self.face_id=nn
         
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5, rgb_rendering_tracking=rgb_rendering_tracking)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):

        y_position_before = self.sim.data.qpos[9]
        z_position_before = self.sim.data.qpos[8]
        self.do_simulation(action, self.frame_skip)
        y_position_after = self.sim.data.qpos[9]
        z_position_after = self.sim.data.qpos[8]

        y_velocity = ((y_position_after - y_position_before)
                      / self.dt)
    
        z_velocity = ((z_position_after - z_position_before)
                      / self.dt)

       
        ctrl_cost = self.control_cost(action)

        if self._ij>999999:

          rot_weight=1
          lift_weight=1

        else:

          rot_weight=1
          lift_weight=1
        

        forward_reward = rot_weight*(self._forward_reward_weight * y_velocity) - lift_weight*(95*abs(z_position_after-0.06))
        
        height=z_position_before

        height_reward=-(95*abs(z_position_after-0.06))

        rotation_reward=(self._forward_reward_weight * y_velocity)

        degree_pos= y_position_after
        
        

        self._ij+=1 
        self._posstd[self._ij,:]=z_position_before
        self._volstd[self._ij,:]=z_position_after+(y_position_after - y_position_before) 
        


        observation = self.get_obs()
        current_reward = forward_reward - ctrl_cost
        
        reward=current_reward
      
        done = False
        info = {
            'y_position': y_position_after,
            'y_velocity': y_velocity,

            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost
        }

        return observation, reward, done, info, height, rotation_reward,height_reward,degree_pos

    def get_obs(self):
        position_tmp = self.sim.data.qpos.flat.copy()
        velocity_tmp = self.sim.data.qvel.flat.copy()
        sensor = self.sim.data.sensordata.flat.copy()

        position=position_tmp[:-3]
        velocity=velocity_tmp[:-3]

        if self._exclude_current_positions_from_observation:
            position = position[1:]


        observation = np.concatenate((position, velocity, sensor)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self.get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)


class HandEnvRot3(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='ball/SWoStiffness/MOhand_ball_sti0sen1_servos.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1*0,
                 reset_noise_scale=0.1,
                 max_reward=0.0,
                 max1=1.0,
                 max2=1.0,
                 max3=1.0,
                 i=-1,
                 posstd=np.zeros([2001000, 1]),
                 volstd=np.zeros([2001000, 7]),
                 sensortd=np.zeros([9500000, 3]),
                 obs_mean = np.zeros(23),
                 obs_var=np.ones(23),
                 nn=0,
                 exclude_current_positions_from_observation=False,
                 rgb_rendering_tracking=True):

        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._max_reward= max_reward

        self._max_1= max1
        self._max_2= max2
        self._max_3= max3
        self._posstd=posstd
        self._volstd=volstd
        self._sensord=sensortd
        self._ij=i
        self.face_id=nn

        self._obs_mean = obs_mean
        self._obs_var = obs_var
        

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5, rgb_rendering_tracking=rgb_rendering_tracking)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):

        y_position_before = self.sim.data.qpos[9]
        z_position_before = self.sim.data.qpos[8]
        self.do_simulation(action, self.frame_skip)
        y_position_after = self.sim.data.qpos[9]
        z_position_after = self.sim.data.qpos[8]


        y_velocity = ((y_position_after - y_position_before)
                      / self.dt)

    
        z_velocity = ((z_position_after - z_position_before)
                      / self.dt)

       
      
        ctrl_cost = self.control_cost(action)


        if self._ij>999999:

          rot_weight=1
          lift_weight=1

        else:

          rot_weight=1
          lift_weight=1
        


        forward_reward = rot_weight*(self._forward_reward_weight * y_velocity) - lift_weight*(95*abs(z_position_after-0.06))
        
        height=z_position_before

        height_reward=-(95*abs(z_position_after-0.06))

        rotation_reward=(self._forward_reward_weight * y_velocity)

        degree_pos= y_position_after

        

        self._ij+=1  
        self._posstd[self._ij]=z_position_before
        
      
        self._volstd[self._ij]=y_position_before
        

        observation = self.get_obs()

        current_reward = forward_reward - ctrl_cost
        
        reward=current_reward

        done = False
        info = {
            'y_position': y_position_after,
            'y_velocity': y_velocity,

            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost
        }

        return observation, reward, done, info, height, rotation_reward,height_reward,degree_pos

    def get_obs(self):
      
        position_tmp = self.sim.data.qpos.flat.copy()
        velocity_tmp = self.sim.data.qvel.flat.copy()
        sensor=[int(bool(self.sim.data.sensordata.copy()[0])),int(bool(self.sim.data.sensordata.copy()[1])),int(bool(self.sim.data.sensordata.copy()[2]))]
      
   

        position=position_tmp[:-3]
        velocity=velocity_tmp[:-3]

        # position=position_tmp
        # velocity=velocity_tmp

        if self._exclude_current_positions_from_observation:
            position = position[1:]


        observation = np.concatenate((position, velocity, sensor)).ravel()
                         
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self.get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

