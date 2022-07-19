import gym
import numpy as np
from envs.hand_dir.HandManipulate import HandEnvRot0
from envs.hand_dir.HandManipulate import HandEnvRot1
from envs.hand_dir.HandManipulate import HandEnvRot2
from envs.hand_dir.HandManipulate import HandEnvRot3
from envs.hand_dir.HandManipulate import HandEnvC0
from envs.hand_dir.HandManipulate import HandEnvC1
from envs.hand_dir.HandManipulate import HandEnvC2
from envs.hand_dir.HandManipulate import HandEnvC3


from gym.wrappers import Monitor

from stable_baselines.common.policies import MlpPolicy as common_MlpPolicy
from stable_baselines.common.policies import MlpLnLstmPolicy as common_MlpLnLstmPolicy
from stable_baselines.ddpg.policies import MlpPolicy as DDPG_MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from envs.ppo1.pposgd_simple import PPO1
from envs.ppo1.pposgd_gamma import PPO1_gamma
from stable_baselines import PPO2, DDPG

from stable_baselines.common.cmd_util import make_vec_env
from gym import Wrapper

import multiprocessing as mp
import tensorflow as tf
from gym.utils import seeding
import random

def ppo1_nmileg_pool(PPO_value):
	RL_method = "PPO1"
	experiment_ID = "handmotorservo_May24"
	save_name_extension = RL_method
	total_timesteps =2000000
	sensory_versions = 1
	PPO_par='results_4sensory_2M_95coff_60runs_gamma0.8_linear_learningrate_episode8'
	for sensory_version_ctrl in range(0, sensory_versions):
		
		sensory_info = "sensory_{}".format(sensory_version_ctrl) 
		PPO_value_str = "{}_{}".format(RL_method,PPO_value)
		log_dir = "./logs/{}/{}/{}/".format(experiment_ID, RL_method, sensory_info)
		model_dir = "./logs/{}/{}/{}/{}".format(experiment_ID, RL_method, sensory_info,PPO_value_str)
		save_par='HandManipulate-v{}'.format(sensory_version_ctrl)

		## defining the environments
		# env = gym.make('HandManipulate-v{}'.format(sensory_version_ctrl))

        ##Single environment 
		env = gym.make('HandManipulate-v0')

		## Normalize environment 
		# env = NormalizedEnv(env,normalize_obs=False,normalize_reward=False,flatten_obs=False)

		## setting the Monitor
		env = gym.wrappers.Monitor(env, model_dir+"Monitor/", video_callable=False, force=True, uid="Monitor_info")
		
		## defining the initial model
		if RL_method == "PPO1":
			model = PPO1_gamma(common_MlpPolicy, env,seed=PPO_value, verbose=1, tensorboard_log=log_dir)
		elif RL_method == "PPO2":
			model = PPO2(common_MlpLnLstmPolicy, env,seed=PPO_value, verbose=1, tensorboard_log=log_dir)
		elif RL_method == "DDPG":
			env = DummyVecEnv([lambda: env])
			n_actions = env.action_space.shape[-1]
			param_noise = None
			action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5)* 5 * np.ones(n_actions))
			model = DDPG(DDPG_MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise, tensorboard_log=log_dir)
		else:
			raise ValueError("Invalid RL mode")

		## setting the environment on the model
		# model.set_env(env)


        ## setting the random seed for some of the random instances
		# random_seed = 1
		# random.seed(random_seed)
		env.seed(1)
		env.action_space.seed(1)
		tf.random.set_random_seed(1)


		## training the model
		model.learn(total_timesteps=total_timesteps)
		## saving the trained model
		model.save(model_dir+"/model")


		## Saving the file results explanation
		with open(log_dir+'parameter.txt', 'w') as a:
			a.writelines([save_par+"\n",PPO_par])

	return None




if __name__ == '__main__':
	pool = mp.Pool(mp.cpu_count())
	PPO_versions =6
	pool.map_async(ppo1_nmileg_pool, [row for row in range(1,PPO_versions)])
	pool.close()

	pool.join()
	
	
	# ppo1_nmileg_pool(1)