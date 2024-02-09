import gym
import numpy as np
from envs.hand.HandManipulate import HandEnvRot0
from envs.hand.HandManipulate import HandEnvRot1
from gym.wrappers import Monitor
from envs.algorithm.pposgd_gamma import PPO1_gamma
from stable_baselines.common.policies import MlpPolicy as common_MlpPolicy
import multiprocessing as mp
import tensorflow as tf
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")
import mujoco_py

print("mujoco-py version:", gym.__version__)

def train_hand_manipulation(PPO_value):
    """
    Train a PPO  for hand manipulation.

    Args:
        PPO_value (int): The PPO value for training.

    Returns:
        None
    """
    RL_method = "PPO1"
    experiment_ID = "hand_manipluation_traninign"
    total_timesteps = 2000000
    sensory_versions = 1
    PPO_par = 'results_2tactile_information_2M'
    
    for sensory_version_ctrl in range(sensory_versions):
        sensory_info = "sensory_{}".format(sensory_version_ctrl) 
        PPO_value_str = "{}_{}".format(RL_method, PPO_value)
        log_dir = f"./logs/{experiment_ID}/{RL_method}/{sensory_info}/"
        model_dir = f"./logs/{experiment_ID}/{RL_method}/{sensory_info}/{PPO_value_str}"
        save_par = 'HandManipulate-v{}'.format(sensory_version_ctrl)

        # Define the environment
        env = gym.make('HandManipulate-v{}'.format(sensory_version_ctrl))

        # Set up monitoring
        env = gym.wrappers.Monitor(env, model_dir + "Monitor/", video_callable=False, force=True, uid="Monitor_info")
        
        # Define the initial model
        if RL_method == "PPO1":
            model = PPO1_gamma(common_MlpPolicy, env, seed=PPO_value, verbose=1, tensorboard_log=log_dir)
        else:
            raise ValueError("Invalid RL mode")

        # Set random seeds
        env.seed(1)
        env.action_space.seed(1)
        tf.random.set_random_seed(1)

        # Train the model
        model.learn(total_timesteps=total_timesteps)

        # Save the trained model
        model.save(model_dir + "/model")

        # Save parameter information
        with open(log_dir + 'parameter.txt', 'w') as f:
            f.writelines([save_par + "\n", PPO_par])

if __name__ == '__main__':
    # Use multiprocessing to train multiple PPO versions
    pool = mp.Pool(mp.cpu_count())
    # number of trails 
    PPO_versions = 6
    # # pool.map_async(train_hand_manipulation, [row for row in range(1, PPO_versions)])
    # pool.close()
    # pool.join()

    train_hand_manipulation(1)
