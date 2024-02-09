from gym.envs.registration import registry, register, make, spec

#No Force
env_name = 'HandManipulate-v0'
if env_name in registry.env_specs:
    del registry.env_specs[env_name]
register(
    id='HandManipulate-v0',
    entry_point='envs.hand:HandEnvRot0',
    max_episode_steps=1000,

    reward_threshold=1000.0,
)

#3D Force
env_name = 'HandManipulate-v1'
if env_name in registry.env_specs:
    del registry.env_specs[env_name]
register(
    id='HandManipulate-v1',
    entry_point='envs.hand:HandEnvRot1',
    max_episode_steps=1000,

    reward_threshold=1000.0,
)

