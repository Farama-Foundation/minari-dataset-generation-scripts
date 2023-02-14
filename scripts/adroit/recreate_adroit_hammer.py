from gymnasium_robotics.envs.adroit_hand.wrappers import SetInitialState
import gymnasium as gym
import h5py
from minari.utils.other_option_data_collector import DataCollectorV0, StepPreProcessor
import minari

class AdroitHammerStepPreProcessor(StepPreProcessor):
    def __call__(self, env, **kwargs):
        step_data = super().__call__(env, **kwargs)

        step_data['state'] = env.get_env_state()
        
        return step_data

class SetInitialState(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

    def reset(self, initial_state_dict=None, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        if initial_state_dict is not None:
            self.env.set_env_state(initial_state_dict)
        return result
    
    
env = SetInitialState(gym.make('AdroitHandHammer-v1', max_episode_steps=650))
env = DataCollectorV0(env, step_preprocessor=AdroitHammerStepPreProcessor, record_infos=True, max_steps_buffer=10000)

with h5py.File('/home/rodrigo/Downloads/hammer-cloned-v1.hdf5', 'r') as f:
    actions = f['actions'][:]
    qposes = f['infos']['qpos'][:]
    qvels = f['infos']['qvel'][:]
    observations = f['observations'][:]
    target_poses = f['infos']['target_pos'][:]
    board_poses = f['infos']['board_pos'][:]
    timeouts = f['timeouts'][:]
    rewards = f['rewards'][:]
reset_called = True
last_episode_step = 0
for i, (timeout, observation, action, target_pos, board_pos, reward, qpos, qvel) in enumerate(zip(timeouts, observations, actions, target_poses, board_poses, rewards, qposes, qvels)):
    if reset_called:
        state_dict = {'qpos': qpos, 'qvel': qvel, 'board_pos': board_pos}
        env.reset(initial_state_dict=state_dict)
        reset_called=False
    # assert np.allclose(observation, obs, rtol=1e-2, atol=1e-4)

    if i % 50000 == 0:
        print(i)
        
    obs, rew, terminated, truncated, info = env.step(action)
    
    if timeout:
        reset_called = True

minari.create_dataset_from_collector_env(collector_env=env, dataset_name="hammer-cloned-v0", code_permalink=None, author="Rodrigo de Lazcano", author_email="rperezvicente@farama.org")        
