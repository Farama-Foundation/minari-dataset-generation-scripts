import h5py
import minari
import gymnasium as gym
from utils import AdroitStepPreProcessor
from minari.utils.other_option_data_collector import DataCollectorV0
from gymnasium_robotics.envs.adroit_hand.wrappers import SetInitialState


if __name__ == "__main__":
    env = SetInitialState(gym.make('AdroitHandPen-v1', max_episode_steps=200))
    env = DataCollectorV0(env, step_preprocessor=AdroitStepPreProcessor, record_infos=True, max_steps_buffer=10000)

    with h5py.File('/home/rodrigo/Downloads/pen-human-v1.hdf5', 'r') as f:
        actions = f['actions'][:]
        observations = f['observations'][:]
        timeouts = f['timeouts'][:]
        rewards = f['rewards'][:]
        qposes = f['infos']['qpos'][:]
        qvels = f['infos']['qvel'][:]
        desired_oriens = f['infos']['desired_orien'][:]
        
    reset_called = True
    last_episode_step = 0
    for i, (timeout, observation, action, qpos, qvel, desired_orien, reward) in enumerate(zip(timeouts, observations, actions, qposes, qvels, desired_oriens, rewards)):
        if reset_called:
            state_dict = {'qpos': qpos, 'qvel': qvel, 'desired_orien': desired_orien}
            env.reset(initial_state_dict=state_dict)
            reset_called=False      
            
        obs, rew, terminated, truncated, info = env.step(action)
        
        if timeout:
            print('TIMEOUT STEP')
            print(i-last_episode_step)
            last_episode_step = i
            reset_called = True

    minari.create_dataset_from_collector_env(collector_env=env, dataset_name="pen-human-v0", code_permalink=None, author="Rodrigo de Lazcano", author_email="rperezvicente@farama.org")