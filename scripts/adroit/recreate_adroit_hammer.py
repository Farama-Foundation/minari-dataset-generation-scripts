import h5py
import minari
import gymnasium as gym
from utils import AdroitStepPreProcessor
from minari.utils.other_option_data_collector import DataCollectorV0
from gymnasium_robotics.envs.adroit_hand.wrappers import SetInitialState

if __name__ == "__main__":
    
    env = SetInitialState(gym.make('AdroitHandHammer-v1', max_episode_steps=650))
    env = DataCollectorV0(env, step_preprocessor=AdroitStepPreProcessor, record_infos=True, max_steps_buffer=10000)

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
