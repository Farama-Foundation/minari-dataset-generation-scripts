import h5py
import minari
import mujoco
import numpy as np
import gymnasium as gym
from utils import AdroitStepPreProcessor
from minari.utils.other_option_data_collector import DataCollectorV0
from gymnasium_robotics.envs.adroit_hand.wrappers import SetInitialState
    

if __name__ == "__main__":    
    env = gym.make('AdroitHandRelocate-v1', max_episode_steps=650)

    with h5py.File('/home/rodrigo/Downloads/relocate-cloned-v1.hdf5', 'r') as f:
        actions = f['actions'][:]
        observations = f['observations'][:]
        timeouts = f['timeouts'][:]
        rewards = f['rewards'][:]
        qposes = f['infos']['qpos'][:]
        qvels = f['infos']['qvel'][:]

    palm_poses = np.zeros((qposes.shape[0],3))
    obj_poses = np.zeros((qposes.shape[0], 3))
    target_poses = np.zeros((qposes.shape[0], 3))

    reset_called = True
    last_episode_step = 0
    env.reset()
    for i, (timeout, action, qpos, qvel, observation) in enumerate(zip(timeouts, actions, qposes, qvels, observations)):
        env.set_state(qpos, qvel)
        mujoco.mj_step(env.model, env.data, nstep=env.frame_skip)
        palm_pos = env.data.site_xpos[env.S_grasp_site_id].ravel().copy()
        obj_poses[i, :] = palm_pos-observation[30:33]
        target_poses[i, :] = obj_poses[i, :]-observation[-3:]

    env.close()

    env = SetInitialState(gym.make('AdroitHandRelocate-v1', max_episode_steps=200))
    env = DataCollectorV0(env, step_preprocessor=AdroitStepPreProcessor, record_infos=True, max_steps_buffer=10000)

    reset_called = True
    last_episode_step = 0
    for i, (timeout, observation, action, target_pos, obj_pos, reward, qpos, qvel) in enumerate(zip(timeouts, observations, actions, target_poses, obj_poses, rewards, qposes, qvels)):
        if reset_called:
            state_dict = {'qpos': qpos, 'qvel': qvel, 'obj_pos': obj_pos, 'target_pos': target_pos}
            env.reset(initial_state_dict=state_dict)
            reset_called=False
        # assert not np.allclose(observation, obs, rtol=1e-2, atol=1e-4)
        obs, rew, terminated, truncated, info = env.step(action)
        
        if i % 50000 == 0:
            print(i)
            
        if timeout:
            reset_called = True 
            # print('TIMEOUT STEP')
            # print(i-last_episode_step)
            # last_episode_step = i   

    minari.create_dataset_from_collector_env(collector_env=env, dataset_name="relocate-cloned-v0", code_permalink=None, author="Rodrigo de Lazcano", author_email="rperezvicente@farama.org")