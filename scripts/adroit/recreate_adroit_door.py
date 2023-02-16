import h5py
import minari
import gymnasium as gym
from utils import AdroitStepPreProcessor
from minari import DataCollectorV0
from gymnasium_robotics.envs.adroit_hand.wrappers import SetInitialState


if __name__ == "__main__":

    env = SetInitialState(gym.make('AdroitHandDoor-v1', max_episode_steps=600))
    env = DataCollectorV0(env, step_preprocessor=AdroitStepPreProcessor, record_infos=True, max_steps_buffer=200000)
    
    

    with h5py.File('/home/rodrigo/Downloads/door-cloned-v1.hdf5', 'r') as f:
        qposes = f['infos']['qpos'][:]
        qvels = f['infos']['qvel'][:]
        actions = f['actions'][:]
        observations = f['observations'][:]
        door_poses = f['infos']['door_body_pos'][:]
        timeouts = f['timeouts'][:]
        
    reset = True
    last_episode_step = 0
    for i, (timeout, observation, action, door_pos, qpos, qvel) in enumerate(zip(timeouts, observations, actions, door_poses, qposes, qvels)):
        if reset:
            state_dict = {'qpos': qpos, 'qvel': qvel, 'door_body_pos': door_pos}
            env.reset(initial_state_dict=state_dict)      
            reset=False
        # assert np.allclose(observation, obs, rtol=1e-2, atol=1e-4)

        if i % 50000 == 0:
            print(i)
            
        obs, rew, terminated, truncated, info = env.step(action)
        
        if timeout:
            reset = True

    minari.create_dataset_from_collector_env(collector_env=env, dataset_name="door-cloned-v0", code_permalink=None, author="Rodrigo de Lazcano", author_email="rperezvicente@farama.org")    
