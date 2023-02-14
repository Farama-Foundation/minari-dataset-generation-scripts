import gymnasium as gym
from gymnasium_robotics.envs.adroit_hand.wrappers import SetInitialState
import h5py
from minari.utils.other_option_data_collector import DataCollectorV0
from utils import AdroitStepPreProcessor
import minari

class AdroitDoorStepPreProcessor(StepPreProcessor):
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


env = SetInitialState(gym.make('AdroitHandDoor-v1', max_episode_steps=600))
env = DataCollectorV0(env, step_preprocessor=AdroitStepPreProcessor, record_infos=True, max_steps_buffer=200000)

with h5py.File('/home/rodrigo/Downloads/door-cloned-v1.hdf5', 'r') as f:
    qposes = f['infos']['qpos'][:]
    qvels = f['infos']['qvel'][:]
    actions = f['actions'][:]
    observations = f['observations'][:]
    door_poses = f['infos']['door_body_pos'][:]
    timeouts = f['timeouts'][:]
    rewards = f['rewards'][:]
    
reset = True
last_episode_step = 0
for i, (timeout, observation, action, door_pos, reward, qpos, qvel) in enumerate(zip(timeouts, observations, actions, door_poses, rewards, qposes, qvels)):
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
