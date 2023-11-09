import argparse
import gymnasium as gym
import minigrid
import minari
from minari import DataCollectorV0
from gymnasium.spaces.text import alphanumeric
import numpy as np
np.random.seed(42)

from policies import RandomPolicy, ExpertPolicy

parser = argparse.ArgumentParser(description='Generate MiniGrid FourRooms datasets')
parser.add_argument('--random', help='Whether to user random policy', action='store_true')

def generate_dataset(random):
    env = gym.make("MiniGrid-FourRooms-v0")
    max_len = len("reach the goal")
    obs_space = gym.spaces.Dict({
        "direction": env.observation_space["direction"],
        "image": env.observation_space["image"],
        "mission": gym.spaces.Text(
            max_length=max_len,
            charset=str(alphanumeric) + ' '
        )
    })

    env = DataCollectorV0(env, record_infos=True, max_buffer_steps=1_000_000, observation_space=obs_space)

    step_lower_bound = 10_000
    step_id = 0
    while step_id < step_lower_bound:
        print("STEP", step_id)
        env.reset(seed=np.random.randint(2**31 - 1))
        policy = RandomPolicy(env) if random else ExpertPolicy(env)
        done = False
        while not done:
            step_id += 1
            action = policy.get_action()
            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated


    minari.create_dataset_from_collector_env(
        dataset_id=f"minigrid-fourrooms{'-random' if random else ''}-v0", 
        collector_env=env,
        algorithm_name="RandomPolicy" if random else "ExpertPolicy",
        author="Omar G. Younis",
        author_email="omar.younis98@gmail.com",
        minari_version=">=0.4.0",
        code_permalink="https://github.com/rodrigodelazcano/d4rl-minari-dataset-generation",
    )


if __name__ == "__main__":
    args = parser.parse_args()
    generate_dataset(random=args.random)