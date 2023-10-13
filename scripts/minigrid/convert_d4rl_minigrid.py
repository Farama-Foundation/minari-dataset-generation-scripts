import argparse
import gym as gym_legacy
import gymnasium as gym
import minari
import d4rl # Import required to register environments, you may need to also import the submodule
import numpy as np


parser = argparse.ArgumentParser(description="Convert MiniGrid datasets")
parser.add_argument("--random", action='store_true', default=False, help="Whether to generate the random dataset")


def convert_dataaset(args):
    dataset_name = f"minigrid-fourrooms{'-random' if args.random else ''}-v0"
    data_env = gym_legacy.make(dataset_name)
    dataset = data_env.get_dataset()
    gymnasium_env = gym.make("MiniGrid-FourRooms-v0")

    observation_space = gym.spaces.Box(
        dataset["observations"].min(),
        dataset["observations"].max(),
        dataset["observations"][0].shape,
        dataset["observations"][0].dtype
    )

    buffer = []
    i = 0
    while i < len(dataset["observations"]):
        dones = np.logical_or(dataset["terminals"][i:], dataset["timeouts"][i:])
        next_ep_idx = i + np.argmax(dones) + 1
        observations = np.empty(
            (next_ep_idx - i + 1, *observation_space.shape),
            dtype=observation_space.dtype
        )
        observations[:-1] = dataset["observations"][i:next_ep_idx]
        observations[-1] = observations[-2]  # repeat last observation
        buffer.append(
            {
                "observations": observations,
                "actions": dataset["actions"][i:next_ep_idx],
                "rewards": dataset["rewards"][i:next_ep_idx],
                "terminations": dataset["terminals"][i:next_ep_idx],
                "truncations": dataset["timeouts"][i:next_ep_idx],
                "infos": {
                    "pos": dataset["infos/pos"][i:next_ep_idx],
                    "goal": dataset["infos/goal"][i:next_ep_idx],
                    "orientation": dataset["infos/orientation"][i:next_ep_idx],
                }
            }
        )

        i = next_ep_idx

    minari.create_dataset_from_buffers(
        dataset_id=dataset_name,                                            
        env=gymnasium_env,
        buffer=buffer,
        observation_space=observation_space,
        algorithm_name="RandomPolicy" if args.random else "ExpertPolicy",
        author="Omar G. Younis",
        author_email="omar.g.younis@gmail.com",
        minari_version=">=0.4.0",
        ref_max_score=data_env.ref_max_score,
        ref_min_score=data_env.ref_min_score,
    )

if __name__ == "__main__":
    args = parser.parse_args()
    convert_dataaset(args)
