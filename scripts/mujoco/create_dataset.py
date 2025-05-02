import gymnasium as gym
import minari
import numpy as np
from huggingface_sb3 import load_from_hub
from minari import StepDataCallback
from sb3_contrib import ARS, TQC, TRPO
from stable_baselines3 import PPO, SAC, TD3
from tqdm import tqdm

from make_env import make_env

"""
ENV_IDS: A list of tuples specifying the environments and parameters for dataset generation.
Each tuple contains:
    - Environment ID (str): The Gymnasium environment ID (e.g., "HalfCheetah").
    - Proficiencies (tuple[str]): A tuple of proficiency levels (e.g., ("medium", "expert")).
    - Number of steps (int): The total number of steps to collect for the dataset.
    - Algorithm name (str): The name of the algorithm used to train the policy (e.g., "SAC", "TQC").
    - (Optional) Observation size (int): The size of the observation space if excluded elements are added via AddExcludedObservationElements callback.
"""
ENV_IDS = [
    ("InvertedPendulum", ("medium", "expert"), 100_000, "SAC"),
    ("InvertedDoublePendulum", ("medium", "expert"), 100_000, "SAC"),
    ("Reacher", ("medium", "expert"), 500_000, "SAC"),
    ("Pusher", ("medium", "expert"), 500_000, "SAC"),
    ("HalfCheetah", ("simple", "medium", "expert"), 1_000_000, "TQC"),
    ("Hopper", ("simple", "medium", "expert"), 1_000_000, "SAC"),
    ("Walker2d", ("simple", "medium", "expert"), 1_000_000, "SAC"),
    ("Swimmer", ("medium", "expert"), 1_000_000, "PPO"),
    ("Ant", ("simple", "medium"), 1_000_000, "SAC"),
    ("Humanoid", ("simple", "medium", "expert"), 1_000_000, "TQC"),
    ("HumanoidStandup", ("simple", "medium", "expert"), 1_000_000, "SAC", 348),
]

DATASET_VERSION = "v0"


class AddExcludedObservationElements(StepDataCallback):
    """Add Excluded observation elements like cfrc_ext to the observation space."""

    def __call__(self, env, **kwargs):
        step_data = super().__call__(env, **kwargs)
        if getattr(env.unwrapped, "_include_cinert_in_observation", None) is False:
            step_data["observation"] = np.concatenate([step_data["observation"], env.unwrapped.data.cinert[1:].flat.copy()])
        if getattr(env.unwrapped, "_include_cvel_in_observation", None) is False:
            step_data["observation"] = np.concatenate([step_data["observation"], env.unwrapped.data.cvel[1:].flat.copy()])
        if getattr(env.unwrapped, "_include_qfrc_actuator_in_observation", None) is False:
            step_data["observation"] = np.concatenate(
                [step_data["observation"], env.unwrapped.data.qfrc_actuator[6:].flat.copy()]
            )
        if getattr(env.unwrapped, "_include_cfrc_ext_in_observation", None) is False:
            step_data["observation"] = np.concatenate([step_data["observation"], env.unwrapped.data.cfrc_ext[1:].flat.copy()])

        return step_data


def create_dataset_from_policy(env_id, proficiency, collector_env, policy, n_steps: int, algorithm_name):
    truncated = True
    terminated = True
    seed = 123
    for step in tqdm(range(n_steps)):
        if terminated or truncated:
            obs, _ = env.reset(seed=seed)
            seed += 1
            if (n_steps - step) < collector_env.spec.max_episode_steps:  # trim trailing non-full episodes
                break

        action = policy(obs)
        obs, _, terminated, truncated, _ = env.step(action)

    return collector_env.create_dataset(
        dataset_id=f"mujoco/{env_id.lower()}/{proficiency}-{DATASET_VERSION}",
        algorithm_name=f"SB3/{algorithm_name}",
        code_permalink="https://github.com/Farama-Foundation/minari-dataset-generation-scripts",
        author="Kallinteris Andreas",
        author_email="kallinteris@protonmail.com",
        requirements=["mujoco==3.2.3", "gymnasium>=1.0.0"],
        description=open(f"./descriptions/{env_id}-{proficiency}.md", "r").read(),
    )


def load_policy(env_id: str, algo: str, proficiency: str):
    model_checkpoint = load_from_hub(
        repo_id=f"farama-minari/{env_id}-v5-{algo.upper()}-{proficiency}",
        filename=f"{env_id.lower()}-v5-{algo.upper()}-{proficiency}.zip",
    )

    match algo:
        case "SAC":
            policy = SAC.load(model_checkpoint)
        case "TD3":
            policy = TD3.load(model_checkpoint)
        case "PPO":
            policy = PPO.load(model_checkpoint)
        case "TQC":
            policy = TQC.load(model_checkpoint)

    return policy


if __name__ == "__main__":
    for env_run_spec in ENV_IDS:
        # unpack dataset spec
        env_id = env_run_spec[0]
        proficiencies = env_run_spec[1]
        n_steps = env_run_spec[2]
        algo = env_run_spec[3]
        add_excluded_obs = len(env_run_spec) == 5
        observation_size = env_run_spec[4] if add_excluded_obs else None

        # make datasets
        for proficiency in proficiencies:
            print(f"\nCREATING {proficiency.upper()} DATASET FOR {env_id}")
            env = make_env(env_id, render_mode=None, use_monitor_wrapper=False)
            env.spec.kwargs = {}  # overwrite the spec for the dataset since we include the observations with the callback
            if add_excluded_obs:
                env = minari.DataCollector(
                    env,
                    step_data_callback=AddExcludedObservationElements,
                    observation_space=gym.spaces.Box(-np.inf, np.inf, (observation_size,), np.float64),
                    record_infos=False,
                )
            else:
                env = minari.DataCollector(env, record_infos=False)  # TODO record_info?

            policy = load_policy(env_id, algo, proficiency)
            dataset = create_dataset_from_policy(
                env_id,
                proficiency,
                env,
                lambda x: policy.predict(x)[0],
                n_steps,
                algo,
            )
