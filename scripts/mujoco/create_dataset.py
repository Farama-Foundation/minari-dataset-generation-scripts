import gymnasium as gym
import minari
import numpy as np
from huggingface_sb3 import load_from_hub
from huggingface_hub.utils import EntryNotFoundError
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
    ("InvertedPendulum", ("expert", "random", "medium"), 100_000, "SAC"),
    ("InvertedDoublePendulum", ("expert", "random", "medium"), 100_000, "SAC"),
    ("Reacher", ("expert", "random", "medium"), 500_000, "SAC"),
    ("Pusher", ("expert", "random", "medium"), 500_000, "SAC"),
    ("HalfCheetah", ("expert", "random", "simple", "medium"), 1_000_000, "TQC"),
    ("Hopper", ("expert", "random", "simple", "medium"), 1_000_000, "SAC"),
    ("Walker2d", ("expert", "random", "simple", "medium"), 1_000_000, "SAC"),
    ("Swimmer", ("expert", "random", "medium"), 1_000_000, "PPO"),
    ("Ant", ("expert", "random", "simple", "medium"), 1_000_000, "SAC"),
    ("Humanoid", ("expert", "random", "simple", "medium"), 1_000_000, "TQC"),
    ("HumanoidStandup", ("expert", "random", "simple", "medium"), 1_000_000, "SAC", 348),
]

DATASET_VERSION = "v2"


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


def create_dataset_from_policy(
        env_id,
        proficiency,
        collector_env,
        policy,
        n_steps: int,
        algorithm_name,
        ref_min_score,
        ref_max_score
):
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

    is_expert = (proficiency == "expert")
    dataset = collector_env.create_dataset(
        dataset_id=f"mujoco/{env_id.lower()}/{proficiency}-{DATASET_VERSION}",
        algorithm_name=f"sb3/{algorithm_name}",
        code_permalink="https://github.com/farama-foundation/minari-dataset-generation-scripts",
        author="kallinteris andreas",
        author_email="kallinteris@protonmail.com",
        requirements=["mujoco==3.2.3", "gymnasium>=1.0.0"],
        description=open(f"./descriptions/{env_id}-{proficiency}.md", "r").read(),
        ref_min_score=None if is_expert else ref_min_score,
        ref_max_score=None if is_expert else ref_max_score,
        expert_policy=policy if is_expert else None,
    )
    ref_min_score = dataset.storage.metadata["ref_min_score"]
    ref_max_score = dataset.storage.metadata["ref_max_score"]

    return dataset, ref_min_score, ref_max_score


def load_policy(env_id: str, algo: str, proficiency: str):
    if proficiency == "random":
        env = make_env(env_id)
        return lambda _: env.action_space.sample()

    repo_id = f"farama-minari/{env_id}-v5-{algo.upper()}-{proficiency}"
    filename_upper = f"{env_id.lower()}-v5-{algo.upper()}-{proficiency}.zip"
    filename_lower = f"{env_id.lower()}-v5-{algo.lower()}-{proficiency}.zip"

    # Some models use uppercase convention, some lowercase
    try:
        model_checkpoint = load_from_hub(repo_id, filename_upper)
    except EntryNotFoundError:
        model_checkpoint = load_from_hub(repo_id, filename_lower)

    print("LOADING", filename_upper)

    match algo:
        case "SAC":
            policy = SAC.load(model_checkpoint)
        case "TD3":
            policy = TD3.load(model_checkpoint)
        case "PPO":
            policy = PPO.load(model_checkpoint)
        case "TQC":
            policy = TQC.load(model_checkpoint)

    return lambda x: policy.predict(x)[0]


if __name__ == "__main__":
    for env_run_spec in ENV_IDS:
        # unpack dataset spec
        env_id = env_run_spec[0]
        proficiencies = env_run_spec[1]
        n_steps = env_run_spec[2]
        algo = env_run_spec[3]
        add_excluded_obs = len(env_run_spec) == 5
        observation_size = env_run_spec[4] if add_excluded_obs else None

        # Populated by expert dataset runs
        ref_min_score = None
        ref_max_score = None

        # make datasets
        for proficiency in proficiencies:
            if env_id == "HumanoidStandup":
                continue
            if f"mujoco/{env_id.lower()}/{proficiency}-{DATASET_VERSION}" in minari.list_local_datasets():
                print(f"\nSkipping {proficiency.upper()} DATASET FOR {env_id}")
                continue
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
            dataset, ref_min_score, ref_max_score = create_dataset_from_policy(
                env_id,
                proficiency,
                env,
                policy,
                n_steps,
                algo,
                ref_min_score,
                ref_max_score,
            )
