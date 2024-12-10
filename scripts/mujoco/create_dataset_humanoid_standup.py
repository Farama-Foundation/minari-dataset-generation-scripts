import gymnasium as gym
import minari
import numpy as np
from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO, SAC, TD3
from tqdm import tqdm

ENV_IDS = [
    ("HumanoidStandup", ("simple", "medium", "expert"), 1_000_000, "SAC"),
]

class AddExcludedObservationElements(minari.StepDataCallback):
    """Add Excluded observation elements like `cfrc_ext` to the observation space."""

    def __call__(self, env, **kwargs):
        step_data = super().__call__(env, **kwargs)
        if getattr(env.unwrapped, "_include_cinert_in_observation", True) is False:
            step_data["observation"] = np.concatenate(
                [step_data["observation"], env.unwrapped.data.cinert[1:].flatten()]
            )
        if getattr(env.unwrapped, "_include_cvel_in_observation", True) is False:
            step_data["observation"] = np.concatenate(
                [step_data["observation"], env.unwrapped.data.cvel[1:].flatten()]
            )
        if getattr(env.unwrapped, "_include_qfrc_actuator_in_observation", True) is False:
            step_data["observation"] = np.concatenate(
                [step_data["observation"], env.unwrapped.data.qfrc_actuator[6:].flatten()]
            )
        if getattr(env.unwrapped, "_include_cfrc_ext_in_observation", True) is False:
            step_data["observation"] = np.concatenate(
                [step_data["observation"], env.unwrapped.data.cfrc_ext[1:].flat.copy()]
            )

        return step_data


def create_dataset_from_policy(dataset_id, collector_env, policy, n_steps: int, algorithm_name):
    truncated = True
    terminated = True
    seed = 123
    for step in tqdm(range(n_steps)):
        if terminated or truncated:
            obs, _ = env.reset(seed=seed)
            seed += 1
            if (n_steps - step) < collector_env.spec.max_episode_steps :  # trim trailing non-full episodes
                break

        action = policy(obs)
        obs, _, terminated, truncated, _ = env.step(action)

    return collector_env.create_dataset(
        dataset_id=f"mujoco/{dataset_id}",
        algorithm_name="SB3/{algorithm_name}",
        code_permalink="https://github.com/Farama-Foundation/minari-dataset-generation-scripts",
        author="Kallinteris Andreas",
        author_email="kallinteris@protonmail.com",
        requirements=[
            "mujoco==3.2.3",
        ],
    )


def load_policy(env_id: str, algo: str, proficiency: str):
    model_checkpoint = load_from_hub(
        repo_id=f"farama-minari/{env_id}-v5-{algo.upper()}-{proficiency}",
        filename=f"{env_id.lower()}-v5-{algo.lower()}-{proficiency}.zip",
    )

    match algo:
        case "SAC":
            policy = SAC.load(model_checkpoint)
        case "TD3":
            policy = TD3.load(model_checkpoint)
        case "PPO":
            policy = PPO.load(model_checkpoint)

    return policy


def make_env(env_id: str):
    """Wrapper to create the appropriate environment."""
    if env_id == "HumanoidStandup":
        env = gym.make(f"{env_id}-v5", include_cinert_in_observation=False, include_cvel_in_observation=False, include_qfrc_actuator_in_observation=False, include_cfrc_ext_in_observation=False)
    else:
        env = gym.make(f"{env_id}-v5", render_mode="rgb_array",)
    return env


if __name__ == "__main__":
    for env_run_spec in ENV_IDS:
        # unpack dataset spec
        env_id = env_run_spec[0]
        proficiencies = env_run_spec[1]
        n_steps = env_run_spec[2]
        algo = env_run_spec[3]


        # make datasets
        print(f"\nCREATING EXPERT DATASET FOR {env_id}")
        if "expert" in proficiencies:
            env = make_env(env_id)
            env.spec.kwargs = {}  # overwrite the spec for the dataset since we include the observations with the callback
            env = minari.DataCollector(env, step_data_callback=AddExcludedObservationElements, record_infos=False, observation_space=gym.spaces.Box(-np.inf, np.inf, (348,), np.float64))  # TODO record_info?

            expert_policy = load_policy(env_id, algo, "expert")
            expert_dataset = create_dataset_from_policy(
                f"{env_id.lower()}/expert-v0",
                env,
                lambda x: expert_policy.predict(x)[0],
                n_steps,
                algo,
            )

        print(f"\nCREATING MEDIUM DATASET FOR {env_id}")
        if "medium" in proficiencies:
            env = make_env(env_id)
            env.spec.kwargs = {}  # overwrite the spec for the dataset since we include the observations with the callback
            env = minari.DataCollector(env, step_data_callback=AddExcludedObservationElements, record_infos=False, observation_space=gym.spaces.Box(-np.inf, np.inf, (348,), np.float64))  # TODO record_info?

            medium_policy = load_policy(env_id, algo, "medium")
            medium_dataset = create_dataset_from_policy(
                f"{env_id.lower()}/medium-v0",
                env,
                lambda x: medium_policy.predict(x)[0],
                n_steps,
                algo,
            )

        print(f"\nCREATING SIMPLE DATASET FOR {env_id}")
        if "simple" in proficiencies:
            env = make_env(env_id)
            env.spec.kwargs = {}  # overwrite the spec for the dataset since we include the observations with the callback
            env = minari.DataCollector(env, step_data_callback=AddExcludedObservationElements, record_infos=False, observation_space=gym.spaces.Box(-np.inf, np.inf, (348,), np.float64))  # TODO record_info?

            simple_policy = load_policy(env_id, algo, "simple")
            simple_dataset = create_dataset_from_policy(
                f"{env_id.lower()}/simple-v0",
                env,
                lambda x: simple_policy.predict(x)[0],
                n_steps,
                algo,
            )
