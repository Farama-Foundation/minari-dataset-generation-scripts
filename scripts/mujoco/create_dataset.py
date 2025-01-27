import gymnasium as gym
import minari
from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO, SAC, TD3
from sb3_contrib import TQC, TRPO, ARS
from tqdm import tqdm

ENV_IDS = [
    ("InvertedPendulum", ("medium", "expert"), 1_000, "SAC"),
    ("InvertedDoublePendulum", ("medium", "expert"), 100_000, "SAC"),
    ("Reacher", ("medium", "expert"), 500_000, "SAC"),
    ("Pusher", ("medium", "expert"), 500_000, "SAC"),
    ("HalfCheetah", ("simple", "medium", "expert"), 1_000_000, "TQC"),
    ("Hopper", ("simple", "medium", "expert"), 1_000_000, "SAC"),
    ("Walker2d", ("simple", "medium", "expert"), 1_000_000, "SAC"),
    #("Swimmer", ("expert",), 1_000_000, "SAC"),
    ("Ant", ("simple", "medium"), 1_000_000, "SAC"),
    #("Humanoid", ("simple", "medium", "expert"), 1_000_000, "SAC"),
    # ("HumanoidStandup", ("simple", "medium", "expert"), 1_000_000, "SAC"),
]


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
        dataset_id=f"mujoco/{env_id.lower()}/{proficiency}-v0",
        algorithm_name="SB3/{algorithm_name}",
        code_permalink="https://github.com/Farama-Foundation/minari-dataset-generation-scripts",
        author="Kallinteris Andreas",
        author_email="kallinteris@protonmail.com",
        requirements=["mujoco==3.2.3", "gymnasium>=1.0.0"],
        description=open(f"./descriptions/{env_id}-{proficiency}.md", 'r').read()
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

        # make datasets
        for proficiency in proficiencies:
            print(f"\nCREATING {proficiency.upper()} DATASET FOR {env_id}")
            env = gym.make(f"{env_id}-v5")
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
