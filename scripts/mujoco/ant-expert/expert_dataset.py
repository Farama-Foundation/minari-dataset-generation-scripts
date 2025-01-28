__credits__ = ["Kallinteris Andreas"]

import gymnasium as gym
import minari
import numpy as np
from minari import DataCollector, StepDataCallback
from stable_baselines3 import SAC

SEED = 12345
NUM_STEPS = int(2e6)


class AddExcludedObservationElements(StepDataCallback):
    """Add Excluded observation elements like cfrc_ext to the observation space."""

    def __call__(self, env, **kwargs):
        step_data = super().__call__(env, **kwargs)
        # if getattr(env, "_include_cinert_in_observation", None) is False:
        # if getattr(env, "_include_cvel_in_observation ", None) is False:
        # if getattr(env, "_include_qfrc_actuator_in_observation ", None) is False:
        if env.unwrapped._include_cfrc_ext_in_observation is False:
            step_data["observation"] = np.concatenate(
                [step_data["observation"], env.unwrapped.contact_forces[1:].flat.copy()]
            )

        return step_data

DATASET_NAME = "mujoco/ant/expert-v0"
assert DATASET_NAME not in minari.list_local_datasets()

env = gym.make("Ant-v5", include_cfrc_ext_in_observation=False, max_episode_steps=1000)
env.spec.kwargs = {}  # overwrite the spec for the dataset since we include the observations with the callback

collector_env = DataCollector(
    env,
    step_data_callback=AddExcludedObservationElements,
    record_infos=False,
    observation_space=gym.spaces.Box(-np.inf, np.inf, (105,), np.float64),
)
obs, _ = collector_env.reset(seed=SEED)

# model is from: https://github.com/Kallinteris-Andreas/gymnasium-mujuco-v5-envs-validation/tree/main/results/ant_v5_without_ctn_SAC/run_9
model = SAC.load(
    path="./best_model.zip",
    env=env,
    device="cpu",
)

for n_step in range(NUM_STEPS):
    action, _ = model.predict(obs, deterministic=True)

    obs, rew, terminated, truncated, info = collector_env.step(action)
    if (n_step + 1) % 200e3 == 0:
        print(f"STEPS RECORDED: {n_step}")

    if terminated or truncated:
        env.reset()


dataset = collector_env.create_dataset(
    dataset_id=DATASET_NAME,
    algorithm_name="SB3/SAC",
    code_permalink="https://github.com/Farama-Foundation/minari-dataset-generation-scripts/tree/main/scripts/MuJoCo/Ant/expert_dataset.py",
    author="Kallinteris Andreas",
    author_email="kallinteris@protonmail.com",
    description="Ant expert fine tuned policy, model training at https://github.com/Kallinteris-Andreas/gymnasium-mujuco-v5-envs-validation .",
    requirements=["mujoco==3.2.3",],
)
