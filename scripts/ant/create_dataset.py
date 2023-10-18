__authors__ = ["Kallinteris Andreas"]
import gymnasium as gym
import minari
assert minari.__version__ == "0.4.1"
import numpy as np
from minari import DataCollectorV0, StepDataCallback
from stable_baselines3 import A2C, PPO, SAC, TD3
import stable_baselines3
assert stable_baselines3.__version__ == "2.0.0a5"



SEED = 12345
NUM_STEPS = int(10e6)
POLICY_NOISE = 0.1

class AddExcludedObservationElements(StepDataCallback):
    """Add Excluded observation elements like cfrc_ext to the observation space."""
    def __call__(self, env, **kwargs):
        step_data = super().__call__(env, **kwargs)
        # if getattr(env, "_include_cinert_in_observation", None) is False:
        # if getattr(env, "_include_cvel_in_observation ", None) is False:
        # if getattr(env, "_include_qfrc_actuator_in_observation ", None) is False:
        if env.unwrapped._include_cfrc_ext_in_observation is False:
            step_data["observations"] = np.concatenate([step_data["observations"], env.unwrapped.contact_forces[1:].flat.copy()])


        return step_data

dataset_name = "ant-v5-expert-v0"
dataset = None

# Check if dataset already exist
assert dataset_name not in minari.list_local_datasets()

# Create Environment
env = gym.make("Ant-v5", include_cfrc_ext_in_observation=False, max_episode_steps=1e9)
# add callback to add cfrc_ext to the obs space
collector_env = DataCollectorV0(env, step_data_callback=AddExcludedObservationElements, record_infos=True)
# we do not observe `cfrc_ext` with the agent, but we keep it in the dataset observations
collector_env.dataset_observation_space = gym.spaces.Box(-np.inf, np.inf, (105,), np.float64)
obs, _ = collector_env.reset(seed=SEED)

# load policy model
# model is from: https://github.com/Kallinteris-Andreas/gymnasium-mujuco-v5-envs-validation/tree/main/results/ant_v5_without_ctn_SAC/run_9
model = SAC.load(
    path="./results/ant_v5_without_ctn_SAC/run_9/best_model.zip",
    env=env,
    device="cpu",
)


for n_step in range(NUM_STEPS):
    action, _ = model.predict(obs, deterministic=True)
    # Add some noise to each step action
    action += np.random.randn(*action.shape) * POLICY_NOISE
    action = action.clip(-1, 1)

    obs, rew, terminated, truncated, info = collector_env.step(action)
    # Checkpoint
    if (n_step + 1) % 200e3 == 0:
        print(f"STEPS RECORDED: {n_step}")
        #if dataset is None:
        #dataset.update_dataset_from_collector_env(collector_env)

    if terminated or truncated:
        env.reset()

# dataset.update_dataset_from_collector_env(collector_env)
dataset = minari.create_dataset_from_collector_env(
    collector_env=collector_env,
    dataset_id=dataset_name,
    algorithm_name="SB3/SAC",
    code_permalink="https://github.com/Kallinteris-Andreas/gymnasium-mujuco-v5-envs-validation/blob/main/create_dataset.py",
    author="Kallinteris Andreas",
    author_email="kallinteris@protonmail.com",
)
collector_env.save_to_disk("test.hdf5")
