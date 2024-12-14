from collections import defaultdict
import minari
from tqdm import tqdm
from expert_bot import Bot
import minigrid
from minigrid.wrappers import FullyObsWrapper
import gymnasium as gym
from gymnasium.spaces.text import alphanumeric
from minari import DataCollector
import json

ENV_NAMES = [env_id for env_id in gym.registry if env_id.startswith("BabyAI")]

EPISODE_NUM = 1000
EXCLUDE_ENV = {
    "BabyAI-GoToImpUnlock-v0",
    "BabyAI-PickupDist-v0",
    "BabyAI-PickupDistDebug-v0",
    "BabyAI-PutNextS5N2Carrying-v0",
    "BabyAI-PutNextS6N3Carrying-v0",
    "BabyAI-PutNextS7N4Carrying-v0",
    "BabyAI-Unlock-v0",
    "BabyAI-KeyInBox-v0",
    "BabyAI-UnlockToUnlock-v0",
    "BabyAI-SynthS5R2-v0",
    "BabyAI-MiniBossLevel-v0"
}


DESCRIPTION = """The dataset was generated using the expert bot from the BabyAI original repository and adapted to the latest version of the environment.
The bot is a hard-coded planner, which solves all the tasks optimally."""

for env_id in tqdm(ENV_NAMES):
    for type_ in {"optimal", "optimal-fullobs"}:
        assert env_id.endswith("-v0")
        env_name = env_id[:-len("-v0")]
        dataset_id = f"minigrid/{env_name}/{type_}-v0"
        if dataset_id in minari.list_local_datasets():
            print(f"Dataset {dataset_id} already exists. Skipping...")
            continue
        if env_id in EXCLUDE_ENV:
            print(f"Skipping {env_id} as bot doesn't work on it...")
            continue
        
        print("Generating dataset for", env_id)
        env = gym.make(env_id)
        if type_ == "optimal-fullobs":
            env = FullyObsWrapper(env)
        observation_space = env.observation_space
        observation_space["mission"] = gym.spaces.Text(
            max_length=256,  # arbitrary number
            charset=str(alphanumeric) + ' '
        )
        print("Obs shape:", observation_space['image'].shape)


        env = DataCollector(
            env,
            observation_space=observation_space,
        )

        for _ in range(EPISODE_NUM):
            obs, _ = env.reset()
            bot = Bot(env.unwrapped)
            done = False
            while not done:
                action = bot.replan()
                _, _, ter, tru, _ = env.step(action)
                done = ter or tru
        
        ds = env.create_dataset(
            dataset_id=dataset_id,
            author="Omar G. Younis",
            author_email="omar@farama.org",
            algorithm_name="BabyAI expert bot",
            description=DESCRIPTION,
            code_permalink="https://github.com/Farama-Foundation/minari-dataset-generation-scripts",
            requirements=["minigrid"]
        )

        # fix minigrid not recording args
        json_path = f"{ds.storage.data_path}/metadata.json"
        with open(json_path, "r") as f:
            metadata = json.load(f)
            env_spec = metadata["env_spec"]
            env_spec = env_spec.replace('"kwargs": null', '"kwargs": {}')
            metadata["env_spec"] = env_spec
        with open(json_path, "w") as f:
            json.dump(metadata, f)

        env.close()
