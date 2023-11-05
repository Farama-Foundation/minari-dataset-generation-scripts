from stable_baselines3.common.save_util import load_from_pkl
from train import InfoReplayBuffer
from stable_baselines3 import SAC
import minari
import gymnasium as gym
from minari.utils import RandomPolicy
from tqdm import tqdm

def create_dataset_from_policy(dataset_id, collector_env, policy, expert_policy, n_steps):
    truncated = True
    terminated = True
    seed = 123
    for _ in tqdm(range(n_steps)):
        if terminated or truncated:
            obs, _ = env.reset(seed=seed)
            seed +=1
        action = policy(obs)
        obs, _, terminated, truncated, _ = env.step(action)

    return minari.create_dataset_from_collector_env(
        collector_env=collector_env,
        dataset_id=dataset_id,
        expert_policy=expert_policy,
        algorithm_name="SAC",
        code_permalink="https://github.com/rodrigodelazcano/d4rl-minari-dataset-generation",
        author="Rodrigo de Lazcano",
        author_email="rperezvicente@farama.org",
    )

for env_id in ["HalfCheetah", 
            #    "Ant", 
            #    "Hopper", 
            #    "Walker2d"
               ]:
    
    env = gym.make(f"{env_id}-v5")
    env = minari.DataCollectorV0(env)

    print(f"\nCREATING EXPERT DATASET FOR {env_id}")
    expert_policy = SAC.load("sac_HalfCheetah_expert")
    expert_dataset = create_dataset_from_policy(f"{env_id.lower()}-expert-v0", env, lambda x: expert_policy.predict(x)[0], lambda x: expert_policy.predict(x)[0], n_steps=int(1e4))

    print(f"\nCREATING MEDIUM DATASET FOR {env_id}")
    medium_trained_policy = SAC.load("rl_model_300000_steps")
    medium_dataset = create_dataset_from_policy(f"{env_id.lower()}-medium-v0", env, lambda x: medium_trained_policy.predict(x)[0], lambda x: expert_policy.predict(x)[0], n_steps=int(1e4))

    print(f"\nCREATING EXPERT-MEDIUM DATASET FOR {env_id}\n")
    minari.combine_datasets([expert_dataset, medium_dataset], f"{env_id.lower()}-medium-expert-v0")

    print(f"\nCREATING RANDOM DATASET FOR {env_id}")
    random_policy = RandomPolicy(env)
    create_dataset_from_policy(f"{env_id.lower()}-random-v0", env, random_policy, lambda x: expert_policy.predict(x)[0], n_steps=int(1e4))

    print(f"\nCREATING MEDIUM-REPLAY DATASET FOR {env_id}")
    replay_buffer: InfoReplayBuffer = load_from_pkl("rl_model_replay_buffer_100000_steps")
    minari_buffer = replay_buffer.convert_to_minari_buffer()
    minari.create_dataset_from_buffers(
        f"{env_id.lower()}-medium-replay-v0", 
        f"{env_id}-v5", 
        minari_buffer,
        expert_policy=lambda x: expert_policy.predict(x)[0],
        algorithm_name="SAC",
        code_permalink="https://github.com/rodrigodelazcano/d4rl-minari-dataset-generation",
        author="Rodrigo de Lazcano",
        author_email="rperezvicente@farama.org"
        )
        
