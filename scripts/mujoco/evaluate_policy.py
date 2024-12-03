import gymnasium as gym
from huggingface_sb3 import load_from_hub
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

ENV_ID = "Ant"
ALGORITHM = "SAC"
# PROFICIENCY = "medium"
PROFICIENCY = "expert-fine-tuned"

eval_env = gym.make(f"{ENV_ID}-v5")
eval_env = gym.make(f"{ENV_ID}-v5", include_cfrc_ext_in_observation=False)

match ALGORITHM:
    case "SAC":
        expert_model_checkpoint = load_from_hub(
            repo_id=f"farama-minari/{ENV_ID}-v5-{ALGORITHM.upper()}-{PROFICIENCY}",
            filename=f"{ENV_ID.lower()}-v5-{ALGORITHM.lower()}-{PROFICIENCY}.zip",
        )
        model = SAC.load(expert_model_checkpoint)
    case "TD3":
        None
    case "PPO":
        None
print("MODEL LOADED")


mean_reward, std_reward = evaluate_policy(model, eval_env, render=False, n_eval_episodes=1000, deterministic=True, warn=False)
print(f"{ENV_ID}-v5/{PROFICIENCY}/{ALGORITHM}")
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
