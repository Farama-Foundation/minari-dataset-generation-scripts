"""Upload trained model to Hugging Face Hub."""


from huggingface_sb3 import package_to_hub
from stable_baselines3 import PPO, SAC, TD3
from sb3_contrib import TQC, TRPO, ARS
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.env_util import make_vec_env
import argparse


HF_REPO = "farama-minari"
parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str)
parser.add_argument("--proficiency", type=str)
parser.add_argument("--environment", type=str)
args = parser.parse_args()

ALGORITHM = args.algo
PROFICIENCY = args.proficiency
ENV_LIST = [args.environment,]
seed = 0

EVAL_ENVS = 1000

if __name__ == "__main__":
    for env_id in ENV_LIST:
        if env_id == "HumandoidStandup":
            eval_env = make_vec_env(
                f"{env_id}-v5",
                n_envs=1,
                env_kwargs={
                    "include_cinert_in_observation": False,
                    "include_cvel_in_observation": False,
                    "include_qfrc_actuator_in_observation": False,
                    "include_cfrc_ext_in_observation": False,
                },
            )
        else:
            eval_env = make_vec_env(f"{env_id}-v5", n_envs=1)

        if ALGORITHM.lower() == "sac":
            model = SAC.load(
                f"models/{env_id}-{ALGORITHM.upper()}-{PROFICIENCY}/run_{seed}/best_model.zip",
                #env=VecEnv,
            )
        elif ALGORITHM.lower() == "td3":
            model = TD3.load(
                f"models/{env_id}-{ALGORITHM.upper()}-{PROFICIENCY}/run_{seed}/best_model.zip",
            )
        elif ALGORITHM.lower() == "ppo":
            model = PPO.load(
                f"models/{env_id}-{ALGORITHM.upper()}-{PROFICIENCY}/run_{seed}/best_model.zip",
            )
        elif ALGORITHM.lower() == "tqc":
            model = TQC.load(
                f"models/{env_id}-{ALGORITHM.upper()}-{PROFICIENCY}/run_{seed}/best_model.zip",
            )

        package_to_hub(
            model=model,
            model_name=f"{env_id.lower()}-v5-{ALGORITHM.upper()}-{PROFICIENCY}",
            model_architecture=ALGORITHM.upper(),
            env_id=f"{env_id}-v5",
            eval_env=eval_env,
            repo_id=f"{HF_REPO}/{env_id}-v5-{ALGORITHM.upper()}-{PROFICIENCY}",
            #repo_id=f"{HF_REPO}/TEST",
            commit_message="model",
            is_deterministic=True,
            n_eval_episodes=EVAL_ENVS,
        )

    print("\nFINISHED UPLOADING TO HF")
