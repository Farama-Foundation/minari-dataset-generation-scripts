import copy
# TODO make eval deterministic

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO, SAC, TD3
from sb3_contrib import TQC, TRPO, ARS
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str)
parser.add_argument("--env_id", type=str)
parser.add_argument("--timesteps", type=int)
parser.add_argument("--proficiency", type=str)
args = parser.parse_args()

# NOTE: different timesteps are used for different environments/proficiencies, check `config.yaml` for indivigual values

ALGORITHM = args.algo.lower()
PROFICIENCY = args.proficiency
assert PROFICIENCY in ["simple", "medium", "expert"]
TIMESTEPS = args.timesteps
EVAL_ENVS = 50
EVAL_FREQ = 1000
RUNS = 1

# ENV_LIST = ["HalfCheetah", "Ant", "Hopper", "Walker2d", "InvertedPendulum", "InvertedDoublePendulum", "Reacher", "Pusher", "Swimmer", "Humanoid", "HumanoidStandup"]
ENV_LIST = [args.env_id,]

print(f"Training {ENV_LIST}/{PROFICIENCY} - {ALGORITHM}/{TIMESTEPS}")


def gen_gamma(env_id: str) -> float:
    if env_id == "Swimmer":
        return 1.0
    return 0.99


def make_env(env_id: str, run_name: str) -> gym.Env:
    """Wrapper to create the appropriate environment."""
    if env_id == "HumanoidStandup":
        env = gym.make(
            f"{env_id}-v5",
            render_mode="rgb_array",
            include_cinert_in_observation=False,
            include_cvel_in_observation=False,
            include_qfrc_actuator_in_observation=False,
            include_cfrc_ext_in_observation=False,
        )
    else:
        env = gym.make(
            f"{env_id}-v5",
            render_mode="rgb_array",
        )
    env = RecordVideo(env, f"videos/{run_name}")
    env = Monitor(env)
    return env


def initialize_model(algo_name):
    if ALGORITHM == "sac":
        model = SAC(
            "MlpPolicy",
            env,
            tensorboard_log=f"runs/{0}",
            learning_starts=10_000,
            gamma=gen_gamma(env_id),
            use_sde=False,
            seed=seed,
        )
    elif ALGORITHM == "td3":
        model = TD3(
            "MlpPolicy",
            env,
            tensorboard_log=f"runs/{0}",
            learning_starts=10_000,
            gamma=gen_gamma(env_id),
            seed=seed,
        )
    elif ALGORITHM == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            tensorboard_log=f"runs/{0}",
            gamma=gen_gamma(env_id),
            seed=seed,
            ent_coef=5e-6,
            device="cpu",
        )
    elif ALGORITHM == "tqc":
        model = TQC(
            "MlpPolicy",
            env,
            tensorboard_log=f"runs/{0}",
            gamma=gen_gamma(env_id),
            seed=seed,
            device="cpu",
        )
    elif ALGORITHM == "trpo":
        model = TRPO(
            "MlpPolicy",
            env,
            tensorboard_log=f"runs/{0}",
            gamma=gen_gamma(env_id),
            seed=seed,
            device="cpu",
        )
    elif ALGORITHM == "ars":
        model = ARS(
            "MlpPolicy",
            env,
            tensorboard_log=f"runs/{0}",
            delta_std=0.01,
            n_delta=1,
            n_top=1,
            seed=seed,
            device="cpu",
        )
    return model


if __name__ == "__main__":
    for env_id in ENV_LIST:
        for seed in range(RUNS):
            run_name = f"{env_id}-{ALGORITHM.upper()}-{seed}"
            eval_path = f"models/{env_id}-{ALGORITHM.upper()}-{PROFICIENCY}/run_{seed}/"
            n_envs = 5
            # Set up WandB run
            run = wandb.init(
                project="Minari",
                name=run_name,
                sync_tensorboard=True,
                monitor_gym=True,
                save_code=True,
            )

            env = make_vec_env(make_env, n_envs=n_envs, env_kwargs={"env_id": env_id, "run_name": run_name})
            eval_env = copy.deepcopy(env)

            model = initialize_model(ALGORITHM)
            model.set_logger(configure(eval_path, ["csv"]))

            wandb_callback = WandbCallback()
            checkpoint_callback = CheckpointCallback(  # Save a checkpoint every 100000 steps
                save_freq=int(100000 / n_envs),
                save_path=f"logs/{env_id}/",
                name_prefix=f"{PROFICIENCY}",
                save_replay_buffer=False,
            )
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=eval_path,
                log_path=eval_path,
                n_eval_episodes=EVAL_ENVS,
                eval_freq=EVAL_FREQ,
                deterministic=True,
                render=False,
                verbose=True,
            )

            model.learn(total_timesteps=TIMESTEPS, callback=[wandb_callback, checkpoint_callback, eval_callback], log_interval=2)
            model.save(f"{eval_path}/latest_model")
            run.finish()

        print("FINISHED LEARNING")

# models can be uploaded to HF using the `upload_model.py` script
