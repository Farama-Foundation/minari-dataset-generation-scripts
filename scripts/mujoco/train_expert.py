import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import SAC
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback



# config = {
#     "policy_type": "MlpPolicy",
#     "total_timesteps": 1_000_000,
#     "env_id": "GoalReachAnt-v0",
#     "goal_threshold": 0.2,
#     "forward_reward_scale": 10.0,
#     "n_envs": 16,
#     "max_episode_steps": 1000,
#     "checkpoint_dir": checkpoint_dir,
#     "checkpoint_filename": checkpoint_filename,
#     "model_kwargs": {
#         "seed": 42,
#         "verbose": 1,
#         "batch_size": 256,
#         "buffer_size": 1000000,
#         "ent_coef": "auto_0.1",
#         "gamma": 0.991,
#         "gradient_steps": 1,
#         "learning_rate": 0.0003,
#         "learning_starts": 10000,
#         "policy_kwargs": {"net_arch": [256] * 4, "activation_fn": nn.ReLU},
#         "tau": 0.03,
#         "train_freq": 1,
#         "use_sde": False,
#         "action_noise": NormalActionNoise(mean=[0.0], sigma=[0.3]),
#     },
# }

def make_env(env_id: str, run_name:str):
    """Wrapper to create the appropriate environment."""
    env = gym.make(
        f"{env_id}-v5",
        render_mode="rgb_array",
    )
    env = RecordVideo(env, f"videos/{run_name}")
    env = Monitor(env)
    return env


if __name__ == "__main__":
    for env_id in ["HalfCheetah", "Ant", "Hopper", "Walker2d"]:
        for seed in range(5):
            run_name = f"{env_id}-SAC-{seed}"
            n_envs = 10
            # Set up WandB run
            run = wandb.init(
                project="Minari",
                name=run_name,
                # config=config,
                sync_tensorboard=True,
                monitor_gym=True,
                save_code=True,
            )

            env = make_vec_env(make_env, n_envs=n_envs, env_kwargs={"env_id": env_id, "run_name": run_name})

            model = SAC("MlpPolicy", env, tensorboard_log=f"runs/{run.id}", learning_rate=0.00073, gamma=0.98, tau=0.02, train_freq=8, gradient_steps=8, learning_starts=10000, use_sde= True, policy_kwargs= dict(log_std_init=-3, net_arch=[400, 300]), seed=seed, buffer_size=300000)
            wandb_callback = WandbCallback(
                gradient_save_freq=100, model_save_path=f"models/{run.id}"
            )
            # Save a checkpoint every 100000 steps
            checkpoint_callback = CheckpointCallback(
            save_freq=int(100000/n_envs),
            save_path=f"logs/{env_id}/",
            name_prefix="rl_model",
            save_replay_buffer=True,
            )

            model.learn(total_timesteps=3000000, callback=[wandb_callback, checkpoint_callback], log_interval=4)
            model.save(f"sac_{env_id}_expert")
            run.finish()
        