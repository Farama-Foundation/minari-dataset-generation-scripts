from huggingface_sb3 import package_to_hub, push_to_hub
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.buffers import ReplayBuffer

env_id = "HalfCheetah"
# # Create the evaluation env
eval_env = make_vec_env("HalfCheetah-v5", n_envs=1)
model = SAC.load(f"logs/{env_id}/rl_model_200000_steps.zip", 
                        custom_objects={"replay_buffer_class": ReplayBuffer, 
                                        "replay_buffer_kwargs": {},
                                        "lr_schedule": lambda _: 0.0
                                        }
                                        )
push_to_hub(
    repo_id=f"farama-minari/{env_id}-v5-SAC-medium",
    filename=f"rl_model_replay_buffer_200000_steps.pkl",
    commit_message="medium replay buffer",
)

package_to_hub(model=model, 
            model_name=f"{env_id.lower()}-v5-sac-medium",
            model_architecture="SAC",
            env_id=f"{env_id}-v5",
            eval_env=eval_env,
            repo_id=f"farama-minari/{env_id}-v5-SAC-medium",
            commit_message="model")


