import numpy as np
from typing import Any, Dict, List

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.buffers import ReplayBuffer

from huggingface_sb3 import package_to_hub


INFO_KEYS = {
    "HalfCheetah": 
                {
                    "x_position",
                    "x_velocity",
                    "reward_forward",
                    "reward_ctrl",
                },
    "Hopper": 
                {
                    "x_position",
                    "x_velocity",
                    "z_distance_from_origin",
                    "reward_forward",
                    "reward_ctrl",
                    "reward_survive",
                },
    "Ant": 
                {
                    "x_position",
                    "y_position",
                    "distance_from_origin",
                    "x_velocity",
                    "y_velocity",
                    "reward_forward",
                    "reward_ctrl",
                    "reward_survive",
                    "reward_contact",
                },
    "Walker2d":
                {
                    "x_position",
                    "x_velocity",
                    "z_distance_from_origin",
                    "reward_forward",
                    "reward_ctrl",
                    "reward_survive",
                },
}

class InfoReplayBuffer(ReplayBuffer):
    """Extend SB3 ReplayBuffer to store the info return.

    Similar to HerReplayBuffer.
    """
    def __init__(self, buffer_size, observation_space, action_space, info_keys: set, **kwargs):
        super().__init__(buffer_size, observation_space, action_space, **kwargs)
        # Add infos buffer
        # All of the Mujoco environments have the same following space for their info keys: spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
        self.infos = {
            key: np.zeros((self.buffer_size, self.n_envs), dtype=np.float64)
            for key in info_keys
        }

        self.ep_start = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.ep_length = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self._current_ep_start = np.zeros(self.n_envs, dtype=np.int64)

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray, infos: List[Dict[str, Any]]) -> None:

        # Update episode start
        self.ep_start[self.pos] = self._current_ep_start.copy()

        
        for key in self.infos.keys():
            self.infos[key][self.pos] = np.array([info.get(key, 0.0) for info in infos])
        
        super().add(obs, next_obs, action, reward, done, infos)

        # When episode ends, compute and store the episode length
        for env_idx in range(self.n_envs):
            if done[env_idx]:
                self._compute_episode_length(env_idx)
    
    def _compute_episode_length(self, env_idx: int) -> None:
        """
        Compute and store the episode length for environment with index env_idx

        :param env_idx: index of the environment for which the episode length should be computed
        """
        episode_start = self._current_ep_start[env_idx]
        episode_end = self.pos
        if episode_end < episode_start:
            # Occurs when the buffer becomes full, the storage resumes at the
            # beginning of the buffer. This can happen in the middle of an episode.
            episode_end += self.buffer_size
        episode_indices = np.arange(episode_start, episode_end) % self.buffer_size
        self.ep_length[episode_indices, env_idx] = episode_end - episode_start
        # Update the current episode start
        self._current_ep_start[env_idx] = self.pos
    
    def truncate_last_trajectory(self) -> None:
        """
        If called, we assume that the last trajectory in the replay buffer was finished
        (and truncate it).
        If not called, we assume that we continue the same trajectory (same episode).
        """
        # If we are at the start of an episode, no need to truncate
        if (self._current_ep_start != self.pos).any():
            # only consider epsiodes that are not finished
            for env_idx in np.where(self._current_ep_start != self.pos)[0]:
                self._compute_episode_length(env_idx)

                self.timeouts[self.pos - 1, env_idx] = True  # not an actual timeout, but it allows bootstrapping
    
    def convert_to_minari_buffer(self) -> List[Dict[str, np.ndarray]]:
        # Auto-truncate if last episode didn't finish
        self.truncate_last_trajectory()

        minari_buffer = []
        for env_id in range(self.n_envs):
            ep_start = 0
            while True:
                ep_length = self.ep_length[ep_start, env_id]
                if ep_length == 0:
                    # No more trajectories were collected for this env
                    break
                ep_end = ep_start + ep_length
                minari_buffer.append({
                    "observations": np.concatenate((self.observations[ep_start:ep_end, env_id], np.expand_dims(self.next_observations[ep_end-1, env_id], axis=0)),axis=0),
                    "actions": self.actions[ep_start:ep_end, env_id],
                    "rewards": self.rewards[ep_start:ep_end, env_id],
                    "terminations": self.dones[ep_start:ep_end, env_id],
                    "truncations": self.timeouts[ep_start:ep_end, env_id],
                    "infos": {key: np.pad(value[ep_start:ep_end, env_id], (1,0)) for key, value in self.infos.items()}
                })

                ep_start = ep_end

        return minari_buffer
        
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
        for seed in range(3):
            run_name = f"{env_id}-SAC-{seed}"
            n_envs = 5
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

            model = SAC("MlpPolicy", env, tensorboard_log=f"runs/{run.id}", learning_starts=10000, use_sde= False, seed=seed, 
                        replay_buffer_class=InfoReplayBuffer, 
                        replay_buffer_kwargs={"info_keys": INFO_KEYS[env_id]}
                        )
            wandb_callback = WandbCallback()
            # Save a checkpoint every 100000 steps
            checkpoint_callback = CheckpointCallback(
            save_freq=int(100000/n_envs),
            save_path=f"logs/{env_id}/",
            name_prefix="rl_model",
            save_replay_buffer=True,
            )

            model.learn(total_timesteps=3000000, callback=[wandb_callback, checkpoint_callback], log_interval=4)
            model.save(f"{env_id.lower()}-v5-sac-expert")
            run.finish()

        eval_env = make_vec_env(f"{env_id}-v5", n_envs=1)

        # Replace InfoReplayBuffer for sb3 ReplayBuffer
        model = SAC.load(f"{env_id.lower()}-v5-sac-expert", 
                        custom_objects={"replay_buffer_class": ReplayBuffer, 
                                        "replay_buffer_kwargs": {},
                                        "lr_schedule": lambda _: 0.0
                                        }
                                        )

        package_to_hub(model=model, 
                    model_name=f"{env_id.lower()}-v5-sac-expert",
                    model_architecture="SAC",
                    env_id=f"{env_id}-v5",
                    eval_env=eval_env,
                    repo_id=f"farama-minari/{env_id}-v5-SAC-expert",
                    commit_message="model")
        
        # Replace InfoReplayBuffer for sb3 ReplayBuffer
        model = SAC.load(f"logs/{env_id}/rl_model_200000_steps.zip", 
                        custom_objects={"replay_buffer_class": ReplayBuffer, 
                                        "replay_buffer_kwargs": {},
                                        "lr_schedule": lambda _: 0.0
                                        }
                                        )

        package_to_hub(model=model, 
                    model_name=f"{env_id.lower()}-v5-sac-medium",
                    model_architecture="SAC",
                    env_id=f"{env_id}-v5",
                    eval_env=eval_env,
                    repo_id=f"farama-minari/{env_id}-v5-SAC-medium",
                    commit_message="model")
        