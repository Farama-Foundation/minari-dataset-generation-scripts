"""
This script generates AntMaze datasets.

Usage:

python create_antmaze_dataset.py

See --help for full list of options.
"""

import gymnasium as gym
import minari
from minari import DataCollectorV0, StepDataCallback
from tqdm import tqdm

import numpy as np
import argparse

from stable_baselines3 import SAC
from controller import WaypointController


class AntMazeStepDataCallback(StepDataCallback):
    """Add environment state information to 'infos'.

    Also, since the environment generates a new target every time it reaches a goal, the
    environment is never terminated or truncated. This callback overrides the truncation
    value to True when the step returns a True 'success' key in 'infos'. This way we can
    divide the Minari dataset into diferent trajectories.
    """

    def __call__(
        self, env, obs, info, action=None, rew=None, terminated=None, truncated=None
    ):
        step_data = super().__call__(env, obs, info, action, rew, terminated, truncated)

        # The AntMaze env in the local Gymnasium-Robotics repo has been modified
        # to include a 'success' key in the info dict to indicate the
        # goal has been reached
        if step_data["infos"]["success"]:
            step_data["truncations"] = True

        # To restore the MuJoCo simulation state, we need to store qpos and qvel
        step_data["infos"]["qpos"] = np.concatenate(
            [obs["achieved_goal"], obs["observation"][:13]]
        )
        step_data["infos"]["qvel"] = obs["observation"][13:]
        step_data["infos"]["goal"] = obs["desired_goal"]

        return step_data


def wrap_maze_obs(obs, waypoint_xy):
    """Wrap the maze obs into one suitable for GoalReachAnt."""
    goal_direction = waypoint_xy - obs["achieved_goal"]
    observation = np.concatenate([obs["observation"], goal_direction])
    return {
        "observation": observation,
        "achieved_goal": obs["achieved_goal"].copy(),
        "desired_goal": waypoint_xy.copy(),
    }


def init_dataset(collector_env, args):
    """Initalise a local Minari dataset."""
    return minari.create_dataset_from_collector_env(
        collector_env=collector_env,
        dataset_id=args.dataset_name,
        algorithm_name=f"{args.maze_solver}+SAC",
        code_permalink="https://github.com/rodrigodelazcano/d4rl-minari-dataset-generation",
        author=args.author,
        author_email=args.author_email,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="AntMaze_UMaze-v3",
                        help="environment id to collect data from")
    parser.add_argument("--maze-solver", type=str, default="QIteration",
                        help="algorithm to solve the maze and generate waypoints, can be DFS or QIteration")
    parser.add_argument("--policy-file", type=str, default="GoalReachAnt_model",
                        help="filepath for goal-reaching policy model")
    parser.add_argument("--total-timesteps", type=int, default=int(1e6),
                        help="total number of timesteps to collect data for")
    parser.add_argument("--timeout-steps", type=int, default=500,
                        help="max number of unsuccessful steps before env reset")
    parser.add_argument("--action-noise", type=float, default=0.2,
                        help="standard deviation of noise added to action")
    parser.add_argument("--dataset-name", type=str, default="antmaze-umaze-v0",
                        help="name of the Minari dataset")
    parser.add_argument("--seed", type=int, default=123,
                        help="seed for Numpy and the Gymnasium environment")
    parser.add_argument("--checkpoint_interval", type=int, default=200000,
                        help="number of steps to collect before caching to disk")
    parser.add_argument("--author", type=str, default=None,
                        help="name of the author of the dataset")
    parser.add_argument("--author-email", type=str, default=None,
                        help="email of the author of the dataset")
    parser.add_argument("--upload-dataset", type=bool, default=False,
                        help="upload dataset to Farama server after collecting the data")
    parser.add_argument("--path_to_private_key", type=str, default=None,
                        help="path to the private key to upload datset to the Farama GCP server")
    args = parser.parse_args()

    # Check if the temp dataset already exist and load to add more data
    if args.dataset_name in minari.list_local_datasets():
        dataset = minari.load_dataset(args.dataset_name)
    else:
        dataset = None

    # Continuing task => the episode doesn't terminate or truncate when reaching a goal
    # it will generate a new target. For this reason we set the maximum episode steps to
    # the desired size of our Minari dataset (evade truncation due to time limit)
    env = gym.make(
        args.env, continuing_task=True, max_episode_steps=args.total_timesteps
    )

    # Data collector wrapper to save temporary data while stepping. Characteristics:
    #   * Custom StepDataCallback to add extra state information to 'infos' and divide dataset in
    #     different episodes by overriding truncation value to True when target is reached
    #   * Record the 'info' value of every step
    collector_env = DataCollectorV0(
        env, step_data_callback=AntMazeStepDataCallback, record_infos=True
    )

    seed = args.seed
    np.random.seed(seed)

    model = SAC.load(args.policy_file)

    def action_callback(obs, waypoint_xy):
        return model.predict(wrap_maze_obs(obs, waypoint_xy))[0]

    waypoint_controller = WaypointController(env.maze, action_callback)
    obs, info = collector_env.reset(seed=seed)

    steps_since_success = 0
    steps_since_ckpt = 0

    for step in tqdm(range(args.total_timesteps)):
        reset = False
        steps_since_success += 1
        steps_since_ckpt += 1

        # Compute action and add some noise
        action = waypoint_controller.compute_action(obs)
        action += args.action_noise * np.random.randn(*action.shape)
        action = np.clip(action, -1.0, 1.0)

        obs, _, _, _, info = collector_env.step(action)

        # Reset timeout counter (but not env, as continuing task)
        if info["success"]:
            steps_since_success = 0

        # Reset env if the Ant has not reached the goal in 500 steps (by default)
        if steps_since_success > args.timeout_steps:
            reset = True

        # Update local Minari dataset every ~200000 steps (once the episode is complete),
        # and at the final step. This works as a checkpoint to not lose the already
        # collected data.
        should_checkpoint = (info["success"] or reset) and (
            steps_since_ckpt >= args.checkpoint_interval
        )

        if should_checkpoint or (step + 1 == args.total_timesteps):
            steps_since_ckpt = 0

            if dataset is None:
                dataset = init_dataset(collector_env, args)
            else:
                dataset.update_dataset_from_collector_env(collector_env)

        # Reset the environment, either due to timeout or checkpointing.
        if reset:
            seed += 1  # Increment the seed to prevent repeating old episodes
            obs, info = collector_env.reset(seed=seed)
            steps_since_success = 0

    if args.upload_dataset:
        minari.upload_dataset(args.dataset_name, args.path_to_private_key)
