
import gymnasium as gym
from controller import WaypointController
from minari import DataCollectorV0, StepDataCallback
import minari
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import argparse
R="r"
G="g"

# single-goal multi-reset location maps:
EVAL_ENV_MAPS = {"open": [
                [1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, G, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1],],
            "umaze": [[1, 1, 1, 1, 1],
                [1, G, 0, 0, 1],
                [1, 1, 1, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1]], 
            "medium": [[1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 1, 1, 0, 0, 1],
                [1, 0, 0, 1, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 0, 0, 1, 0, 1],
                [1, 0, 0, 0, 1, 0, G, 1],
                [1, 1, 1, 1, 1, 1, 1, 1]],
            "large": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 0, 1, 0, G, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
                }

DATASET_ID_TO_ENV_ID = {"pointmaze-open-v1": "PointMaze_Open-v3", 
                        "pointmaze-open-dense-v1": "PointMaze_OpenDense-v3", 
                        "pointmaze-umaze-v1": "PointMaze_UMaze-v3", 
                        "pointmaze-umaze-dense-v1": "PointMaze_UMazeDense-v3", 
                        "pointmaze-medium-v1": "PointMaze_Medium-v3", 
                        "pointmaze-medium-dense-v1": "PointMaze_MediumDense-v3",
                        "pointmaze-large-v1": "PointMaze_Large-v3",
                        "pointmaze-large-dense-v1": "PointMaze_LargeDense-v3"
                    }

class PointMazeStepDataCallback(StepDataCallback):
    """Add environment state information to 'infos'.
    
    Also, since the environment generates a new target every time it reaches a goal, the environment is
    never terminated or truncated. This callback overrides the truncation value to True when the step
    returns a True 'succes' key in 'infos'. This way we can divide the Minari dataset into diferent trajectories.
    """
    def __call__(self, env, obs, info, action=None, rew=None, terminated=None, truncated=None):
        qpos = obs['observation'][:2]
        qvel = obs['observation'][2:]
        goal = obs['desired_goal']
        
        step_data = super().__call__(env, obs, info, action, rew, terminated, truncated)
    
        if step_data['infos']['success']:
            step_data['truncations'] = True
           
        step_data['infos']['qpos'] = qpos
        step_data['infos']['qvel'] = qvel
        step_data['infos']['goal'] = goal
        
        return step_data
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--maze-solver", type=str, default="QIteration", help="algorithm to solve the maze and generate waypoints, can ve DFS or QIteration")
    parser.add_argument("--author", type=str, help="name of the author of the dataset", default=None)
    parser.add_argument("--author-email", type=str, help="email of the author of the dataset", default=None)
    parser.add_argument("--upload-dataset", type=bool, default=False, help="upload dataset to Farama server after collecting the data")
    parser.add_argument("--path_to_private_key", type=str, help="path to the private key to upload datset to the Farama GCP server", default=None)
    args = parser.parse_args()
    
    for dataset_id, env_id in DATASET_ID_TO_ENV_ID.items():
        # Check if dataset already exist and load to add more data
        if dataset_id in minari.list_local_datasets():
            dataset = minari.load_dataset(dataset_id)
        else:
            dataset = None
        
        split_dataset_id = dataset_id.split('-')
        # continuing task => the episode doesn't terminate or truncate when reaching a goal
        # it will generate a new target. For this reason we set the maximum episode steps to
        # the desired size of our Minari dataset (evade truncation due to time limit)
        env = gym.make(env_id, continuing_task=True, reset_target=True, max_episode_steps=1e6)
        
        # Data collector wrapper to save temporary data while stepping. Characteristics:
        #   * Custom StepDataCallback to add extra state information to 'infos' and divide dataset in different episodes by overridng 
        #     truncation value to True when target is reached
        #   * Record the 'info' value of every step
        #   * Record 100000 in in-memory buffers before dumpin everything to temporary file in disk       
        collector_env = DataCollectorV0(env, step_data_callback=PointMazeStepDataCallback, record_infos=True)

        obs, _ = collector_env.reset(seed=123)

        waypoint_controller = WaypointController(maze=env.maze)

        for n_step in tqdm(range(1_000_000)):
            action = waypoint_controller.compute_action(obs)
            # Add some noise to each step action
            action += np.random.randn(*action.shape)*0.5
            action = np.clip(action, -1, 1)

            obs, rew, terminated, truncated, info = collector_env.step(action)

            if (n_step + 1) % 200000 == 0:
                if dataset is None:
                    eval_env_id = env.spec.id
                    eval_env = gym.make(eval_env_id, maze_map=EVAL_ENV_MAPS[split_dataset_id[1]],
                                        continuing_task=True,
                                        reset_target=False)
                    eval_waypoint_controller = WaypointController(eval_env.maze)
                    dataset = minari.create_dataset_from_collector_env(collector_env=collector_env, 
                                                                       dataset_id=dataset_id,
                                                                       eval_env=eval_env,
                                                                       expert_policy=eval_waypoint_controller.compute_action,
                                                                       algorithm_name=args.maze_solver, 
                                                                       code_permalink="https://github.com/rodrigodelazcano/d4rl-minari-dataset-generation", 
                                                                       author=args.author, 
                                                                       author_email=args.author_email
                                                                       )

                    eval_env.close()
                else:
                    # Update local Minari dataset every 200000 steps.
                    # This works as a checkpoint to not lose the already collected data
                    dataset.update_dataset_from_collector_env(collector_env)
        
        if args.upload_dataset:
            minari.upload_dataset(dataset_id=args.dataset_name, path_to_private_key=args.path_to_private_key)
