
import gymnasium as gym
from controller import WaypointController
from minari import DataCollectorV0, StepDataCallback
import minari
import numpy as np
import argparse

 
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
    parser.add_argument("--env", type=str, default="PointMaze_UMaze-v3", help="environment id to collect data from")
    parser.add_argument("--maze-solver", type=str, default="QIteration", help="algorithm to solve the maze and generate waypoints, can ve DFS or QIteration")
    parser.add_argument("--dataset-name", type=str, default="pointmaze-umaze-v0", help="name of the Minari dataset")
    parser.add_argument("--author", type=str, help="name of the author of the dataset", default=None)
    parser.add_argument("--author-email", type=str, help="email of the author of the dataset", default=None)
    parser.add_argument("--upload-dataset", type=bool, default=False, help="upload dataset to Farama server after collecting the data")
    parser.add_argument("--path_to_private_key", type=str, help="path to the private key to upload datset to the Farama GCP server", default=None)
    args = parser.parse_args()
    
    # Check if dataset already exist and load to add more data
    if args.dataset_name in minari.list_local_datasets(verbose=False):
        dataset = minari.load_dataset(args.dataset_name)
    else:
        dataset = None
    
    # continuing task => the episode doesn't terminate or truncate when reaching a goal
    # it will generate a new target. For this reason we set the maximum episode steps to
    # the desired size of our Minari dataset (evade truncation due to time limit)
    env = gym.make(args.env, continuing_task=True, max_episode_steps=1e6)
    
    # Data collector wrapper to save temporary data while stepping. Characteristics:
    #   * Custom StepDataCallback to add extra state information to 'infos' and divide dataset in different episodes by overridng 
    #     truncation value to True when target is reached
    #   * Record the 'info' value of every step
    #   * Record 100000 in in-memory buffers before dumpin everything to temporary file in disk       
    collector_env = DataCollectorV0(env, step_data_callback=PointMazeStepDataCallback, record_infos=True, max_buffer_steps=100000)

    obs, _ = collector_env.reset(seed=123)

    waypoint_controller = WaypointController(maze=env.maze)

    for n_step in range(1, int(1e6)+1):
        action = waypoint_controller.compute_action(obs)
        # Add some noise to each step action
        action += np.random.randn(*action.shape)*0.5

        obs, rew, terminated, truncated, info = collector_env.step(action)  

        if n_step % 200000 == 0:
            print('STEPS RECORDED:')
            print(n_step)
            if args.dataset_name not in minari.list_local_datasets(verbose=False):
                dataset = minari.create_dataset_from_collector_env(collector_env=collector_env, dataset_name=args.dataset_name,  algorithm_name=args.maze_solver, code_permalink="https://github.com/rodrigodelazcano/d4rl-minari-dataset-generation", author=args.author, author_email=args.author_email)
            else:
                # Update local Minari dataset every 200000 steps.
                # This works as a checkpoint to not lose the already collected data
                dataset.update_dataset_from_collector_env(collector_env)
    
    if args.upload_dataset:
        minari.upload_dataset(dataset_name=args.dataset_name, path_to_private_key=args.path_to_private_key)
