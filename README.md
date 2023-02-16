# D4RL to Minari datasets
This repository contains the scripts to convert the [D4RL](https://github.com/Farama-Foundation/D4RL) environment datasets based on MuJoCo to Minari datasets. The Minari tool is currently under development in the following PR https://github.com/Farama-Foundation/Minari/pull/31

The envrionments used to regenerate the datasets are refactored versions of the originals. These new environment versions are now maintained in the [Gymnasium-Robotics](https://robotics.farama.org/) project, they follow the Gymnasium API, and have been updated to use the latest mujoco bindings from Deepmind.
## Installation

```
git clone https://github.com/rodrigodelazcano/d4rl-minari-dataset-generation
cd d4rl-minari-dataset-generation/
pip install -r requirements.txt
```

## Create datasets
### Point Maze
The point maze datasets have been regenerated using the same `q_iteration` expert policy as in the original D4RL paper. The environments used can be found [here](https://robotics.farama.org/envs/maze/point_maze/).

You can run the script used to regenerate the datasets with:
```
python scripts/pointmaze/create_pointmaze_dataset --env "PointMaze_UMaze-v3" --dataset_name="pointmaze-umaze-v0" --maze-solver="QIteration"
```

This will generate a local Minari dataset named `pointmaze-umaze-v0` for the `PointMaze_UMaze-v3` environment, using `q_iteration` as the expert policy, Depth First Search can
also be used as the algorithm to generate a path to the goal by passing "DFS" instead of "QIteration".

### Adroit Hand
The Minari datasets for the Adroit Hand environments

```
python recreate_adroit_door.py
```

```
python recreate_adroit_hammer.py
```

```
python recreate_adroit_pen.py
```

```
python recreate_adroit_relocate.py
```

### More datasets to come
* `AntMaze`
* `Gymnasium Mujoco`
* `KitchenFranka`
* `Minigrid`