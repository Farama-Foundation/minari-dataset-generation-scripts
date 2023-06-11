"""
This script runs standard sanity checks that apply to all Minari datasets.

Usage:

python check_dataset.py <dataset_id>
"""

import numpy as np
import scipy
import argparse

import minari
from gymnasium import spaces


def get_infos(dataset, key):
    """ Get a numpy array of infos/key, for all episodes. """
    return dataset._data.apply(lambda ep: ep["infos"][key][()])


def print_avg_returns(dataset):
    """ Print returns for manual sanity checking. """
    all_returns = []
    num_truncations = 0
    num_terminations = 0

    for ep in dataset:
        all_returns.append(np.sum(ep.rewards))
        num_truncations += np.sum(ep.truncations)
        num_terminations += np.sum(ep.terminations)

    print("  | Average returns:", np.mean(all_returns))
    print("  | Num truncations:", num_truncations)
    print("  | Num terminations:", num_terminations)



def check_identical_values(dataset, check_keys=['actions', 'observations']):
    """ Check that values of actions/observations in episodes are not identical. """
    for i, episode in enumerate(dataset):
        for key in check_keys:
            values = getattr(episode, key)

            if len(values) < 3:
                continue

            values_0 = values[0]
            values_mid = values[values.shape[0]//2]
            values_last = values[-1]

            test_values = np.c_[values_0, values_mid, values_last].T
            dists = scipy.spatial.distance.pdist(test_values)
            not_same = dists > 0

            message = (f"Some of first/mid/last values of ep[{i}].{key} "
                       f"(length {len(values)}) are identical:\n{test_values}")
            assert np.all(not_same), message


def get_obs_act_shape(dataset):
    """ Compute obs/act space shape, taking into account flattening. """
    env = dataset.recover_environment()
    action_space_shape = env.action_space.shape

    obs_space = env.observation_space
    act_space = env.action_space

    # Flatten the observation and action space shapes if they are Dicts or Tuples
    if isinstance(obs_space, (spaces.Dict, spaces.Tuple)):
        obs_shape = spaces.flatten_space(obs_space).shape
    else:
        obs_shape = obs_space.shape

    if isinstance(act_space, (spaces.Dict, spaces.Tuple)):
        act_shape = spaces.flatten_space(act_space).shape
    else:
        act_shape = act_space.shape

    return obs_shape, act_shape


def check_episode_shapes(dataset):
    """ Check that all entries in each episode have correct shape.
        From https://minari.farama.org/content/dataset_standards/,
        each entry in an episode is of type np.ndarray, with shape:
          - actions:      (num_steps, action_space_shape)
          - observations: (num_steps + 1, obs_space_shape)
          - rewards:      (num_steps)
          - terminations: (num_steps)
          - truncations:  (num_steps)
        NOTE: We have squeezed the latter 3 shapes which were (num_steps, 1)
              in the documentation.
    """
    check_keys = ['actions', 'observations', 'rewards', 'terminations', 'truncations']

    obs_shape, act_shape = get_obs_act_shape(dataset)

    # Check episodes
    for i, ep in enumerate(dataset):

        num_steps = ep.total_timesteps

        # Check that all keys are numpy arrays
        for key in check_keys:
            nd_array_message = f"ep[{i}].{key} is not np.ndarray"
            assert isinstance(getattr(ep, key), np.ndarray), nd_array_message
        
        # Check each key has the correct shape
        target_shapes = {
            "actions": (num_steps, *act_shape),
            "observations": (num_steps + 1, *obs_shape),
            "rewards": (num_steps,),
            "terminations": (num_steps,),
            "truncations": (num_steps,)
        }

        for key, target_shape in target_shapes.items():
            actual_shape = getattr(ep, key).shape
            shape_message = (f"Expected episode[{i}].{key} to have shape "
                             f"{target_shape}, got {actual_shape}")
            assert actual_shape == target_shape, shape_message


def run_all_checks(dataset, verbose=True):
    """ Run all of the standard Minari checks and print results. """
    passed = True

    for check_fn in check_functions:
        try:
            check_fn(dataset)
        except AssertionError as err:
            passed = False
            print('FAILED:', check_fn.__name__)
            print(err)
        else:
            if verbose:
                print(f"PASSED: {check_fn.__name__}")

    return passed


check_functions = [
    print_avg_returns,
    check_identical_values,
    check_episode_shapes
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str,
                        help='name of minari AntMaze dataset to check')
    args = parser.parse_args()

    dataset = minari.load_dataset(args.dataset_name)

    print('Checking:', args.dataset_name)
    passed = run_all_checks(dataset)

    print("All tests passed" if passed else "Tests FAILED")
