"""
This script runs sanity checks on AntMaze datasets.

Usage:

python check_antmaze_dataset.py <dataset_id>
"""

import numpy as np
import argparse
import minari
import scipy

import check_dataset


def check_antmaze_reset_nonterminal(dataset, reset_threshold=0.5):
    """ Check if a reset (a jump in position) occured on a non-terminal state. """
    for i, ep in enumerate(dataset):
        # Compute the distance between the x, y positions at successive
        # timesteps. The x, y coordinates are the [0, 1] indices.
        # See https://robotics.farama.org/envs/maze/ant_maze/
        positions = ep.observations[:-1, 0:1]
        next_positions = ep.observations[1:, 0:1]
        diff = np.linalg.norm(positions - next_positions, axis=1)

        assert np.all(diff <= reset_threshold), f"Non-terminal reset in episode {i}."


def check_qpos_pvel_identical_values(dataset):
    """ Check infos/qpos and infos/qvel do not have identical values. """
    qpos = check_dataset.get_infos(dataset, "qpos")
    qvel = check_dataset.get_infos(dataset, "qvel")

    for i in range(dataset.total_episodes):
        for values in [qpos[i], qvel[i]]:

            if len(values) < 3:
                continue

            values_0 = values[0]
            values_mid = values[values.shape[0]//2]
            values_last = values[-1]

            test_values = np.c_[values_0, values_mid, values_last].T
            dists = scipy.spatial.distance.pdist(test_values)
            not_same = dists > 0

            message = (f"Some of first/mid/last values of ep[{i}][infos/qpos] or qvel"
                       f" (length {len(values)}) are identical:\n{test_values}")
            assert np.all(not_same), message


def check_qpos_qvel_shapes(dataset):
    """ Check infos/qpos and infos/qvel have the correct shapes. """
    env = dataset.recover_environment()
    qpos = check_dataset.get_infos(dataset, "qpos")
    qvel = check_dataset.get_infos(dataset, "qvel")

    qpos_message = f"Expected infos/qpos to have length {dataset.total_episodes}, got {len(qpos)}"
    qvel_message = f"Expected infos/qvel to have length {dataset.total_episodes}, got {len(qvel)}"
    assert len(qpos) == dataset.total_episodes, qpos_message
    assert len(qvel) == dataset.total_episodes, qvel_message

    for i, ep in enumerate(dataset):
        num_steps = ep.total_timesteps + 1      # Same number of steps as observation
        num_q = env.unwrapped.ant_env.model.nq
        num_v = env.unwrapped.ant_env.model.nv

        qpos_shape_message = (f"Expected infos/qpos (episode {i}) to have shape "
                              f"{(num_steps, num_q)}, got {qpos[i].shape}")
        qvel_shape_message = (f"Expected infos/qvel (episode {i}) to have shape "
                              f"{(num_steps, num_v)}, got {qvel[i].shape}")
        assert qpos[i].shape == (num_steps, num_q), qpos_shape_message
        assert qvel[i].shape == (num_steps, num_v), qvel_shape_message


antmaze_check_functions = [
    check_antmaze_reset_nonterminal,
    check_qpos_pvel_identical_values,
    check_qpos_qvel_shapes
]


def run_antmaze_checks(dataset, verbose=True):
    """ Run all of the Minari and AntMaze dataset checks. """
    # Run all of the common checks first
    passed = check_dataset.run_all_checks(dataset)

    # Then run AntMaze specific checks
    for check_fn in antmaze_check_functions:
        try:
            check_fn(dataset)
        except AssertionError as err:
            passed = False
            print('FAILED:', check_fn.__name__)
            print(err)
        else:
            if verbose:
                print(f"PASSED: {check_fn.__name__}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str,
                        help="name of minari AntMaze dataset to check")
    args = parser.parse_args()

    dataset = minari.load_dataset(args.dataset_name)

    print("Checking:", args.dataset_name)
    passed = run_antmaze_checks(dataset)
    print("All tests passed" if passed else "Tests FAILED")
