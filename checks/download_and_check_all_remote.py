"""
This script downloads all remote datasets and runs sanity checks on them.

Usage:

python download_and_check_all_remote.py
"""

import os
import tempfile
import minari

from check_dataset import run_all_checks
from check_antmaze_dataset import run_antmaze_checks

if __name__ == "__main__":
    tmp_dir = tempfile.TemporaryDirectory()
    os.environ["MINARI_DATASETS_PATH"] = tmp_dir.name

    try:
        passed = True
        remote_datasets = minari.list_remote_datasets(compatible_minari_version=True)

        # Download all datasets first to prevent download messages cluttering the test results
        for dataset_id in remote_datasets:
            minari.download_dataset(dataset_id)

        for dataset_id in remote_datasets:
            dataset = minari.load_dataset(dataset_id)

            print(f"\n=== CHECKING {dataset_id} ===")

            if "antmaze" in dataset_id:
                passed &= run_antmaze_checks(dataset)
            else:
                passed &= run_all_checks(dataset, check_identical_values=False)

        print("All tests passed" if passed else "Tests FAILED")

    finally:
        tmp_dir.cleanup()
