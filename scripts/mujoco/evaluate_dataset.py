import minari
import numpy as np

for dataset_id in minari.list_local_datasets().keys():
    ds = minari.load_dataset(dataset_id)

    episodic_returns = [ep.rewards.sum() for ep in ds]
    episodic_return = sum([ep.rewards.sum() / len(ds) for ep in ds])
    episodic_terminations_percentage = sum([ep.terminations[-1] for ep in ds]) / len(ds) * 100
    avg_episode_length = ds.total_steps / ds.total_episodes

    print(f"{dataset_id} - Averate return: {episodic_return}")
    print(f"{dataset_id} - Minimum return: {min(episodic_returns)}")
    percentile_1 = np.percentile(episodic_returns, 1)
    print(f"{dataset_id} - 1% low return: {percentile_1}")
    print(f"{dataset_id} - Episodes with a termination (%): {episodic_terminations_percentage}")
    print(f"{dataset_id} - Average epsidode lenght: {avg_episode_length}")

    print(f"{dataset_id} - Observation_space: {ds.observation_space}")
    print(f"{dataset_id} - Action_space: {ds.action_space}")
