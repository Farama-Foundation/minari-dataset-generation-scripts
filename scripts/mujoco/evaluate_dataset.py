import minari

for dataset_id in minari.list_local_datasets().keys():
    ds = minari.load_dataset(dataset_id)
    episodic_return = sum([ep.rewards.sum() / len(ds) for ep in ds])
    episodic_terminations_percentage = sum([ep.terminations[-1] for ep in ds]) / len(ds) * 100
    avg_episode_length = ds.total_steps / ds.total_episodes

    print(f"{dataset_id} - Averate return: {episodic_return}")
    print(f"{dataset_id} - Episodes with a termination (%): {episodic_terminations_percentage}")
    print(f"{dataset_id} - Average epsidode lenght: {avg_episode_length}")
