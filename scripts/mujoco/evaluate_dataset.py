import minari

for dataset_id in minari.list_local_datasets().keys():
    ds = minari.load_dataset(dataset_id)
    episodic_return = sum([ep.rewards.sum() / len(ds) for ep in ds])
    print(f"{dataset_id} - return: {episodic_return}")
