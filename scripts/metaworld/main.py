from pathlib import Path
from metaworld import MT1
from metaworld import policies
import minari
from minari import DataCollector


policies_v2 = [p for p in policies.__all__ if p.endswith('V2Policy')]
patch_env_name = {
    "peg-insertion-side-v2": "peg-insert-side-v2"
}

def main():
    local_datasets = minari.list_local_datasets()
    for policy_name in policies_v2:
        policy = getattr(policies, policy_name)()
        assert policy_name.startswith('Sawyer')
        env_name = policy_name[len('Sawyer'):-len('V2Policy')]
        env_name = ''.join([
            c if not c.isupper() else f'-{c.lower()}' for c in env_name
        ]).strip('-') # transform from CamelCase to snake_case
        env_name += '-v2'
        env_name = patch_env_name.get(env_name, env_name)
        dataset_id = f"metaworld/{env_name[:-len('-v2')]}/expert-v0"
        if dataset_id in local_datasets:
            print(f"Dataset {dataset_id} already exists, skipping")
            continue

        print(f"Generating dataset for {env_name} with {policy_name}")
        mt1 = MT1(env_name, seed=42)
        env = mt1.train_classes[env_name]()
        env = DataCollector(env)
        for i in range(len(mt1.train_tasks)):
            env.unwrapped.set_task(mt1.train_tasks[i])

            obs, info = env.reset()
            done = False
            while not done:
                a = policy.get_action(obs)
                obs, _, terminated, truncated, info = env.step(a)
                done = terminated or truncated

        env.create_dataset(
            dataset_id=dataset_id,
            author="Omar G. Younis",
            author_email="omar@farama.org",
            algorithm_name="Metaworld Repository Expert Policy",
            description=Path(__file__).parent.joinpath("description.md").read_text().format(env=env_name, policy=policy.__class__.__name__),
            code_permalink="https://github.com/Farama-Foundation/minari-dataset-generation-scripts",
            requirements=["git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld"]
        )
        env.close()


if __name__ == "__main__":
    main()
