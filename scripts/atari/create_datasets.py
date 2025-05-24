import os
from pathlib import Path
import warnings
import gymnasium as gym
import ale_py
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError
import jax
import jax.numpy as jnp
import numpy as np
import flax
import flax.linen as nn
from minari import DataCollector
import minari


gym.register_envs(ale_py)


class ResidualBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        inputs = x
        x = nn.relu(x)
        x = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
        )(x)
        return x + inputs


class ConvSequence(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
        )(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = ResidualBlock(self.channels)(x)
        x = ResidualBlock(self.channels)(x)
        return x


class Network(nn.Module):
    channelss = (16, 32, 32)

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        for channels in self.channelss:
            x = ConvSequence(channels)(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        return x


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1)(x)


class Actor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.action_dim)(x)



def make_env(env_id):
    env = gym.make(env_id, repeat_action_probability=0, max_episode_steps=108000 // 4)
    data_collector = DataCollector(env)
    env = gym.wrappers.ResizeObservation(data_collector, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)
    return env, data_collector


def generate_dataset(env_id, num_episodes):
    env_id = env_id.replace("ALE/", "")
    exp_name = "cleanba_ppo_envpool_impala_atari_wrapper"
    model_path = hf_hub_download(repo_id=f"cleanrl/{env_id}-{exp_name}-seed1", filename=f"{exp_name}.cleanrl_model")
    
    print(f"Generating dataset for {env_id}...")
    envs, data_collector = make_env(f"ALE/{env_id}")
    network = Network()
    actor = Actor(action_dim=envs.action_space.n)
    critic = Critic()
    key = jax.random.PRNGKey(42)
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)
    network_params = network.init(network_key, np.array([envs.observation_space.sample()]))
    actor_params = actor.init(actor_key, network.apply(network_params, np.array([envs.observation_space.sample()])))
    critic_params = critic.init(critic_key, network.apply(network_params, np.array([envs.observation_space.sample()])))
    # note: critic_params is not used in this script
    with open(model_path, "rb") as f:
        (args, (network_params, actor_params, critic_params)) = flax.serialization.from_bytes(
            (None, (network_params, actor_params, critic_params)), f.read()
        )

    @jax.jit
    def get_action_and_value(
        network_params: flax.core.FrozenDict,
        actor_params: flax.core.FrozenDict,
        next_obs: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        hidden = network.apply(network_params, next_obs)
        logits = actor.apply(actor_params, hidden)
        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        return action, key

    returns = []
    for _ in range(num_episodes):
        episodic_return = 0
        next_obs, _ = envs.reset()
        done = False

        while not done:
            next_obs = jnp.array(next_obs)[None]
            actions, key = get_action_and_value(network_params, actor_params, next_obs, key)
            next_obs, rew, ter, tru, infos = envs.step(int(np.array(actions[0])))
            episodic_return += rew
            done = ter or tru
        returns.append(episodic_return)
        
    print(f"Average return for {env_id}: {np.mean(returns)}")
    
    hf_url = f"https://huggingface.co/cleanrl/{env_id}-{exp_name}-seed1"
    dataset = data_collector.create_dataset(
        dataset_id=f"atari/{env_id[:-len('-v5')].lower()}/expert-v0",
        author="Omar G. Younis",
        author_email="omar@farama.org",
        algorithm_name="CleanBA PPO Impala",
        description=Path(__file__).parent.joinpath("description.md").read_text().format(hf_url=hf_url),
        code_permalink="https://github.com/Farama-Foundation/minari-dataset-generation-scripts",
        requirements=["gymnasium[atari,accept-rom-license]"]
    )
    envs.close()

    return dataset


if __name__ == "__main__":
    ENV_LIST = [
        env_id
        for env_id in gym.registry
        if env_id.startswith("ALE/") and "-ram-" not in env_id
    ]

    for env_id in ENV_LIST:
        if f"atari/{env_id[len('ALE/'):-len('-v5')].lower()}/expert-v0" in minari.list_remote_datasets(prefix="atari"):
            print(f"Dataset for {env_id} already exists. Skipping...")
            continue
        try:
            dataset = generate_dataset(env_id, num_episodes=10)

            minari.upload_dataset(dataset.spec.dataset_id, token=os.environ["HF_TOKEN"])
            minari.delete_dataset(dataset.spec.dataset_id)
        except RepositoryNotFoundError:
            warnings.warn(f"There is no model for {env_id} on CleanRL repo. Skipping...")
        


