from huggingface_sb3 import load_from_hub
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym

checkpoint = load_from_hub(
    repo_id="farama-minari/HalfCheetah-v5-SAC-expert",
    filename="halfcheetah-v5-sac-expert.zip",
)
model = SAC.load(checkpoint)

env = gym.make("HalfCheetah-v5", render_mode="human")

terminated = True
truncated = True

while True:
    if terminated or truncated:
        obs, _ = env.reset()
    action = model.predict(obs)
    obs, _, terminated, truncated,
