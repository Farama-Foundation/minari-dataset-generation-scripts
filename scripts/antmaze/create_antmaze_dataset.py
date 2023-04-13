import gymnasium as gym
from gymnasium.envs.registration import register


register(
    id="GoalReachAnt-v0",
    entry_point="reach_goal_ant:GoalReachAnt",
    max_episode_steps=300,
    )


env = gym.make('GoalReachAnt-v0')

