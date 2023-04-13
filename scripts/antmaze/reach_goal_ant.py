from gymnasium_robotics import GoalEnv
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames
from gymnasium import spaces

import numpy as np
import tempfile
import xml.etree.ElementTree as ET

import sys
from gymnasium.envs.mujoco.ant_v4 import AntEnv
from gymnasium.utils.ezpickle import EzPickle

from typing import Dict, List, Optional, Union
from os import path


class GoalReachAnt(GoalEnv, EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        reward_type: str = "sparse",
        continuing_task: bool = True,
        **kwargs,
    ):
        
        # Get the ant.xml path from the Gymnasium package
        ant_xml_file_path = path.join(
            path.dirname(sys.modules[AntEnv.__module__].__file__), "assets/ant.xml"
        )
        
        tree = ET.parse(ant_xml_file_path)
        worldbody = tree.find(".//worldbody")
        
        # Add target site for visualization
        ET.SubElement(
            worldbody,
            "site",
            name="target",
            pos=f"0 0 0.75",
            size="0.25",
            rgba="1 0 0 0.7",
            type="sphere",
        )
        
        # Save new xml with maze to a temporary file
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_xml_path = path.join(path.dirname(tmp_dir), "ant_maze.xml")
            tree.write(temp_xml_path)
        
        self.ant_env = AntEnv(
            xml_file=temp_xml_path,
            exclude_current_positions_from_observation=False,
            render_mode=render_mode,
            reset_noise_scale=0.0,
            **kwargs,
        )
        
        self._model_names = MujocoModelNames(self.ant_env.model)
        self.target_site_id = self._model_names.site_name2id["target"]
        self.action_space = self.ant_env.action_space
        obs_shape: tuple = self.ant_env.observation_space.shape
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(
                    -np.inf, np.inf, shape=(obs_shape[0] - 2,), dtype="float64"
                ),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64"),
                desired_goal=spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64"),
            )
        )

        self.render_mode = render_mode
        
        self.reward_type = reward_type

        EzPickle.__init__(
            self,
            render_mode,
            reward_type,
            continuing_task,
            **kwargs,
        )
    
    def reset(self, *, seed: Optional[int] = None, **kwargs):
        super().reset(seed=seed)
        self.goal = self._generate_goal()
        self.ant_env.model.site_pos[self.target_site_id] = np.append(
            self.goal, 0.75)
        obs, info = self.ant_env.reset(seed=seed)
        obs_dict = self._get_obs(obs)

        return obs_dict, info
    
    def step(self, action):
        ant_obs, _, _, _, info = self.ant_env.step(action)
        obs = self._get_obs(ant_obs)

        terminated = self.compute_terminated(obs["achieved_goal"], self.goal, info)
        truncated = self.compute_truncated(obs["achieved_goal"], self.goal, info)

        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info
    
    def _get_obs(self, ant_obs: np.ndarray) -> Dict[str, np.ndarray]:
        achieved_goal = ant_obs[:2]
        goal_direction = self.goal - achieved_goal
        observation = np.concatenate([ant_obs[2:], goal_direction])
        return {
            "observation": observation,
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }
    
    def _generate_goal(self, goal_region_radius=10.):
        th = 2 * np.pi * self.np_random.uniform()
        radius = goal_region_radius * self.np_random.uniform()
        return radius * np.array([np.cos(th), np.sin(th)])

    def render(self):
        return self.ant_env.render()

    def close(self):
        super().close()
        self.ant_env.close()
    
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> float:
        if self.reward_type == "dense":
            return np.exp(-np.linalg.norm(desired_goal - achieved_goal))
        elif self.reward_type == "sparse":
            return 1.0 if np.linalg.norm(achieved_goal - desired_goal) <= 0.45 else 0.0
    
    def compute_truncated(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> bool:
        return False

    def compute_terminated(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> bool:
        return bool(np.linalg.norm(achieved_goal - desired_goal) <= 0.45)