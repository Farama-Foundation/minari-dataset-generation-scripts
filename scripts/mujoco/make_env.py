import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.monitor import Monitor

gym.register_envs(gymnasium_robotics)


def make_env(
    env_id: str, run_name: str | None = None, render_mode="rgb_array", eval=False, use_monitor_wrapper: bool = True
) -> gym.Env:
    """Wrapper to create the appropriate environment."""
    if env_id == "HumanoidStandup":
        env = gym.make(
            f"{env_id}-v5",
            include_cinert_in_observation=False,
            include_cvel_in_observation=False,
            include_qfrc_actuator_in_observation=False,
            include_cfrc_ext_in_observation=False,
            ctrl_cost_weight=0,
            impact_cost_weight=0,
            render_mode=render_mode,
        )
    elif env_id == "Go1":
        env = gym.make(
            "Ant-v5",
            xml_file="./mujoco_menagerie/unitree_go1/scene.xml",
            forward_reward_weight=1,
            ctrl_cost_weight=0.05,
            contact_cost_weight=5e-4,
            healthy_reward=1,
            main_body=1,
            healthy_z_range=(0.195, 0.75),
            include_cfrc_ext_in_observation=False,
            exclude_current_positions_from_observation=False,
            reset_noise_scale=0.1,
            frame_skip=25,
            max_episode_steps=1000,
            render_mode=render_mode,
        )
    elif env_id == "Go2":
        env = gym.make(
            "Ant-v5",
            xml_file="./mujoco_menagerie/unitree_go2/scene.xml",
            # forward_reward_weight=1,
            forward_reward_weight=5,
            # ctrl_cost_weight=0.05,
            ctrl_cost_weight=0.001,
            contact_cost_weight=5e-4,
            healthy_reward=1,
            main_body=1,
            healthy_z_range=(0.195, 0.75),
            include_cfrc_ext_in_observation=False,
            exclude_current_positions_from_observation=False,
            reset_noise_scale=0.1,
            frame_skip=25,
            max_episode_steps=1000,
            render_mode=render_mode,
        )
    elif env_id == "OP3":
        env = gym.make(
            "Humanoid-v5",
            xml_file="./mujoco_menagerie/robotis_op3/scene.xml",
            healthy_z_range=(0.275, 0.5),
            include_cinert_in_observation=False,
            include_cvel_in_observation=False,
            include_qfrc_actuator_in_observation=False,
            include_cfrc_ext_in_observation=False,
            ctrl_cost_weight=0,
            contact_cost_weight=0,
            frame_skip=25,
            render_mode=render_mode,
        )
    else:
        env = gym.make(
            f"{env_id}-v5",
            # include_cfrc_ext_in_observation=False,
            render_mode=render_mode,
        )
    if run_name is not None:
        env = RecordVideo(env, f"videos/{run_name}")
    if use_monitor_wrapper:
        env = Monitor(env)
    return env
