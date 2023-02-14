import gymnasium as gym
from minari.utils.other_option_data_collector import StepPreProcessor


class AdroitStepPreProcessor(StepPreProcessor):
    def __call__(self, env, **kwargs):
        step_data = super().__call__(env, **kwargs)
        step_data['state'] = env.get_env_state()
        return step_data
