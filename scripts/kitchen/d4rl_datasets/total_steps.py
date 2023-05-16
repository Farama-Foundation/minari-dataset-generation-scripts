import h5py
import gym
import d4rl

env = gym.make('kitchen-complete-v0')
env.reset()

with h5py.File('kitchen_microwave_kettle_light_slider-v0(1).hdf5', 'r') as f:
    actions = f['actions'][:]
    terminals = f['terminals'][:]
    timeouts = f['timeouts'][:]
    max_steps = 0
    steps = 0
    min_steps = 0
    episode = 0
    for term, tout, act in zip(terminals, timeouts, actions):
        steps += 1
        if episode%20 == 0:
            env.render(mode='human')
            env.step(act)
        if term == 1:
            print(f'EPISODE STEPS: {steps}')
            env.reset()
            if steps > max_steps:
                max_steps = steps
            if episode == 0:
                min_steps = steps
            else:
                if min_steps > steps:
                    min_steps = steps
            episode += 1
            steps = 0

print(f'MAX STEPS: {max_steps}')
print(f'MIN STEPS: {min_steps}')