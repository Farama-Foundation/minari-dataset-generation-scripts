import h5py
import numpy as np


with h5py.File('mini_kitchen_microwave_kettle_light_slider-v0.hdf5', 'a') as f:
    terminals = f['terminals'][:]
    timeouts = f['timeouts'][:]
    max_steps = 0
    steps = 0
    min_steps = 0
    episode = 0
    for term, tout in zip(terminals, timeouts):
        steps += 1
        if term == 1:
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