import pickle
import torch
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

def read_torch(filename):
    return torch.load(filename, map_location=torch.device('cpu'))

lvl10 = read_torch('generalization-32fd/10levels.pt')
lvl100 = read_torch('generalization-32fd/100levels.pt')
lvl400 = read_torch('generalization-32fd/400levels.pt')
r = lvl10['step']

# Figure and figure options
plt.style.use('seaborn-poster')
plt.rc('grid', linestyle="--", color='grey', alpha=0.2)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 1000})
rc('text', usetex=True)

fig, ax = plt.subplots(figsize=(12,5))

ax.grid(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ax.plot(r, exp['train_mean_reward'], 'C0')
ax.plot(r, lvl400['test_mean_reward'])
ax.plot(r, lvl100['test_mean_reward'])
ax.plot(r, lvl10['test_mean_reward'], 'C3')

ax.legend(['400 levels', '100 levels', '10 levels'])

ax.set_xlabel('Number of steps')
ax.set_ylabel('Mean episodic reward')

plt.show()
fig.savefig("generalization-steps.pdf", bbox_inches='tight')
