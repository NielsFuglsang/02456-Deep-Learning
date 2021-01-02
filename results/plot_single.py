import torch
import matplotlib.pyplot as plt
from matplotlib import rc

def read_torch(filename):
    return torch.load(filename, map_location=torch.device('cpu'))

exp = read_torch('singles/ppo-1lvl-32fd-impala.pt')

# Figure and figure options
plt.style.use('seaborn-poster')
plt.rc('grid', linestyle="--", color='grey', alpha=0.2)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 1000})
rc('text', usetex=True)

fig, ax = plt.subplots(figsize=(12,5))

ax.grid(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

nums = exp['step']

ax.plot(nums, exp['train_mean_reward'], 'C3', nums, exp['test_mean_reward'], 'C3--')
ax.set_xlabel('Number of steps')
ax.set_ylabel('Mean episodic reward')
ax.legend(['Train', 'Test'])

plt.show()
fig.savefig("nature-vs-impala-32fd.pdf", bbox_inches='tight')
