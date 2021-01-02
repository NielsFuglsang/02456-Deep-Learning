import torch
import matplotlib.pyplot as plt
from matplotlib import rc

def read_torch(filename):
    return torch.load(filename, map_location=torch.device('cpu'))


ppo = read_torch('ppo-vs-trpo/ppo-100lvl-128fd-impala.pt')
trpo = read_torch('ppo-vs-trpo/trpo-100lvl-128fd-impala.pt')

# Figure and figure options
plt.style.use('seaborn-poster')
plt.rc('grid', linestyle="--", color='grey', alpha=0.2)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 1000})
rc('text', usetex=True)

fig, ax = plt.subplots(figsize=(12,5))

ax.grid(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

r = ppo['step']

ax.plot(r, ppo['train_mean_reward'], 'C3', r, ppo['test_mean_reward'], 'C3--')
ax.plot(r, trpo['train_mean_reward'], 'C0', r, trpo['test_mean_reward'], 'C0--')
ax.legend(['PPO train', 'PPO test', 'TRPO train', 'TRPO test'])
ax.set_xlabel('Number of steps')
ax.set_ylabel('Mean episodic reward')

plt.show()
fig.savefig("trpo-vs-ppo-100.pdf", bbox_inches='tight')
