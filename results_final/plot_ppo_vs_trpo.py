import torch
import matplotlib.pyplot as plt
from matplotlib import rc

dtu_red = 'C3'
dtu_red2 = '#E83F48'
dtu_orange = 'C1'
dtu_blue = 'C0'
dtu_navy = '#030F4F'

def read_torch(filename):
    return torch.load(filename, map_location=torch.device('cpu'))


ppo = read_torch('ppo-vs-trpo/ppo-400.pt')
trpo = read_torch('ppo-vs-trpo/trpo-400.pt')

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

ax.plot(r, ppo['train_mean_reward'], dtu_red)
ax.plot(r, ppo['test_mean_reward'], dtu_red, linestyle='dashed')
ax.plot(r, trpo['train_mean_reward'], dtu_blue)
ax.plot(r, trpo['test_mean_reward'], dtu_blue, linestyle='dashed')
ax.legend(['PPO train', 'PPO test', 'TRPO train', 'TRPO test'])
ax.set_xlabel('Number of steps')
ax.set_ylabel('Mean episodic reward')

plt.show()
fig.savefig("trpo-vs-ppo.pdf", bbox_inches='tight')
