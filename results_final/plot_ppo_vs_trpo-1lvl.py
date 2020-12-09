import torch
import matplotlib.pyplot as plt
from matplotlib import rc

dtu_red = '#990000'
dtu_red2 = '#E83F48'
dtu_orange = '#FC7634'
dtu_blue = '#2F3EEA'
dtu_navy = '#030F4F'

def read_torch(filename):
    return torch.load(filename, map_location=torch.device('cpu'))


ppo = read_torch('ppo-vs-trpo/ppo-1lvl-32fd-impala.pt')
trpo = read_torch('ppo-vs-trpo/trpo-1lvl-32fd-impala.pt')

# Figure and figure options
plt.style.use('seaborn-poster')
plt.rc('grid', linestyle="--", color='grey', alpha=0.2)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

fig, ax = plt.subplots(figsize=(12,5))

ax.grid(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

r = ppo['step']

ax.plot(r, ppo['train_mean_reward'], dtu_red, r, ppo['test_mean_reward'], dtu_red, '--')
ax.plot(r, trpo['train_mean_reward'], dtu_blue, r, trpo['test_mean_reward'], dtu_blue, '--')
ax.legend(['PPO train', 'PPO test', 'TRPO train', 'TRPO test'])
ax.set_xlabel('Number of steps')
ax.set_ylabel('Mean episodic reward')

plt.show()
# fig.savefig("trpo-vs-ppo.pdf", bbox_inches='tight')
