import torch
import matplotlib.pyplot as plt
from matplotlib import rc

def read_torch(filename):
    return torch.load(filename, map_location=torch.device('cpu'))

impala = read_torch('encoder/impala-1000-32fd.pt')
nature = read_torch('encoder/nature-1000-32fd.pt')

# Figure and figure options
plt.style.use('seaborn-poster')
plt.rc('grid', linestyle="--", color='grey', alpha=0.2)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 1000})
rc('text', usetex=True)

fig, ax = plt.subplots(figsize=(12,5))

ax.grid(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

nums = impala['step']

ax.plot(nums, impala['train_mean_reward'], 'C3', nums, impala['test_mean_reward'], 'C3--')
ax.plot(nums, nature['train_mean_reward'], 'C0', nums, nature['test_mean_reward'], 'C0--')
ax.set_xlabel('Number of steps')
ax.set_ylabel('Mean episodic reward')
ax.legend(['Impala train', 'Impala test', 'Nature train', 'Nature test'])

plt.show()
fig.savefig("nature-vs-impala-32fd.pdf", bbox_inches='tight')
