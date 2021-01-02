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

ax.plot(nums, impala['train_mean_reward'], dtu_red)
ax.plot(nums, impala['test_mean_reward'], dtu_red, linestyle='dashed')
ax.plot(nums, nature['train_mean_reward'], dtu_blue)
ax.plot(nums, nature['test_mean_reward'], dtu_blue, linestyle='dashed')
ax.set_xlabel('Number of steps')
ax.set_ylabel('Mean episodic reward')
ax.legend(['Impala train', 'Impala test', 'Nature train', 'Nature test'])

plt.show()
fig.savefig("nature-vs-impala-32fd.pdf", bbox_inches='tight')
