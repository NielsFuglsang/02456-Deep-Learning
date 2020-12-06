import torch
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np


def read_torch(filename):
    return torch.load(filename, map_location=torch.device('cpu'))

x = [0.01, 0.1, 0.25, 0.5, 1, 2, 5, 10]
betas = ['001', '01', '025', '05', '1', '2', '5', '10']
train_reward = []
test_reward = []
train_std = []
test_std = []

for beta in betas:
    log = torch.load(f'trpo/trpo-10lvl-beta{beta}.pt', map_location=torch.device('cpu'))
    last_train = log['train_mean_reward'][-20:-1]
    last_test = log['test_mean_reward'][-20:-1]
    train_reward.append(np.mean(last_train))
    test_reward.append(np.mean(last_test))
    train_std.append(np.std(last_train))
    test_std.append(np.std(last_test))

train_reward = np.array(train_reward)
test_reward = np.array(test_reward)
train_std = np.array(train_std)
test_std = np.array(test_std)

# Figure and figure options
plt.style.use('seaborn-poster')
plt.rc('grid', linestyle="--", color='grey', alpha=0.2)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

fig, ax = plt.subplots(figsize=(12,5))

ax.grid(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.plot(x, train_reward, color='C3')
ax.plot(x, test_reward)

ax.fill_between(x, train_reward+2*test_std, train_reward-2*test_std, alpha=0.2, color='C3')
ax.fill_between(x, test_reward+2*test_std, test_reward-2*test_std, alpha=0.2, color='C0')
# ax.errorbar(nums, test_reward, xerr=0.5, yerr=test_std, linestyle='', color='C1')
# ax.errorbar(nums, train_reward, xerr=0.5, yerr=train_std, linestyle='', color='C0')
ax.legend(['Train reward', 'Test reward'], loc ="upper right")
ax.set_xlabel('Value of beta')
ax.set_ylabel('Mean episodic reward')
plt.xscale('log')

plt.show()
fig.savefig("trpo-beta-log.pdf", bbox_inches='tight')
