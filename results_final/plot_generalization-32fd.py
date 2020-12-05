import torch
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

plt.style.use('seaborn-poster')
plt.rc('grid', linestyle="--", color='grey', alpha=0.2)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 1000})
rc('text', usetex=True)

def read_torch(filename):
    return torch.load(filename, map_location=torch.device('cpu'))

nums = [10, 20, 30, 40, 50, 60, 70, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
nums = [10, 40, 80, 100, 200, 300, 400, 500, 600, 700]
train_reward = []
test_reward = []
train_std = []
test_std = []

for num in nums:
    log = torch.load(f'generalization-32fd/{num}levels.pt', map_location=torch.device('cpu'))
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
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 1000})
rc('text', usetex=True)

fig, ax = plt.subplots(figsize=(12,5))

ax.grid(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.plot(nums, train_reward, color='C3')
ax.plot(nums, test_reward)
ax.set_title('Test and train reward for different levels')

ax.fill_between(nums, train_reward+2*test_std, train_reward-2*test_std, alpha=0.2, color='C3')
ax.fill_between(nums, test_reward+2*test_std, test_reward-2*test_std, alpha=0.2, color='C0')
# ax.errorbar(nums, test_reward, xerr=0.5, yerr=test_std, linestyle='', color='C1')
# ax.errorbar(nums, train_reward, xerr=0.5, yerr=train_std, linestyle='', color='C0')
ax.legend(['Test reward', 'Train reward'], loc ="lower right")
ax.set_xlabel('Number of levels')
ax.set_ylabel('Mean episodic reward')

plt.show()
fig.savefig("foo.pdf", bbox_inches='tight')
