import torch
import matplotlib.pyplot as plt
import numpy as np

def read_torch(filename):
    return torch.load(filename, map_location=torch.device('cpu'))

nums = [10, 20, 30, 40, 50, 60, 70, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
nums = [10, 40, 80, 100, 200, 300, 400, 500, 600, 700]
train_reward = []
test_reward = []

for num in nums:
    log = torch.load(f'generalization-32fd/{num}levels.pt', map_location=torch.device('cpu'))

    train_reward.append(np.mean(log['train_mean_reward'][-20:-1]))
    test_reward.append(np.mean(log['test_mean_reward'][-20:-1]))
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(nums, train_reward)
ax.plot(nums, test_reward)
ax.set_title('Test and train reward for different levels, numSteps=256, totalSteps=2e6')

plt.show()
