import torch
import matplotlib.pyplot as plt
import numpy as np

def read_torch(filename):
    return torch.load(filename, map_location=torch.device('cpu'))

nums = [10, 30, 50, 70, 100, 200, 300, 400, 500, 600]
train_reward = []
test_reward = []

for num in nums:
    log = torch.load(f'generalization-128fd/{num}levels-128fd.pt', map_location=torch.device('cpu'))

    train_reward.append(log['train_mean_reward'][-1])
    test_reward.append(log['test_mean_reward'][-1])
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(nums, train_reward)
ax.plot(nums, test_reward)
ax.set_title('Test and train reward for different levels, numSteps=256, totalSteps=2e6, featuredim=128')

plt.show()
