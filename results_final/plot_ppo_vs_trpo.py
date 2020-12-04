import torch
import matplotlib.pyplot as plt

def read_torch(filename):
    return torch.load(filename, map_location=torch.device('cpu'))


ppo = read_torch('ppo-vs-trpo/ppo-400.pt')
trpo = read_torch('ppo-vs-trpo/trpo-400.pt')

fig, ax = plt.subplots(figsize=(12,5))

r = ppo['step']

ax.plot(r, trpo['train_mean_reward'], 'C0', r, trpo['test_mean_reward'], 'C0--')
ax.plot(r, ppo['test_mean_reward'], 'C1', r, ppo['train_mean_reward'], 'C1--')
ax.set_title('TRPO vs PPO')

plt.show()
