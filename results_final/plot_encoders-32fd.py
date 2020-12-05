import pickle
import torch
import matplotlib.pyplot as plt

def read_torch(filename):
    return torch.load(filename, map_location=torch.device('cpu'))

impala = read_torch('encoder/impala-1000-32fd.pt')
nature = read_torch('encoder/nature-1000-32fd.pt')

fig, ax = plt.subplots(figsize=(12,5))

r = impala['step']

ax.plot(r, impala['train_mean_reward'], 'C0', r, impala['test_mean_reward'], 'C0--')
ax.plot(r, nature['train_mean_reward'], 'C1', r, nature['test_mean_reward'], 'C1--')
ax.set_title('Nature vs Impala. featureDim: 32')

plt.show()
