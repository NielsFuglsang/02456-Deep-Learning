import pickle
import torch
import matplotlib.pyplot as plt

def read_torch(filename):
    return torch.load(filename, map_location=torch.device('cpu'))

impala = read_torch('encoder-128fd/impala-400-128fd.pt')
nature = read_torch('encoder-128fd/nature-400-128fd.pt')

fig, ax = plt.subplots(nrows=2, ncols= 1, figsize=(12,5))

r = impala['step']

ax[0].plot(r, impala['train_mean_reward'], 'C0', r, nature['train_mean_reward'], 'C1')
ax[1].plot(r, impala['test_mean_reward'], 'C0', r, nature['test_mean_reward'], 'C1')
ax[0].set_title('Nature vs Impala. featureDim: 128')

plt.show()
