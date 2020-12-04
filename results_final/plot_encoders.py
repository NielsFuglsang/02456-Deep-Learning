import pickle
import torch
import matplotlib.pyplot as plt

def read_torch(filename):
    return torch.load(filename, map_location=torch.device('cpu'))

def read_pickle(filename):
    with open(filename, 'rb') as f:
        p = pickle.load(f)
    return p

impala = read_torch('encoder/impala-1000-levels-32-featuredim.pt')
nature = read_pickle('encoder/nature-1000-levels-32-featuredim.pt')

fig, ax = plt.subplots(figsize=(12,5))

r = impala['step']

ax.plot(r, impala['train_mean_reward'], 'C0', r, impala['test_mean_reward'], 'C0--')
ax.plot(r, nature['test_mean_reward'], 'C1', r, nature['train_mean_reward'], 'C1--')
ax.set_title('Nature vs Impala. featureDim: 32')

plt.show()
