import sys
import pickle
import json

import torch

from src.utils import make_env, Storage
from src import Nature, PPO, TRPO, Experiment, Impala


def read_json(filename):
    """Reads json file and returns as dict.

    Args:
        filename (str): Filename.

    Returns:
        Dictionary containing json content.

    """
    with open(filename) as file_in:
        return json.load(file_in)

if len(sys.argv) != 2:
    raise Exception("Filename must be specified as argument.")

# Read parameters.
params = read_json(sys.argv[1])

# Define environment. Check utils.py file for info on arguments
env = make_env(params["num_envs"], num_levels=params["num_levels"])

# Define network
feature_dim = params['feature_dim']
num_actions = env.action_space.n
in_channels = 3 # RGB

# Define encoders
encoders = {
    "impala": Impala(params['in_channels'], params['feature_dim']),
 #   "nature": Nature(params['in_channels'], params['feature_dim'])
    }
encoder = encoders[params['encoder']]

policies = {
    "ppo": PPO(encoder, feature_dim, num_actions),
    "trpo": TRPO(encoder, feature_dim, num_actions, beta=params['beta'])
}
policy = policies[params['policy']]

policy = PPO(encoder=encoder, feature_dim=feature_dim, num_actions=num_actions)
policy.cuda()

# Define optimizer
# these are reasonable values but probably not optimal
optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4, eps=1e-5)

# Define temporary storage. We use this to collect transitions during each iteration
storage = Storage(
    obs_shape=env.observation_space.shape,
    num_steps=params["num_steps"],
    num_envs=params["num_envs"]
)

# Create experiment class for training and evaluation.
exp = Experiment(params)

# Train.
policy, train_reward, test_reward = exp.train(env, policy, optimizer, storage, verbose=True)

with open(params['pickle_name'], 'wb') as f:
    pickle.dump({'train_reward': train_reward, 'test_reward': test_reward}, f)

# Evaluate final policy.
# exp.evaluate(policy)

# Generate output video.
exp.generate_video(policy)


# torch.save(policy.state_dict, 'checkpoint.pt')
