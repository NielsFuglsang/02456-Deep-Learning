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

# Define train environment. Check utils.py file for info on arguments
env = make_env(params["num_envs"], start_level=0, num_levels=params["num_levels"])

# Define network parameters
feature_dim = params['feature_dim']
num_actions = env.action_space.n
in_channels = 3 # RGB

# Define encoders
encoders = {
    "impala": Impala(in_channels, feature_dim),
    "nature": Nature(in_channels, feature_dim)
    }
encoder = encoders[params['encoder']]

policies = {
    "ppo": PPO(encoder, feature_dim, num_actions),
    "trpo": TRPO(encoder, feature_dim, num_actions, beta=params['beta'])
}
policy = policies[params['policy']]
policy.cuda()

# Define optimizer.
optimizer = torch.optim.Adam(policy.parameters(), lr=params['lr'], eps=1e-5)

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

# Save policy
torch.save(policy.state_dict, 'exp/'+params['name']+'.pt')

# Save train and test reward.
with open('exp/'+params['name']+'.pkl', 'wb') as f:
    pickle.dump({'train_reward': train_reward, 'test_reward': test_reward}, f)

# Generate output video for test levels.
test_video_name = 'exp/'+params['name']+'-test.mp4'
exp.generate_video(policy, test_video_name)

# Generate output video for test levels.
train_video_name = 'exp/'+params['name']+'-train.mp4'
exp.generate_video(policy, train_video_name, start_level=0, num_levels=params['num_levels'])
