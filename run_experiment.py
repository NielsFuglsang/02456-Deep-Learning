import sys
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
name = sys.argv[1]

# Read parameters.
params = read_json('params/'+name+'.json')

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
    num_envs=params["num_envs"],
    act_shape=num_actions
)

# Create experiment class for training and evaluation.
exp = Experiment(params)

# Train.
policy, log = exp.train(env, policy, optimizer, storage, verbose=True)

# Save policy
torch.save(policy.state_dict, 'exp/'+name+'-policy.pt')

# Save logging.
torch.save(log, 'exp/'+name+'.pt')

# Generate output video for test levels.
test_video_name = 'exp/'+name+'-test.mp4'
exp.generate_video(policy, test_video_name, start_level=params['num_levels'], num_levels=0)

# Generate output video for train levels.
train_video_name = 'exp/'+name+'-train.mp4'
exp.generate_video(policy, train_video_name, start_level=0, num_levels=params['num_levels'])
