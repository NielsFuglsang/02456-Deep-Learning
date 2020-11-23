import torch

from src.utils import make_env, Storage
from src import Encoder, PPO, Experiment

# Hyperparameters
params = {
    "total_steps" : 8e4,
    "num_envs" : 32,
    "num_levels" : 10,
    "num_steps" : 256,
    "num_epochs" : 3,
    "batch_size" : 512,
    "eps" : .2,
    "grad_eps" : .5,
    "value_coef" : .5,
    "entropy_coef" : .01,
}


# Define environment
# check the utils.py file for info on arguments
env = make_env(params["num_envs"], num_levels=params["num_levels"])

# Define network
feature_dim = 32
num_actions = env.action_space.n
in_channels = 3 # RGB
encoder = Encoder(in_channels=in_channels, feature_dim=feature_dim)
policy = PPO(encoder=encoder, feature_dim=feature_dim, num_actions=num_actions)
policy.cuda()

# Define optimizer
# these are reasonable values but probably not optimal
optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4, eps=1e-5)

# Define temporary storage
# we use this to collect transitions during each iteration
storage = Storage(
    env.observation_space.shape,
    params["num_steps"],
    params["num_envs"]
)


# Create experiment class for training and evaluation.
exp = Experiment(params)

# Train.
policy = exp.train(env, policy, optimizer, storage)

# Evaluate policy.
exp.evaluate(policy)