import pickle
#import ConfigParser
#import argparse

import torch

from src.utils import make_env, Storage
from src import Encoder, PPO, Experiment, Impala

"""
def read_ini_file(file_name):
    parser = ConfigParser.ConfigParser()
    parser.read(file_name)
    confdict = {section: dict(parser.items(section)) for section in parser.sections()}

    return confdict
"""

if __name__=='__main__':
    #parser = argparse.ArgumentParser()
    #args = parser.parse_args()
    #params = read_ini_file(args["--ini_file"])

    # Hyperparameters
    params = {
        "total_steps" : 8e5,
        "num_envs" : 32,
        "num_levels" : 10,
        "num_steps" : 256,
        "num_epochs" : 3,
        "batch_size" : 256,
        "feature_dim" : 64,
        "gamma" : 0.99,
        "lmbda" : 0.975,
        "eps" : .2,
        "grad_eps" : .5,
        "value_coef" : .5,
        "entropy_coef" : .01,
        "recurrent" : True,
        "video_name" : 'vid1.mp4',
        "pickle_name" : 'test.pkl'
    }

    # Define environment
    # check the utils.py file for info on arguments
    env = make_env(params["num_envs"], num_levels=params["num_levels"])

    # Define network
    num_actions = env.action_space.n
    in_channels = 3 # RGB
    # encoder = Encoder(in_channels=in_channels, feature_dim=params["feature_dim"])
    encoder = Impala(in_channels=in_channels, feature_dim=params["feature_dim"])
    policy = PPO(encoder=encoder, feature_dim=params["feature_dim"], num_actions=num_actions)
    policy.cuda()

    # Define optimizer
    # these are reasonable values but probably not optimal
    optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4, eps=1e-5)

    # Define temporary storage
    # we use this to collect transitions during each iteration
    storage = Storage(
        obs_shape=env.observation_space.shape,
        num_steps=params["num_steps"],
        num_envs=params["num_envs"],
        gamma=params["gamma"],
        lmbda=params["lmbda"],
        feature_dim=params["feature_dim"],
        recurrent=params["recurrent"]
    )

    # Create experiment class for training and evaluation.
    exp = Experiment(params)

    # Train.
    policy, hidden_states, train_reward, test_reward = exp.train(env, policy, optimizer, storage, verbose=True, hidden_states_size=params["feature_dim"])

    with open(params['pickle_name'], 'wb') as f:
        pickle.dump({'train_reward': train_reward, 'test_reward': test_reward}, f)

    # Generate output video.
    exp.generate_video(policy, hidden_states)

    # torch.save(policy.state_dict, 'checkpoint.pt')
