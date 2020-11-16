import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio

from utils import make_env, Storage, orthogonal_init

# Hyperparameters
total_steps = 8e4
num_envs = 32
num_levels = 10
num_steps = 256
num_epochs = 3
batch_size = 512
eps = .2
grad_eps = .5
value_coef = .5
entropy_coef = .01


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Encoder(nn.Module):
    def __init__(self, in_channels, feature_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=1024, out_features=feature_dim), nn.ReLU()
        )
        self.apply(orthogonal_init)

    def forward(self, x):
        return self.layers(x)


class Policy(nn.Module):
    def __init__(self, encoder, feature_dim, num_actions):
        super().__init__()
        self.encoder = encoder
        self.policy = orthogonal_init(nn.Linear(feature_dim, num_actions), gain=.01)
        self.value = orthogonal_init(nn.Linear(feature_dim, 1), gain=1.)
        self.clip = eps
        self.c1 = value_coef
        self.c2 = entropy_coef

    def act(self, x):
        with torch.no_grad():
        x = x.cuda().contiguous()
        dist, value = self.forward(x)
        value = value.cpu()
        action = dist.sample()
        log_prob = dist.log_prob(action).cpu()
        action = action.cpu()

        return action, log_prob, value

    def forward(self, x):
        x = self.encoder(x)
        logits = self.policy(x)
        value = self.value(x).squeeze(1)
        dist = torch.distributions.Categorical(logits=logits)

        return dist, value

    def pi_loss(self, log_pi, sampled_log_pi, advantage):
        ratio = torch.exp(log_pi - sampled_log_pi)
        clipped_ratio = ratio.clamp(min=1.0 - self.clip, max=1.0 + self.clip)
        policy_reward = torch.min(ratio * advantage, clipped_ratio * advantage)
        return - policy_reward.mean()

    def value_loss(self, value, sampled_value, sampled_return):
        clipped_value = sampled_value + (value - sampled_value).clamp(min=-self.clip, max=self.clip)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        return self.c1 * 0.5 * vf_loss.mean()

    def entropy_loss(self, dist):
        return self.c2 * torch.mean(dist.entropy())


# Define environment
# check the utils.py file for info on arguments
env = make_env(num_envs, num_levels=num_levels)
print('Observation space:', env.observation_space)
print('Action space:', env.action_space.n)

# Define network
feature_dim = 32
num_actions = env.action_space.n
in_channels = 3 # RGB
encoder = Encoder(in_channels=in_channels, feature_dim=feature_dim)
policy = Policy(encoder=encoder, feature_dim=feature_dim, num_actions=num_actions)
policy.cuda()

# Define optimizer
# these are reasonable values but probably not optimal
optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4, eps=1e-5)

# Define temporary storage
# we use this to collect transitions during each iteration
storage = Storage(
    env.observation_space.shape,
    num_steps,
    num_envs
)

# Run training
obs = env.reset()
step = 0
while step < total_steps:

    # Use policy to collect data for num_steps steps
    policy.eval()
    for _ in range(num_steps):
        # Use policy
        action, log_prob, value = policy.act(obs)

        # Take step in environment
        next_obs, reward, done, info = env.step(action)

        # Store data
        storage.store(obs, action, reward, done, info, log_prob, value)

        # Update current observation
        obs = next_obs

    # Add the last observation to collected data
    _, _, value = policy.act(obs)
    storage.store_last(obs, value)

    # Compute return and advantage
    storage.compute_return_advantage()

    # Optimize policy
    policy.train()
    for epoch in range(num_epochs):

        # Iterate over batches of transitions
        generator = storage.get_generator(batch_size)
        for batch in generator:
        b_obs, b_action, b_log_prob, b_value, b_returns, b_advantage = batch

        # Get current policy outputs
        new_dist, new_value = policy(b_obs)
        new_log_prob = new_dist.log_prob(b_action)

        # Clipped policy objective
        pi_loss = policy.pi_loss(new_log_prob, b_log_prob, b_advantage)

        # Clipped value function objective
        value_loss = policy.value_loss(new_value, b_value, b_returns)

        # Entropy loss
        entropy_loss = policy.entropy_loss(new_dist)

        # Backpropagate losses
        loss = pi_loss + value_loss - entropy_loss
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_eps)

        # Update policy
        optimizer.step()
        optimizer.zero_grad()

    # Update stats
    step += num_envs * num_steps
    print(f'Step: {step}\tMean reward: {storage.get_reward()}')

print('Completed training!')
torch.save(policy.state_dict, 'checkpoint.pt')

# Make evaluation environment
eval_env = make_env(num_envs, start_level=num_levels, num_levels=num_levels)
obs = eval_env.reset()

total_reward = []

# Evaluate policy
policy.eval()
for _ in range(512):

    # Use policy
    action, log_prob, value = policy.act(obs)

    # Take step in environment
    obs, reward, done, info = eval_env.step(action)
    total_reward.append(torch.Tensor(reward))

# Calculate average return
total_reward = torch.stack(total_reward).sum(0).mean(0)
print('Average return:', total_reward)
