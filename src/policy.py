import torch
import torch.nn as nn

from .utils import orthogonal_init


class PPO(nn.Module):
    """PPO policy"""
    def __init__(self, encoder, feature_dim, num_actions, value_coef=0.5, entropy_coef=0.01):
        super().__init__()
        self.encoder = encoder
        self.policy = orthogonal_init(nn.Linear(feature_dim, num_actions), gain=.01)
        self.value = orthogonal_init(nn.Linear(feature_dim, 1), gain=1.)
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def act(self, x):
        """Class act method."""
        with torch.no_grad():
            x = x.cuda().contiguous()
            dist, value = self.forward(x)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.cpu(), log_prob.cpu(), value.cpu(), dist

    def forward(self, x):
        """Class forward method."""
        x = self.encoder(x)
        logits = self.policy(x)
        value = self.value(x).squeeze(1)
        dist = torch.distributions.Categorical(logits=logits)

        return dist, value

    def pi_loss(self, log_pi, sampled_log_pi, advantage, clip=0.2):
        """Computes the clipped policy loss."""
        ratio = torch.exp(log_pi - sampled_log_pi)
        clipped_ratio = ratio.clamp(min=1.0 - clip, max=1.0 + clip)
        policy_reward = torch.min(ratio * advantage, clipped_ratio * advantage)
        policy_loss = -policy_reward.mean()
        return policy_loss

    def value_loss(self, value, sampled_value, sampled_return, clip=0.2):
        """Computes the clipped value loss."""
        clipped_value = sampled_value + (value - sampled_value).clamp(min=-clip, max=clip)
        vf_loss = torch.max((value - sampled_return)**2, (clipped_value - sampled_return)**2)
        return vf_loss.mean()

    def entropy_loss(self, dist):
        """Computes negative mean entropy bonus from a distribution."""
        return -torch.mean(dist.entropy())

    def loss(self, batch):
        """Returns the PPO loss of a given batch iteration."""
        b_obs, b_action, b_log_prob, b_value, b_returns, b_advantage, _ = batch

        # Get current policy outputs
        new_dist, new_value = self(b_obs)
        new_log_prob = new_dist.log_prob(b_action)

        # Clipped policy objective
        pi_loss = self.pi_loss(new_log_prob, b_log_prob, b_advantage)

        # Clipped value function objective
        value_loss = self.value_loss(new_value, b_value, b_returns)

        # Entropy loss
        entropy_loss = self.entropy_loss(new_dist)

        # Total loss.
        loss = pi_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        return loss, (pi_loss, value_loss, entropy_loss)


class TRPO(nn.Module):
    """ TRPO POLICY """
    def __init__(self, encoder, feature_dim, num_actions, beta=0.5):
        super().__init__()
        self.encoder = encoder
        self.policy = orthogonal_init(nn.Linear(feature_dim, num_actions), gain=.01)
        self.value = orthogonal_init(nn.Linear(feature_dim, 1), gain=1.)
        self.beta = beta  # For KL

    def act(self, x):
        with torch.no_grad():
            x = x.cuda().contiguous()
            dist, value = self.forward(x)
            value = value.cpu()
            action = dist.sample()
            log_prob = dist.log_prob(action).cpu()
            action = action.cpu()

        return action, log_prob, value, dist

    def forward(self, x):
        x = self.encoder(x)
        logits = self.policy(x)
        value = self.value(x).squeeze(1)
        dist = torch.distributions.Categorical(logits=logits)

        return dist, value

    def loss(self, batch):
        b_obs, b_action, b_log_prob, _, _, b_advantage, b_dist = batch

        # Get current policy outputs
        new_dist, _ = self(b_obs)
        new_log_prob = new_dist.log_prob(b_action)

        ratio = torch.exp(new_log_prob - b_log_prob)
        policy_reward = ratio * b_advantage
        
        # Policy loss.
        pi_loss = -policy_reward.mean()

        # KL-divergence loss.
        kl_loss = self.kl(new_dist.probs, b_dist)

        # Total loss.
        loss = pi_loss + self.beta * kl_loss

        # Return loss and tuple of three components.
        return loss, (pi_loss, kl_loss, 0)

    def kl(self, new_dist, old_dist):
        """
        https://stats.stackexchange.com/questions/72611/kl-divergence-between-two-categorical-multinomial-distributions-gives-negative-v
        """
        # To avoid pi(a|s) = 0 and log(0) = -inf
        epsilon = 10e-6
        new_dist = torch.clamp(new_dist, epsilon, 1)
        old_dist = torch.clamp(old_dist, epsilon, 1)

        kl_mean = torch.sum(old_dist * (torch.log(old_dist) - torch.log(new_dist)), dim=1).mean()
        
        return kl_mean
