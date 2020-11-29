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

    def act(self, x, hidden_states=None):
        """Class act method."""
        with torch.no_grad():
            x = x.cuda().contiguous()
            hidden_states = (hidden_states[0].cuda.contiguous(),
                            hidden_states[0].cuda.contiguous())
            dist, value, hidden_states = self.forward(x, hidden_states=hidden_states)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.cpu(), log_prob.cpu(), value.cpu(), (hidden_states[0].cpu(), hidden_states[1].cpu())

    def forward(self, x, hidden_states=None):
        """Class forward method."""
        x, hidden_states = self.encoder(x, hidden_states)
        logits = self.policy(x)
        value = self.value(x).squeeze(1)
        dist = torch.distributions.Categorical(logits=logits)

        return dist, value, hidden_states

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
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        return vf_loss.mean()

    def entropy_loss(self, dist):
        """Computes negative mean entropy bonus from a distribution."""
        return -torch.mean(dist.entropy())

    def loss(self, batch):
        """Returns the PPO loss of a given batch iteration."""
        b_obs, b_action, b_log_prob, b_value, b_returns, b_advantage, b_hidden_states = batch
        # Get current policy outputs.
        new_dist, new_value, _ = self(b_obs, b_hidden_states)

        new_log_prob = new_dist.log_prob(b_action)

        # Clipped policy objective.
        pi_loss = self.pi_loss(new_log_prob, b_log_prob, b_advantage)

        # Clipped value function objective.
        value_loss = self.value_loss(new_value, b_value, b_returns)

        # Entropy loss.
        entropy_loss = self.entropy_loss(new_dist)

        return pi_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss


class TRPO(nn.Module):
    """ TRPO POLICY """
    def __init__(self, encoder, feature_dim, num_actions, beta):
        super().__init__()
        self.encoder = encoder
        self.policy = orthogonal_init(nn.Linear(feature_dim, num_actions), gain=.01)
        self.value = orthogonal_init(nn.Linear(feature_dim, 1), gain=1.)
        # TRPO
        self.beta = beta # For KL

    def act(self, x):
        with torch.no_grad():
            # x = x.cuda().contiguous()
            x = x.contiguous()
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

    def loss(self, log_pi, sampled_log_pi, advantage, new_dist, old_dist):
        ratio = torch.exp(log_pi - sampled_log_pi)
        policy_reward = ratio * advantage
        return - (policy_reward.mean() - self.beta * self.kl(new_dist.probs, old_dist.probs))

    def kl(self, new_dist, old_dist):
        """
        https://stats.stackexchange.com/questions/72611/kl-divergence-between-two-categorical-multinomial-distributions-gives-negative-v
        """
        # To avoid pi(a|s) = 0 and log(0) = -inf
        epsilon = 10e-6
        new_dist = torch.clamp(new_dist, epsilon,1)
        old_dist = torch.clamp(new_dist, epsilon,1)
        
        kl_mean = torch.sum(new_dist*(torch.log(new_dist)-torch.log(old_dist)),dim=1).mean()
        return kl_mean
