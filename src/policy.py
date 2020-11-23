import torch
import torch.nn as nn

from utils import orthogonal_init


class PPO(nn.Module):
    """PPO policy"""

    def __init__(self, encoder, feature_dim, num_actions):
        super().__init__()
        self.encoder = encoder
        self.policy = orthogonal_init(nn.Linear(feature_dim, num_actions), gain=.01)
        self.value = orthogonal_init(nn.Linear(feature_dim, 1), gain=1.)

    def act(self, x):
        with torch.no_grad():
            x = x.cuda().contiguous()
            dist, value = self.forward(x)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.cpu(), log_prob.cpu(), value.cpu()

    def forward(self, x):
        x = self.encoder(x)
        logits = self.policy(x)
        value = self.value(x).squeeze(1)
        dist = torch.distributions.Categorical(logits=logits)

        return dist, value

    def pi_loss(self, log_pi, sampled_log_pi, advantage, clip=0.2):
        ratio = torch.exp(log_pi - sampled_log_pi)
        clipped_ratio = ratio.clamp(min=1.0 - clip, max=1.0 + clip)
        policy_reward = torch.min(ratio * advantage, clipped_ratio * advantage)
        return - policy_reward.mean()

    def value_loss(self, value, sampled_value, sampled_return, clip=0.2):
        clipped_value = sampled_value + (value - sampled_value).clamp(min=-clip, max=clip)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        return 0.5 * vf_loss.mean()

    def entropy_loss(self, dist):
        return torch.mean(dist.entropy())
