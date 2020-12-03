import torch
import imageio
import numpy as np

from .utils import make_env


class Experiment:
    def __init__(self, params):
        self.total_steps = params["total_steps"]
        self.num_steps = params["num_steps"]
        self.num_epochs = params["num_epochs"]
        self.batch_size = params["batch_size"]
        self.grad_eps = params["grad_eps"]
        self.num_envs = params["num_envs"]
        self.num_levels = params["num_levels"]

    def train(self, env, policy, optimizer, storage, verbose=False):
        """Train policy."""
        steps = []
        train_mean_reward = []
        train_min_reward = []
        train_max_reward = []
        test_mean_reward = []
        test_min_reward = []
        test_max_reward = []
        pi_losses = []
        value_losses = []
        entropy_losses = []

        # Run training - we need env, policy (with encoder), optimizer and storage.
        obs = env.reset()
        step = 0
        while step < self.total_steps:

            # Use policy to collect data for num_steps steps
            policy.eval()
            for _ in range(self.num_steps):
                # Use policy
                action, log_prob, value, dist = policy.act(obs)

                # Take step in environment
                next_obs, reward, done, info = env.step(action)

                # Store data
                storage.store(obs, action, reward, done, info, log_prob, value, dist.probs)

                # Update current observation
                obs = next_obs

            # Add the last observation to collected data
            _, _, value, _ = policy.act(obs)
            storage.store_last(obs, value)

            # Compute return and advantage
            storage.compute_return_advantage()

            # Optimize policy
            policy.train()

            pi_loss = 0
            value_loss = 0
            entropy_loss = 0

            for epoch in range(self.num_epochs):

                # Iterate over batches of transitions
                generator = storage.get_generator(self.batch_size)
                for batch in generator:
                    loss, loss_components = policy.loss(batch)
                    pi_loss += loss_components[0]
                    value_loss += loss_components[1]
                    entropy_loss += loss_components[2]
                    loss.backward()

                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), self.grad_eps)

                    # Update policy
                    optimizer.step()
                    optimizer.zero_grad()

            pi_loss /= self.num_epochs * self.batch_size
            value_loss /= self.num_epochs * self.batch_size
            entropy_loss /= self.num_epochs * self.batch_size

            # Update stats for logging.
            step += self.num_envs * self.num_steps
            steps.append(step)

            train_mean_rew, train_min_rew, train_max_rew = self.evaluate(policy, start_level=0, num_levels=self.num_levels)
            train_mean_reward.append(train_mean_rew)
            train_min_reward.append(train_min_rew)
            train_max_reward.append(train_max_rew)

            test_mean_rew, test_min_rew, test_max_rew = self.evaluate(policy, start_level=self.num_levels, num_levels=0)
            test_mean_reward.append(test_mean_rew)
            test_min_reward.append(test_min_rew)
            test_max_reward.append(test_max_rew)

            pi_losses.append(pi_loss)
            value_losses.append(value_loss)
            entropy_losses.append(entropy_loss)

            if verbose:
                print(f'Step: {step}\tMean train reward: {train_mean_rew}', flush=True)
                print(f'\tMean test reward: {test_mean_rew}', flush=True)

        log = {
            'step': steps,
            'train_mean_reward': train_mean_reward,
            'train_min_reward': train_min_reward,
            'train_max_reward': train_max_reward,
            'test_mean_reward': test_mean_reward,
            'test_min_reward': test_min_reward,
            'test_max_reward': test_max_reward,
            'pi_loss': pi_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
        }

        return policy, log

    def evaluate(self, policy, start_level, num_levels):
        """Evaluate performance of policy on new environment."""

        # Make evaluation environment.
        env = make_env(self.num_envs, start_level=start_level, num_levels=num_levels)
        obs = env.reset()

        total_reward = []

        # Evaluate policy.
        policy.eval()

        workers_finished = np.zeros((self.num_envs), dtype=bool)
        while not np.all(workers_finished):

            # Use policy.
            action, _, _, __dict__ = policy.act(obs)

            # Take step in environment.
            obs, reward, done, _ = env.step(action)
            for i in range(self.num_envs):
                if done[i]:
                    workers_finished[i] = True
                if workers_finished[i]:
                    reward[i] = 0

            total_reward.append(torch.Tensor(reward))

        # Calculate average return
        mean_reward = torch.stack(total_reward).sum(0).mean(0)
        min_reward = torch.stack(total_reward).sum(0).min(0).values
        max_reward = torch.stack(total_reward).sum(0).max(0).values

        return mean_reward, min_reward, max_reward

    def generate_video(self, policy, filename, start_level, num_levels, framecount=512):
        """Generate .mp4 video."""

        # Make evaluation environment.
        env = make_env(1, start_level=start_level, num_levels=num_levels)
        obs = env.reset()

        frames = []

        # Evaluate policy
        policy.eval()

        for _ in range(framecount):

            # Use policy.
            action, _, _, _ = policy.act(obs)

            # Take step in environment.
            env.step(action)

            # Render environment and store.
            frame = (torch.Tensor(env.render(mode='rgb_array')) * 255.).byte()
            frames.append(frame)

        # Save frames as video.
        frames = torch.stack(frames)
        imageio.mimsave(filename, frames, fps=25)
