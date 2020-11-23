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
        self.video_name = params["video_name"]

    def train(self, env, policy, optimizer, storage):
        """Train our policy."""

        # Run training - we need env, policy (with encoder), optimizer and storage.
        obs = env.reset()
        step = 0
        while step < self.total_steps:

            # Use policy to collect data for num_steps steps
            policy.eval()
            for _ in range(self.num_steps):
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
            for epoch in range(self.num_epochs):

                # Iterate over batches of transitions
                generator = storage.get_generator(self.batch_size)
                for batch in generator:
                    loss = policy.loss(batch)
                    loss.backward()

                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), self.grad_eps)

                    # Update policy
                    optimizer.step()
                    optimizer.zero_grad()

            # Update stats
            step += self.num_envs * self.num_steps
            print(f'Step: {step}\tMean train reward: {storage.get_reward()}')
            print(f'\tMean test reward: {self.evaluate(policy)}')
            
        print('Completed training!')
        torch.save(policy.state_dict, 'checkpoint.pt')

        return policy


    def evaluate(self, policy):
        """Evaluate performance of policy on new environment."""

        # Make evaluation environment.
        env = make_env(self.num_envs, start_level=self.num_levels, num_levels=self.num_levels)
        obs = env.reset()

        total_reward = []

        # Evaluate policy.
        policy.eval()
        
        workers_finished = np.zeros((self.num_envs), dtype=bool)
        while not np.all(workers_finished):

            # Use policy.
            action, _, _ = policy.act(obs)

            # Take step in environment.
            obs, reward, done, _ = env.step(action)
            for i in range(self.num_envs):
                if done[i]:
                    workers_finished[i] = True
                if workers_finished[i]:
                    reward[i] = 0

            total_reward.append(torch.Tensor(reward))

        # Calculate average return
        total_reward = torch.stack(total_reward).sum(0).mean(0)

        return total_reward

    def generate_video(self, policy):
        """Generate .mp4 video."""
        # Make evaluation environment.
        env = make_env(1, start_level=self.num_levels, num_levels=self.num_levels)
        obs = env.reset()

        frames = []

        # Evaluate policy
        policy.eval()
        
        for _ in range(512):

            # Use policy.
            action, _, _ = policy.act(obs)

            # Take step in environment.
            env.step(action)

            # Render environment and store.
            frame = (torch.Tensor(env.render(mode='rgb_array'))*255.).byte()
            frames.append(frame)

        # Save frames as video.
        frames = torch.stack(frames)
        imageio.mimsave(self.video_name, frames, fps=25)