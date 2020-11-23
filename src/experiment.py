import torch
import imageio

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

    def train(self, env, policy, optimizer, storage):
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
            print(f'Step: {step}\tMean reward: {storage.get_reward()}')

        print('Completed training!')
        torch.save(policy.state_dict, 'checkpoint.pt')

        return policy

    def evaluate(self, policy):
        # Make evaluation environment
        eval_env = make_env(self.num_envs, start_level=self.num_levels, num_levels=self.num_levels)
        obs = eval_env.reset()

        frames = []
        total_reward = []

        # Evaluate policy
        policy.eval()
        for _ in range(512):

            # Use policy
            action, log_prob, value = policy.act(obs)

            # Take step in environment
            obs, reward, done, info = eval_env.step(action)
            total_reward.append(torch.Tensor(reward))

            # Render environment and store
            frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()
            frames.append(frame)

        # Calculate average return
        total_reward = torch.stack(total_reward).sum(0).mean(0)
        print('Average return:', total_reward)

        # Save frames as video
        frames = torch.stack(frames)
        imageio.mimsave('vid1.mp4', frames, fps=25)