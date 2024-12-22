import ale_py
import gymnasium as gym
import torch

from ptnn3.DreamerV3 import DreamerV3
from ptnn3.PrioritizedReplayBuffer import PrioritizedReplayBuffer


isHuman = False
epochs = 20

z_dim = 65
h_dim = 128
action_dim = 4

gym.register_envs(ale_py)
env = gym.make("ALE/Breakout-v5", render_mode="human" if isHuman else "rgb_array")
buffer = PrioritizedReplayBuffer(1000)

y_dim, x_dim, obs_dim = env.observation_space.shape
dreamer = DreamerV3(obs_dim, z_dim, h_dim, action_dim, x_dim, y_dim)

for i in range(epochs):
    done = False
    obs, info = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).permute(2, 1, 0) / 255.0

    h = torch.randn(h_dim).unsqueeze(0)
    action = torch.zeros(action_dim).unsqueeze(0)

    while not done:
        next_obs, reward, terminated, truncated, info = env.step(2)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).permute(2, 1, 0) / 255.0

        done = terminated or truncated

        h_next, z_pred, reward, cont_flag, obs_pred, action_next, value = dreamer(
            obs.unsqueeze(0), h, action
        )

        # buffer.add((obs, next_obs, action, reward, done))
        h = h_next
        action = action_next
        obs = next_obs

env.close()
