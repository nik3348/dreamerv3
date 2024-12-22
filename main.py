import ale_py
import gymnasium as gym
import torch

from ptnn3.DreamerV3 import DreamerV3 


isHuman = True
epochs = 20

obs_dim = 3
x_dim = 128
y_dim = 63
z_dim = 65
h_dim = 128
action_dim = 4

dreamer = DreamerV3(obs_dim, z_dim, h_dim, action_dim, x_dim, y_dim)

# Random input tensors
obs = torch.randn(16, obs_dim, x_dim, y_dim)
h = torch.randn(16, h_dim)
action = torch.randn(16, action_dim)

# Forward pass
h_next, z_pred, reward, cont_flag, obs_next, action, value = dreamer(obs, h, action)

# Print output shapes
print(h_next.shape)
print(z_pred.shape)
print(reward.shape)
print(cont_flag.shape)
print(obs_next.shape)
print(action.shape)
print(value.shape)

# gym.register_envs(ale_py)
# env = gym.make("ALE/Breakout-v5", render_mode="human" if isHuman else "rgb_array")

# print(env.observation_space)
# print(env.action_space)

# for i in range(epochs):
#     done = False
#     obs, info = env.reset()

#     while not done:
#         action = 0

#         next_obs, reward, terminated, truncated, info = env.step(action)
#         done = terminated or truncated

# env.close()

