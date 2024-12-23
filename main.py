import ale_py
import gymnasium as gym
import torch

from ptnn3.DreamerV3 import DreamerV3
from ptnn3.PrioritizedReplayBuffer import PrioritizedReplayBuffer


isHuman = False
epochs = 1

z_dim = 64
h_dim = 128
action_dim = 4

gym.register_envs(ale_py)
env = gym.make("ALE/Breakout-v5", render_mode="human" if isHuman else "rgb_array")
world_model_buffer = PrioritizedReplayBuffer(1000)

y_dim, x_dim, obs_dim = env.observation_space.shape
dreamer = DreamerV3(obs_dim, z_dim, h_dim, action_dim, x_dim, y_dim)

for i in range(epochs):
    done = False
    obs, info = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).permute(2, 1, 0) / 255.0
    obs = obs[:, :, 2:]  # Remove two rows of the height

    h = torch.randn(h_dim)
    action = torch.zeros(action_dim)

    while not done:
        selected_action = action.argmax().item()
        next_obs, reward, terminated, truncated, info = env.step(selected_action)
        done = terminated or truncated

        next_obs = torch.tensor(next_obs, dtype=torch.float32).permute(2, 1, 0) / 255.0
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(0)

        h_next, z, z_pred, reward_pred, cont_pred, obs_pred, action_next, value = (
            dreamer(obs.unsqueeze(0), h.unsqueeze(0), action.unsqueeze(0))
        )

        # Unsqueeze to remove the batch dimension
        h_next = h_next.squeeze(0)
        z = z.squeeze(0)
        z_pred = z_pred.squeeze(0)
        reward_pred = reward_pred.squeeze(0)
        cont_pred = cont_pred.squeeze(0)
        obs_pred = obs_pred.squeeze(0)
        action_next = action_next.squeeze(0)

        world_model_buffer.add(
            (
                obs,
                obs_pred,
                z,
                z_pred,
                reward,
                reward_pred,
                done,
                cont_pred,
            )
        )

        obs = next_obs
        h = h_next
        action = action_next

    # Train the model
    batch, indices, weights = world_model_buffer.sample(1)

    obs_batch = torch.stack([item[0] for item in batch])
    obs_pred_batch = torch.stack([item[1] for item in batch])
    z_batch = torch.stack([item[2] for item in batch])
    z_pred_batch = torch.stack([item[3] for item in batch])
    reward_batch = torch.stack([item[4] for item in batch])
    reward_pred_batch = torch.stack([item[5] for item in batch])
    done_batch = torch.stack([item[6] for item in batch])
    cont_pred_batch = torch.stack([item[7] for item in batch])

    batch = (
        obs_batch,
        obs_pred_batch,
        z_batch,
        z_pred_batch,
        reward_batch,
        reward_pred_batch,
        done_batch,
        cont_pred_batch,
    )

    loss = dreamer.train_world_model(batch)
    dreamer.scheduler.step(loss)

env.close()
