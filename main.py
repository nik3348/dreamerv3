import ale_py
import gymnasium as gym
import torch

from ptnn3.ReplayBuffer import ReplayBuffer
from ptnn3.DreamerV3 import DreamerV3
from ptnn3.PrioritizedReplayBuffer import PrioritizedReplayBuffer


isHuman = False
epochs = 10

z_dim = 64
h_dim = 128
action_dim = 4

gym.register_envs(ale_py)
env = gym.make("ALE/Breakout-v5", render_mode="human" if isHuman else "rgb_array")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

y_dim, x_dim, obs_dim = env.observation_space.shape
dreamer = DreamerV3(obs_dim, z_dim, h_dim, action_dim, x_dim, y_dim).to(device)
world_model_buffer = PrioritizedReplayBuffer(500)
actor_critic_buffer = ReplayBuffer(500)

try:
    dreamer.load_state_dict(torch.load("dreamer.pt", weights_only=True))
except FileNotFoundError:
    print("No model found")

for i in range(epochs):
    done = False
    steps = 0
    obs, info = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).to(device).permute(2, 1, 0) / 255.0

    h = torch.randn(h_dim).to(device)
    action = torch.zeros(action_dim).to(device)

    while not done and steps < 500:
        steps += 1
        selected_action = action.argmax().item()
        next_obs, reward, terminated, truncated, info = env.step(selected_action)
        done = terminated or truncated
        obs = obs[:, :, 2:]

        next_obs = (
            torch.tensor(next_obs, dtype=torch.float32).to(device).permute(2, 1, 0)
            / 255.0
        )
        reward = torch.tensor(reward, dtype=torch.float32).to(device).unsqueeze(0)
        done = torch.tensor(done, dtype=torch.float32).to(device).unsqueeze(0)

        h_next, z, z_pred, reward_pred, cont_pred, obs_pred, action_next, value = (
            dreamer(obs, h, action)
        )

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

        actor_critic_buffer.add(
            action_next,
            reward,
            value,
            done,
        )

        obs = next_obs
        h = h_next
        action = action_next

    # Train the model
    print("==============================")
    print("Starting training for epoch", i)
    batch, indices, weights = world_model_buffer.sample(10)

    obs_batch = []
    obs_pred_batch = []
    z_batch = []
    z_pred_batch = []
    reward_batch = []
    reward_pred_batch = []
    done_batch = []
    cont_pred_batch = []

    for item in batch:
        obs_batch.append(item[0])
        obs_pred_batch.append(item[1])
        z_batch.append(item[2])
        z_pred_batch.append(item[3])
        reward_batch.append(item[4])
        reward_pred_batch.append(item[5])
        done_batch.append(item[6])
        cont_pred_batch.append(item[7])

    obs_batch = torch.stack(obs_batch)
    obs_pred_batch = torch.stack(obs_pred_batch)
    z_batch = torch.stack(z_batch)
    z_pred_batch = torch.stack(z_pred_batch)
    reward_batch = torch.stack(reward_batch)
    reward_pred_batch = torch.stack(reward_pred_batch)
    done_batch = torch.stack(done_batch)
    cont_pred_batch = torch.stack(cont_pred_batch)

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

    print("Training world model")
    loss = dreamer.train_world_model(batch)
    dreamer.scheduler.step(loss)

    print("Training actor critic")
    batch = actor_critic_buffer.sample(4)
    dreamer.train_actor_critic(batch)

    torch.save(dreamer.state_dict(), "dreamer.pt")

env.close()
