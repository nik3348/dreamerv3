import ale_py
import datetime
import torch
import gymnasium as gym

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from ptnn3.ReplayBuffer import ReplayBuffer
from ptnn3.DreamerV3 import DreamerV3
from ptnn3.PrioritizedReplayBuffer import PrioritizedReplayBuffer


isHuman = False
MAX_STEPS = 2
epochs = 1

h_dim = 64
action_dim = 4

gym.register_envs(ale_py)
env = gym.make("ALE/Breakout-v5", render_mode="human" if isHuman else "rgb_array")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

y_dim, x_dim, obs_dim = env.observation_space.shape
dreamer = DreamerV3(obs_dim, h_dim, action_dim).to(device)
world_model_buffer = PrioritizedReplayBuffer(500)
actor_critic_buffer = ReplayBuffer(500)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)

try:
    dreamer.load_state_dict(torch.load("dreamer.pt", weights_only=True))
except FileNotFoundError:
    print("No model found")

for epoch in range(epochs):
    done = False
    steps = 0
    obs, info = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).to(device).permute(2, 1, 0) / 255.0

    h = torch.randn(h_dim).to(device)
    action = torch.zeros(action_dim).to(device)

    while not done and steps < MAX_STEPS:
        steps += 1
        selected_action = action.argmax().item()
        next_obs, reward, terminated, truncated, info = env.step(selected_action)
        done = terminated or truncated
        obs = transforms.Compose([transforms.Resize((32, 32))])(obs)

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

        obs = next_obs
        h = h_next
        action = action_next

    # Train the model
    print("==============================")
    print("Starting training for epoch", epoch)
    batch, indices, weights = world_model_buffer.sample(1)

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

    world_model_loss = dreamer.train_world_model(batch)
    dreamer.scheduler.step(world_model_loss)
    print("World Model Loss", world_model_loss)

    # actor_critic_loss = dreamer.train_actor_critic(batch[0])
    # print("Actor Critic Loss", actor_critic_loss)

    # writer.add_scalar('World Model Loss/train', world_model_loss, epoch)
    # writer.add_scalar('Actor Critic Loss/train', actor_critic_loss, epoch)

    torch.save(dreamer.state_dict(), "dreamer.pt")

env.close()
