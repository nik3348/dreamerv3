import ale_py
import datetime
import torch
import gymnasium as gym

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from ptnn3.DreamerV3 import DreamerV3


isHuman = False
number_of_episodes = 150
number_of_steps = 10000
batch_size = 512

h_dim = 256
action_dim = 4
height = 64
width = 64

gym.register_envs(ale_py)
env = gym.make("ALE/Breakout-v5", render_mode="human" if isHuman else "rgb_array")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

y_dim, x_dim, obs_dim = env.observation_space.shape
dreamer = DreamerV3(obs_dim, h_dim, action_dim, height=height, width=width).to(device)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)

pytorch_total_params = sum(p.numel() for p in dreamer.parameters())
print("Total parameters", pytorch_total_params)

try:
    dreamer.load_state_dict(torch.load("dreamer.pt", weights_only=True))
except FileNotFoundError:
    print("No model found")

for episode in range(number_of_episodes):
    done = False
    steps = 0
    score = 0
    obs, info = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).to(device).permute(2, 1, 0) / 255.0

    h = torch.randn(h_dim).to(device)
    action = torch.zeros(action_dim).to(device)

    trajectory = {
        "obs": [],
        "obs_pred": [],
        "z": [],
        "z_pred": [],
        "reward": [],
        "reward_pred": [],
        "done": [],
        "cont_pred": [],
        "h": [],
    }

    while not done and steps < number_of_steps:
        steps += 1
        selected_action = action.argmax().item()

        if False:
            try:
                selected_action = int(input())
            except ValueError:
                selected_action = 0

        if torch.rand(1).item() < 0.05:
            selected_action = env.action_space.sample()

        next_obs, reward, terminated, truncated, info = env.step(selected_action)
        done = terminated or truncated
        obs = transforms.Compose([transforms.Resize((height, width))])(obs)

        next_obs = (
            torch.tensor(next_obs, dtype=torch.float32).to(device).permute(2, 1, 0)
            / 255.0
        )
        reward = torch.tensor(reward, dtype=torch.float32).to(device).unsqueeze(0)
        done = torch.tensor(done, dtype=torch.float32).to(device).unsqueeze(0)

        h_next, z, z_pred, reward_pred, cont_pred, obs_pred, action_next, value = (
            dreamer(obs, h, action)
        )

        trajectory["obs"].append(obs)
        trajectory["obs_pred"].append(obs_pred)
        trajectory["z"].append(z)
        trajectory["z_pred"].append(z_pred)
        trajectory["reward"].append(reward)
        trajectory["reward_pred"].append(reward_pred)
        trajectory["done"].append(done)
        trajectory["cont_pred"].append(cont_pred)
        trajectory["h"].append(h)

        obs = next_obs
        h = h_next
        action = action_next
        score += reward.item()

    obs_batch = torch.stack(trajectory["obs"])
    obs_pred_batch = torch.stack(trajectory["obs_pred"])
    z_batch = torch.stack(trajectory["z"])
    z_pred_batch = torch.stack(trajectory["z_pred"])
    reward_batch = torch.stack(trajectory["reward"])
    reward_pred_batch = torch.stack(trajectory["reward_pred"])
    done_batch = torch.stack(trajectory["done"])
    cont_pred_batch = torch.stack(trajectory["cont_pred"])
    h_batch = torch.stack(trajectory["h"])

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

    # Train the model
    print("==============================")
    print("Steps:", steps)
    print("Starting training for epoch", episode)

    world_model_loss = dreamer.train_world_model(batch)
    actor_loss, critic_loss, td_errors = dreamer.train_actor_critic(
        obs_batch, h_batch
    )

    dreamer.model_optimizer.zero_grad()
    dreamer.actor_optimizer.zero_grad()
    dreamer.critic_optimizer.zero_grad()

    (world_model_loss).backward(retain_graph=True)
    (actor_loss).backward(retain_graph=True)
    (critic_loss).backward()

    dreamer.model_optimizer.step()
    dreamer.actor_optimizer.step()
    dreamer.critic_optimizer.step()

    dreamer.model_scheduler.step(world_model_loss)
    dreamer.actor_scheduler.step(actor_loss)
    dreamer.critic_scheduler.step(critic_loss)

    print("World Model Loss", world_model_loss)
    print("Actor Loss", actor_loss)
    print("Critic Loss", critic_loss)

    writer.add_scalar("train/World Model Loss", world_model_loss, episode)
    writer.add_scalar("train/Actor Loss", actor_loss, episode)
    writer.add_scalar("train/Critic Loss", critic_loss, episode)
    writer.add_scalar("train/Score", score, episode)

    torch.save(dreamer.state_dict(), "dreamer.pt")

env.close()
