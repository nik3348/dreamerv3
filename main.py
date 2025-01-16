import ale_py
import datetime
import torch
import gymnasium as gym

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from ptnn3.DreamerV3 import DreamerV3
from ptnn3.PrioritizedReplayBuffer import PrioritizedReplayBuffer


isHuman = False
MAX_STEPS = 20000
epochs = 150
batch_size = 1024

h_dim = 256
action_dim = 4
height = 64
width = 64

gym.register_envs(ale_py)
env = gym.make("ALE/Breakout-v5", render_mode="human" if isHuman else "rgb_array")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

y_dim, x_dim, obs_dim = env.observation_space.shape
dreamer = DreamerV3(obs_dim, h_dim, action_dim, height=height, width=width).to(device)
buffer = PrioritizedReplayBuffer(500)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)
pytorch_total_params = sum(p.numel() for p in dreamer.parameters())
print("Total parameters", pytorch_total_params)

try:
    dreamer.load_state_dict(torch.load("dreamer.pt", weights_only=True))
except FileNotFoundError:
    print("No model found")

for epoch in range(epochs):
    done = False
    steps = 0
    score = 0
    obs, info = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).to(device).permute(2, 1, 0) / 255.0

    h = torch.randn(h_dim).to(device)
    action = torch.zeros(action_dim).to(device)

    while not done and steps < MAX_STEPS:
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
        score += reward.item()
        done = torch.tensor(done, dtype=torch.float32).to(device).unsqueeze(0)

        h_next, z, z_pred, reward_pred, cont_pred, obs_pred, action_next, value = (
            dreamer(obs, h, action)
        )

        buffer.add(
            (
                obs.detach(),
                obs_pred.detach(),
                z.detach(),
                z_pred.detach(),
                reward.detach(),
                reward_pred.detach(),
                done.detach(),
                cont_pred.detach(),
            )
        )

        obs = next_obs.detach()
        h = h_next.detach()
        action = action_next.detach()

    # Train the model
    print("==============================")
    print("Starting training for epoch", epoch)
    batch, indices, weights = buffer.sample(batch_size)
    (
        obs_batch,
        obs_pred_batch,
        z_batch,
        z_pred_batch,
        reward_batch,
        reward_pred_batch,
        done_batch,
        cont_pred_batch,
    ) = zip(*batch)

    obs_batch = torch.stack(obs_batch)
    obs_pred_batch = torch.stack(obs_pred_batch)
    z_batch = torch.stack(z_batch)
    z_pred_batch = torch.stack(z_pred_batch)
    reward_batch = torch.stack(reward_batch)
    reward_pred_batch = torch.stack(reward_pred_batch)
    done_batch = torch.stack(done_batch)
    cont_pred_batch = torch.stack(cont_pred_batch)

    batch = (
        obs_batch.requires_grad_(True),
        obs_pred_batch.requires_grad_(True),
        z_batch.requires_grad_(True),
        z_pred_batch.requires_grad_(True),
        reward_batch.requires_grad_(True),
        reward_pred_batch.requires_grad_(True),
        done_batch.requires_grad_(True),
        cont_pred_batch.requires_grad_(True),
    )

    world_model_loss = dreamer.train_world_model(batch)
    actor_loss, critic_loss, td_errors = dreamer.train_actor_critic(batch[0])
    buffer.update_priorities(indices, td_errors.cpu().detach().numpy())

    dreamer.model_scheduler.step(world_model_loss)
    dreamer.actor_scheduler.step(actor_loss)
    dreamer.critic_scheduler.step(critic_loss)

    print("World Model Loss", world_model_loss)
    print("Actor Loss", actor_loss)
    print("Critic Loss", critic_loss)

    writer.add_scalar("train/World Model Loss", world_model_loss, epoch)
    writer.add_scalar("train/Actor Loss", actor_loss, epoch)
    writer.add_scalar("train/Critic Loss", critic_loss, epoch)
    writer.add_scalar("train/Total Loss", actor_loss + critic_loss, epoch)
    writer.add_scalar("train/Score", score, epoch)

    torch.save(dreamer.state_dict(), "dreamer.pt")

env.close()
