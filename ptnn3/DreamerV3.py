import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, z_dim, input_width=64, input_height=64):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)

        self.feature_width = input_width // 8
        self.feature_height = input_height // 8

        self.fc = nn.Linear(128 * self.feature_width * self.feature_height, z_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z


class Decoder(nn.Module):
    def __init__(self, z_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(z_dim, 128 * 8 * 8)
        self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(
            32, output_dim, kernel_size=4, stride=2, padding=1
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 128, 8, 8)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class RSSM(nn.Module):
    def __init__(self, z_dim, h_dim, action_dim):
        super(RSSM, self).__init__()
        self.sequence = nn.GRU(z_dim + action_dim, h_dim)
        self.dynamics = nn.Linear(h_dim, z_dim + action_dim)

        self.reward_predictor = nn.Linear(h_dim + z_dim, 1)
        self.continue_predictor = nn.Linear(h_dim + z_dim, 1)

    def sample_latent(self, mean, log_std):
        std = torch.exp(0.5 * log_std)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, z, h, action):
        input = torch.cat([z, action], dim=-1)
        h_next, _ = self.sequence(input.unsqueeze(0), h.unsqueeze(0))
        h_next = h_next.squeeze(0)

        # dynamics tries to predict the latent state from the previous hidden state
        z_pred = self.dynamics(h)

        # latent_state is the bundle of h the hidden state and z the latent state
        latent_state = torch.cat([h, z], dim=-1)
        reward = self.reward_predictor(latent_state)
        cont_flag = torch.sigmoid(self.continue_predictor(latent_state))

        return h_next, z_pred, reward, cont_flag


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        return action


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class DreamerV3(nn.Module):
    def __init__(self, input_dim, z_dim, h_dim, action_dim, input_width=None, input_height=None):
        super(DreamerV3, self).__init__()
        self.encoder = Encoder(input_dim, z_dim, input_width, input_height)
        self.decoder = Decoder(z_dim, input_dim)
        self.rssm = RSSM(z_dim, h_dim, action_dim)
        self.actor = Actor(h_dim, action_dim)
        self.critic = Critic(h_dim)

    def forward(self, obs, h, action):
        z = self.encoder(obs)
        h_next, z_pred, reward, cont_flag = self.rssm(z, h, action)
        obs_next = self.decoder(z)

        action = self.actor(h_next)
        value = self.critic(h_next)

        return h_next, z_pred, reward, cont_flag, obs_next, action, value
