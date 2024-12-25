import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, z_dim, feature_width=8, feature_height=8):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)

        self.fc = nn.Linear(128 * feature_width * feature_height, z_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z


class Decoder(nn.Module):
    def __init__(self, z_dim, output_dim, feature_width=8, feature_height=8):
        super(Decoder, self).__init__()
        self.feature_width = feature_width
        self.feature_height = feature_height
        self.fc = nn.Linear(z_dim, 128 * feature_width * feature_height)

        self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(
            32, output_dim, kernel_size=4, stride=2, padding=1
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 128, self.feature_width, self.feature_height)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class RSSM(nn.Module):
    def __init__(self, z_dim, h_dim, action_dim):
        super(RSSM, self).__init__()
        self.sequence = nn.GRU(z_dim + action_dim, h_dim)
        self.dynamics = nn.Linear(h_dim, z_dim)

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

        # model_state is a bundle of h the hidden state and z the latent state
        model_state = torch.cat([h, z], dim=-1)
        reward_pred = self.reward_predictor(model_state)
        cont_flag = torch.sigmoid(self.continue_predictor(model_state))

        return h_next, z_pred, reward_pred, cont_flag


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
    def __init__(
        self, input_dim, z_dim, h_dim, action_dim, input_width=64, input_height=64
    ):
        super(DreamerV3, self).__init__()
        feature_width = input_width // 8
        feature_height = input_height // 8

        self.encoder = Encoder(input_dim, z_dim, feature_width, feature_height)
        self.decoder = Decoder(z_dim, input_dim, feature_width, feature_height)
        self.rssm = RSSM(z_dim, h_dim, action_dim)
        self.actor = Actor(h_dim + z_dim, action_dim)
        self.critic = Critic(h_dim + z_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )

    def forward(self, obs, h, action):
        # Unsqueeze to add the batch dimension
        obs = obs.unsqueeze(0)
        h = h.unsqueeze(0)
        action = action.unsqueeze(0)

        z = self.encoder(obs)
        h_next, z_pred, reward_pred, cont_pred = self.rssm(z, h, action)
        obs_pred = self.decoder(z)

        model_state = torch.cat([h, z_pred], dim=-1)
        action_pred = self.actor(model_state)
        value = self.critic(model_state)

        # Squeeze to remove the batch dimension
        h_next = h_next.squeeze(0)
        z = z.squeeze(0)
        z_pred = z_pred.squeeze(0)
        reward_pred = reward_pred.squeeze(0)
        cont_pred = cont_pred.squeeze(0)
        obs_pred = obs_pred.squeeze(0)
        action_pred = action_pred.squeeze(0)
        value = value.squeeze(0)

        return (
            h_next,
            z,
            z_pred,
            reward_pred,
            cont_pred,
            obs_pred,
            action_pred,
            value,
        )

    def train_world_model(self, batch):
        (
            obs,
            obs_pred,
            z,
            z_pred,
            reward,
            reward_pred,
            done,
            cont_pred,
        ) = batch

        # Compute the prediction losses
        obs_loss = F.mse_loss(obs_pred, obs)
        reward_loss = F.mse_loss(reward_pred, reward)
        cont_loss = F.binary_cross_entropy(cont_pred, done.float())

        pred_coef = 1
        pred_loss = obs_loss + reward_loss + cont_loss

        dynamics_coef = 0.5
        dynamics_loss = torch.clamp(
            F.kl_div(z.detach(), z_pred, reduction="batchmean"), min=1.0
        )

        representation_coef = 0.1
        representation_loss = torch.clamp(
            F.kl_div(z, z_pred.detach(), reduction="batchmean"), min=1.0
        )

        total_loss = (
            pred_coef * pred_loss
            + dynamics_coef * dynamics_loss
            + representation_coef * representation_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss

    def train_actor_critic(self, batch):
        action_pred, rewards, values, continuation_flags = batch
        action_pred.requires_grad_(True)

        eta = 3e-4  # Entropy weight
        gamma = 0.997  # Discount factor
        lambda_ = 0.95  # Lambda parameter

        T = rewards.size(0)  # Number of time steps
        lambda_returns = torch.zeros_like(rewards)

        # Bootstrap with the critic's value prediction for the last state
        lambda_returns[-1] = values[-1]

        # Backward recursion to compute lambda-returns
        for t in reversed(range(T - 1)):
            bootstrap = (1 - lambda_) * values[t + 1] + lambda_ * lambda_returns[t + 1]
            lambda_returns[t] = rewards[t] + gamma * continuation_flags[t] * bootstrap

        # Stop-gradient operation
        lambda_returns = lambda_returns.detach()  # Treat λ-returns as constants

        scaling_factor = torch.quantile(lambda_returns, 0.95) - torch.quantile(
            lambda_returns, 0.05
        )
        scaling_factor = torch.clamp(scaling_factor, min=1.0)
        scaled_returns = lambda_returns / scaling_factor
        policy_gradient_loss = -torch.sum(scaled_returns * action_pred)

        policy_probs = F.softmax(action_pred, dim=-1)  # Shape [B, T, A]
        policy_log_probs = F.log_softmax(action_pred, dim=-1)  # Shape [B, T, A]
        entropy = -(policy_probs * policy_log_probs).sum(dim=-1)  # Shape [B, T]
        entropy_regularization = -eta * torch.sum(entropy)

        actor_loss = policy_gradient_loss + entropy_regularization

        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        return actor_loss

    # Convert to neural net approximate
    # Symlog predictionis sin decoder, reward predictor, and critic
    def symlog(self, x):
        return torch.sign(x) * torch.log(torch.abs(x) + 1)

    def symexp(self, x):
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
