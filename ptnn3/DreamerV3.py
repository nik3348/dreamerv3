import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, z_dim, height=32, width=32):
        super(Encoder, self).__init__()
        # TODO: Change to kernal size 3
        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, z_dim, kernel_size=4, stride=2, padding=1)

        self.norm1 = nn.LayerNorm(32)
        self.norm2 = nn.LayerNorm(64)
        self.norm3 = nn.LayerNorm(z_dim)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(z_dim * (height // 8) * (width // 8), 256)
        self.fc2 = nn.Linear(256, z_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x.permute(0, 2, 3, 1))
        x = F.silu(x.permute(0, 3, 1, 2))

        x = self.conv2(x)
        x = self.norm2(x.permute(0, 2, 3, 1))
        x = F.silu(x.permute(0, 3, 1, 2))

        x = self.conv3(x)
        x = self.norm3(x.permute(0, 2, 3, 1))
        x = F.silu(x.permute(0, 3, 1, 2))

        x = self.flatten(x)
        x = F.silu(self.fc1(x))
        z = self.fc2(x)
        return z


class Decoder(nn.Module):
    def __init__(self, h_dim, z_dim, output_dim, height, width):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.height = height
        self.width = width

        self.fc1 = nn.Linear(h_dim + z_dim, 256)
        self.fc2 = nn.Linear(256, z_dim * (height // 8) * (width // 8))

        self.conv1 = nn.ConvTranspose2d(z_dim, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(
            32, output_dim, kernel_size=4, stride=2, padding=1
        )

        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(32)

    def forward(self, z):
        x = F.silu(self.fc1(z))
        x = F.silu(self.fc2(x))
        x = x.view(x.size(0), self.z_dim, (self.height // 8), (self.width // 8))

        x = self.conv1(x)
        x = self.norm1(x.permute(0, 2, 3, 1))
        x = F.silu(x.permute(0, 3, 1, 2))

        x = self.conv2(x)
        x = self.norm2(x.permute(0, 2, 3, 1))
        x = F.silu(x.permute(0, 3, 1, 2))

        x = self.conv3(x)
        x = F.sigmoid(x)

        return x


class RSSM(nn.Module):
    def __init__(self, input_dim, h_dim):
        super(RSSM, self).__init__()
        self.sequence = nn.GRU(input_dim, h_dim)

    def forward(self, input, h):
        h_next, _ = self.sequence(input.unsqueeze(0), h.unsqueeze(0))
        h_next = h_next.squeeze(0)

        return h_next


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.silu(self.fc1(state))
        x = F.silu(self.fc2(x))
        action = self.fc3(x)
        return action


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, state):
        x = F.silu(self.fc1(state))
        x = F.silu(self.fc2(x))
        value = self.fc3(x)
        return value


class DreamerV3(nn.Module):
    def __init__(
        self,
        input_dim,
        h_dim,
        action_dim,
        latent_dim=32,
        num_categories=32,
        height=32,
        width=32,
    ):
        super(DreamerV3, self).__init__()
        self.h_dim = h_dim
        self.latent_dim = latent_dim
        self.num_categories = num_categories
        z_dim = latent_dim * num_categories

        self.encoder = Encoder(input_dim, z_dim, height, width)
        self.decoder = Decoder(h_dim, z_dim, input_dim, height, width)
        self.rssm = RSSM(z_dim + action_dim, h_dim)
        self.dynamics_predictor = nn.Linear(h_dim, z_dim)
        self.reward_predictor = nn.Linear(h_dim + z_dim, 1)
        self.continue_predictor = nn.Linear(h_dim + z_dim, 1)

        self.actor = Actor(h_dim + z_dim, action_dim)
        self.critic = Critic(h_dim + z_dim)

        nn.init.zeros_(self.reward_predictor.weight)
        nn.init.zeros_(self.reward_predictor.bias)

        self.model_params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.rssm.parameters())
            + list(self.reward_predictor.parameters())
            + list(self.continue_predictor.parameters())
        )

        self.model_optimizer = torch.optim.Adam(self.model_params, lr=1e-4)
        self.model_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.model_optimizer, mode="min", factor=0.1, patience=10
        )

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.actor_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.actor_optimizer, mode="min", factor=0.1, patience=10
        )

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
        self.critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_optimizer, mode="min", factor=0.1, patience=10
        )

    def forward(self, obs, h, action):
        # Unsqueeze to add the batch dimension
        obs = obs.unsqueeze(0)
        h = h.unsqueeze(0)
        action = action.unsqueeze(0)

        z = self.encoder(obs)
        z_sample = self.sample_latent(z)

        h_next = self.rssm(torch.cat([z_sample, action], dim=-1), h)
        model_state = torch.cat([z_sample, h], dim=-1)

        reward_pred = self.reward_predictor(model_state)
        cont_pred = torch.sigmoid(self.continue_predictor(model_state))
        obs_pred = self.decoder(model_state)
        action_pred = self.actor(model_state)
        value = self.critic(model_state)

        # z_pred is from current h
        z_pred = self.dynamics_predictor(h)

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

        dynamics_coef = 1
        dynamics_kl_div = self.compute_kl_divergence(z.detach(), z_pred).mean()
        dynamics_loss = torch.clamp(dynamics_kl_div, min=1.0)

        representation_coef = 0.1
        representation_kl_div = self.compute_kl_divergence(z, z_pred.detach()).mean()
        representation_loss = torch.clamp(representation_kl_div, min=1.0)

        total_loss = (
            pred_coef * pred_loss
            + dynamics_coef * dynamics_loss
            + representation_coef * representation_loss
        )

        return total_loss, obs_loss, reward_loss, cont_loss

    def train_actor_critic(self, obs, h):
        eta = 3e-4  # Entropy weight
        gamma = 0.997  # Discount factor
        lambda_ = 0.95  # Lambda parameter

        z = self.encoder(obs)

        action_t = []
        value_t = []
        reward_t = []
        cont_t = []

        T = 16
        for t in range(T):
            z_sample = self.sample_latent(z)
            model_state = torch.cat([z_sample, h], dim=-1)

            action = self.actor(model_state)
            value = self.critic(model_state)
            reward_pred = self.reward_predictor(model_state)
            cont_pred = torch.sigmoid(self.continue_predictor(model_state))

            h = self.rssm(torch.cat([z_sample, action], dim=-1), h)
            z = self.dynamics_predictor(h)

            # Squeeze since the output is a single value (b, 1) -> (b)
            action_t.append(action)
            value_t.append(value.squeeze(-1))
            reward_t.append(reward_pred.squeeze(-1))
            cont_t.append(cont_pred.squeeze(-1))

        action_t = torch.stack(action_t)
        value_t = torch.stack(value_t)
        reward_t = torch.stack(reward_t)
        cont_t = torch.stack(cont_t)

        # Bootstrap with the critic's value prediction for the last state
        lambda_returns = torch.zeros_like(reward_t)
        lambda_returns[-1] = value_t[-1]

        # Backward recursion to compute lambda-returns
        for t in reversed(range(T - 1)):
            bootstrap = (1 - lambda_) * value_t[t + 1] + lambda_ * lambda_returns[t + 1]
            lambda_returns[t] = reward_t[t] + gamma * cont_t[t] * bootstrap

        # Stop-gradient operation
        lambda_returns = lambda_returns.detach()  # Treat λ-returns as constants

        scaling_factor = torch.quantile(lambda_returns, 0.95) - torch.quantile(
            lambda_returns, 0.05
        )
        scaling_factor = torch.clamp(scaling_factor, min=1.0)
        scaled_returns = lambda_returns / scaling_factor
        policy_gradient_loss = -torch.sum(scaled_returns, dim=0).mean()

        policy_probs = F.softmax(action_t, dim=-1)  # Shape [T, B, A]
        policy_probs = self.mix_probabilities(policy_probs)
        policy_log_probs = F.log_softmax(action_t, dim=-1)  # Shape [T, B, A]
        entropy = -(policy_probs * policy_log_probs).sum(dim=-1)  # Shape [T, B]
        entropy_regularization = -eta * torch.sum(entropy, dim=0).mean()

        actor_loss = policy_gradient_loss + entropy_regularization
        critic_loss = F.mse_loss(value_t, lambda_returns)

        return (
            actor_loss,
            critic_loss,
            torch.sum(value_t - lambda_returns, dim=0).mean(),
        )

    def compute_kl_divergence(self, p_logits, q_logits):
        p_logits = p_logits.view(-1, self.latent_dim, self.num_categories)
        q_logits = q_logits.view(-1, self.latent_dim, self.num_categories)

        q_probs = F.softmax(q_logits, dim=-1)
        q_probs = self.mix_probabilities(q_probs)

        q_log_probs = F.log_softmax(q_logits, dim=-1)
        p_log_probs = F.log_softmax(p_logits, dim=-1)

        kl_per_category = q_probs * (q_log_probs - p_log_probs)
        kl_per_latent = kl_per_category.sum(dim=-1)

        kl_loss = kl_per_latent.sum(dim=-1)
        return kl_loss.mean()

    def sample_latent(self, z):
        z = z.view(-1, self.latent_dim, self.num_categories)

        probs = F.softmax(z, dim=-1)
        probs = self.mix_probabilities(probs)

        z_hard = torch.argmax(probs, dim=-1)
        sample = F.one_hot(z_hard, num_classes=self.num_categories).float()

        sample = (sample + probs) - probs.detach()
        sample = sample.view(sample.size(0), -1)
        return sample

    def mix_probabilities(self, probs, mix_ratio=0.01):
        uniform_probs = torch.full_like(probs, 1.0 / self.num_categories)
        mixed_probs = (1 - mix_ratio) * probs + mix_ratio * uniform_probs
        return mixed_probs

    # Convert to neural net approximate
    # Symlog prediction is in decoder, reward predictor, and critic
    def symlog(self, x):
        return torch.sign(x) * torch.log(torch.abs(x) + 1)

    def symexp(self, x):
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
