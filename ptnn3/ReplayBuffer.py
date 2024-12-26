import numpy as np
import torch
from collections import deque


class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, obs, action_next, reward, value, done):
        self.buffer.append((obs, action_next, reward, value, done))

    def sample(self, sequence_size):
        index = np.random.choice(len(self.buffer) - sequence_size, 1)[0]

        obss = []
        action_nexts = []
        rewards = []
        values = []
        dones = []

        for i in range(index, index + sequence_size):
            obs, action_next, reward, value, done = self.buffer[i]

            obss.append(obs)
            action_nexts.append(action_next)
            rewards.append(reward)
            values.append(value)
            dones.append(done)

        obss = torch.stack(obss)
        action_nexts = torch.stack(action_nexts)
        rewards = torch.stack(rewards)
        values = torch.stack(values)
        dones = torch.stack(dones)

        return (obss, action_nexts, rewards, values, dones)

    def __len__(self):
        return len(self.buffer)
