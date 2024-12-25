import numpy as np
import torch
from collections import deque


class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, action_next, reward, value, done):
        self.buffer.append((action_next, reward, value, done))

    def sample(self, sequence_size):
        index = np.random.choice(len(self.buffer) - sequence_size, 1)[0]

        action_nexts = []
        rewards = []
        values = []
        dones = []

        for i in range(index, index + sequence_size):
            action_next, reward, value, done = (
                self.buffer[i]
            )

            action_nexts.append(action_next)
            rewards.append(reward)
            values.append(value)
            dones.append(done)

        action_nexts = torch.stack(rewards)
        rewards = torch.stack(rewards)
        values = torch.stack(dones)
        dones = torch.stack(dones)

        return (action_nexts, rewards, values, dones)

    def __len__(self):
        return len(self.buffer)
