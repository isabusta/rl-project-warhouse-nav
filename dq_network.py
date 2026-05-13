import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class QNetwork(nn.Module):
    """
    Q-Network architecture for agent to approach Q-values 
    for all actions a ∈ A: shape [n_actions]
    Possible actions: MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, PICKUP, DELIVER  
    """
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        return self.linear3(x)

class ReplayBuffer:
    def __init__(self, size):
        self.memory = deque([], maxlen=size)

    # this method samples transitions and returns tensors of each type registered in the environment step
    def sample(self, sample_size):
        sample = random.sample(self.memory, sample_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        next_masks = []
        dones = []
        for x in sample:
            states.append(x[0])
            actions.append(x[1])
            rewards.append(x[2])
            next_states.append(x[3])
            next_masks.append(x[4])
            dones.append(x[5])
        states = torch.tensor(states).to(device)
        actions = torch.tensor(actions).to(device)
        rewards = torch.tensor(rewards).to(device)
        next_states = torch.tensor(next_states).to(device)
        next_masks = torch.tensor(next_masks).to(device)
        dones = torch.tensor(dones, dtype=torch.int).to(device)
        return states, actions, rewards, next_states, next_masks, dones
    
    # add transition to the buffer
    def append(self, item):
        self.memory.append(item)

    def __len__(self):
        return len(self.memory)

