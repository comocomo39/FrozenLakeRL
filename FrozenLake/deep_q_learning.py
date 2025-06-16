import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

from config import (
    DQN_LR, DQN_BUFFER_SIZE,
    DQN_BATCH_SIZE, DQN_GAMMA,
    DQN_TARGET_UPDATE, DQN_MIN_REPLAY,
    DQN_EPS_START, DQN_EPS_END, DQN_EPS_DECAY
)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.uint8)
        )

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    """
    Rete LINEARE: equivale a tabular Q.
    """
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.linear1 = nn.Linear(state_size, hidden_size)
        self.relu = nn.ReLU()
        self.hidden_layer1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, action_size)
        
        self.linearX = nn.Linear(state_size, action_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.hidden_layer1(x)
        x = self.relu(x)
        return self.linear2(x)

class DQNAgent:
    def __init__(self, state_size, action_size, device=None, seed: int = 0):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        random.seed(seed)

        self.state_size = state_size
        self.action_size = action_size

        self.policy_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayBuffer(capacity=DQN_BUFFER_SIZE)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=DQN_LR)

        self.steps_done = 0


    # ---- ESERCIZIO SELECT_ACTION 3 ----------------------------------
    
    def select_action(self, state):
        if isinstance(state, (int, np.integer)):
            state = np.eye(self.state_size, dtype=np.float32)[state]
        self.steps_done += 1
        eps = DQN_EPS_END + (DQN_EPS_START - DQN_EPS_END) * math.exp(-1. * self.steps_done / DQN_EPS_DECAY)
        if random.random() < eps:
            return random.randrange(self.action_size)
        with torch.no_grad():
            x = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            return self.policy_net(x).argmax(dim=1).item()
    
    # ---------------------------------------------------------------

    def store_transition(self, state, action, reward, next_state, done):
        if isinstance(state, (int, np.integer)):
            state = np.eye(self.state_size, dtype=np.float32)[state]
        if isinstance(next_state, (int, np.integer)):
            next_state = np.eye(self.state_size, dtype=np.float32)[next_state]
        self.memory.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.memory) < DQN_MIN_REPLAY:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(DQN_BATCH_SIZE)
        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)

        q_vals = self.policy_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
        target = rewards + DQN_GAMMA * next_q * (1 - dones)

        loss = F.smooth_l1_loss(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % DQN_TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    # alias per compatibilità con MCTS
    def choose_action(self, state):
        return self.select_action(state)
    
    def greedy_action(self, state):
        # Se lo stato è un intero, trasformalo in one-hot
        if isinstance(state, (int, np.integer)):
            state = np.eye(self.state_size, dtype=np.float32)[state]
        # Prepara il tensore e disattiva i gradienti
        state_v = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            # usa policy_net, non model
            q_vals = self.policy_net(state_v)
        return int(q_vals.argmax(dim=1).item())




