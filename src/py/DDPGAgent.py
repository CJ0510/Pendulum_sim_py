import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# actor net
# 2 Linear layers with ReLU activation ending with a tanh activation
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


# critic net
# 2 Linear layers with ReLU activation ending with a linear activation
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, u):
        x = torch.relu(self.fc1(torch.cat([x, u], 1)))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# replay buffer
# deque with 15000 max length
# sample, push, and __len__ methods
class ReplayBuffer:
    def __init__(self, max_len=15000):
        self.buffer = deque(maxlen=max_len)
        
    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in idx])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def __len__(self):
        return len(self.buffer)
    
# DDPGAgent
# get_action, update, and __init__ methods
class DDPGAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=1e-2):
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer()
        
    def get_action(self, state):
        state = torch.FloatTensor(state).to(device)
        action = self.actor(state)
        return action.detach().cpu().numpy()
    
    def update(self, batch_size):
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        # update critic
        next_actions = self.actor_target(next_states)
        next_Q = self.critic_target(next_states, next_actions)
        target_Q = rewards + self.gamma * next_Q * (1 - dones)
        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q.detach())
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        # update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        # update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
            
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)