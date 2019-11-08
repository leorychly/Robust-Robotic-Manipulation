import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
  def __init__(self, state_dim, action_dim, max_action):
    super(Actor, self).__init__()

    self.l1 = nn.Linear(state_dim, 256)
    self.l2 = nn.Linear(256, 256)
    self.l3 = nn.Linear(256, action_dim)

    self.max_action = max_action

  def forward(self, state):
    a = F.relu(self.l1(state))
    a = F.relu(self.l2(a))
    return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
  def __init__(self, state_dim, action_dim):
    super(Critic, self).__init__()

    # Q1 architecture
    self.l1 = nn.Linear(state_dim + action_dim, 256)
    self.l2 = nn.Linear(256, 256)
    self.l3 = nn.Linear(256, 1)

    # Q2 architecture
    self.l4 = nn.Linear(state_dim + action_dim, 256)
    self.l5 = nn.Linear(256, 256)
    self.l6 = nn.Linear(256, 1)


  def forward(self, state, action):
    sa = torch.cat([state, action], 1)

    q1 = F.relu(self.l1(sa))
    q1 = F.relu(self.l2(q1))
    q1 = self.l3(q1)

    q2 = F.relu(self.l4(sa))
    q2 = F.relu(self.l5(q2))
    q2 = self.l6(q2)
    return q1, q2


  def Q1(self, state, action):
    sa = torch.cat([state, action], 1)

    q1 = F.relu(self.l1(sa))
    q1 = F.relu(self.l2(q1))
    q1 = self.l3(q1)
    return q1


class ReplayBuffer(object):
  def __init__(self, state_dim, action_dim, max_size=int(1e6)):
    self.max_size = max_size
    self.ptr = 0
    self.size = 0

    self.state = np.zeros((max_size, state_dim))
    self.action = np.zeros((max_size, action_dim))
    self.next_state = np.zeros((max_size, state_dim))
    self.reward = np.zeros((max_size, 1))
    self.not_done = np.zeros((max_size, 1))

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def add(self, state, action, next_state, reward, done):
    self.state[self.ptr] = state
    self.action[self.ptr] = action
    self.next_state[self.ptr] = next_state
    self.reward[self.ptr] = reward
    self.not_done[self.ptr] = 1. - done

    self.ptr = (self.ptr + 1) % self.max_size
    self.size = min(self.size + 1, self.max_size)

  def sample(self, batch_size):
    ind = np.random.randint(0, self.size, size=batch_size)
    return (torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device))