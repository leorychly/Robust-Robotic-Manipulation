import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_nn_layer(layer_def):
  """
  Create a PyTorch layer given a definition file.

  :param layer_def: dict
    Dictionary containing the layer definition.

  :return pytorch.nn.module:
    Return the corresponding PyTorch layer module.
  """
  if layer_def["type"] == "linear":
    return nn.Linear(layer_def["n_neurons"][0], layer_def["n_neurons"][1])
  elif layer_def["type"] == "relu":
    return nn.ReLU()
  elif layer_def["type"] == "tanh":
    return nn.Tanh()


class Actor(nn.Module):
  def __init__(self, state_dim, action_dim, max_action, actor_layer):
    super(Actor, self).__init__()

    actor_layer[0]["n_neurons"][0] = state_dim
    actor_layer[-2]["n_neurons"][1] = action_dim
    self.layer_param = actor_layer

    #self.layer_param = [
    #  {"type": "linear", "n_neurons": [state_dim, 256]},
    #  {"type": "relu"},
    #  {"type": "linear", "n_neurons": [256, 256]},
    #  {"type": "relu"},
    #  {"type": "linear", "n_neurons": [256, action_dim]},
    #  {"type": "tanh"}
    #]
    self.module_list = nn.ModuleList()
    for layer_def in self.layer_param:
      layer = create_nn_layer(layer_def)
      self.module_list.append(layer)
    #self.l1 = nn.Linear(self.layer[0], self.layer[1])
    #self.l2 = nn.Linear(self.layer[1], self.layer[2])
    #self.l3 = nn.Linear(self.layer[2], self.layer[3])
    self.max_action = max_action

  def forward(self, x):
    for layer in self.module_list:
      x = layer(x)
    x = x.clone() * self.max_action
    return x
    #state = F.relu(self.l1(state))
    #state = F.relu(self.l2(state))
    #state = torch.tanh(self.l3(state)) * self.max_action
    #return state


class Critic(nn.Module):
  def __init__(self, state_dim, action_dim, critic_layer):
    super(Critic, self).__init__()

    critic_layer[0]["n_neurons"][0] = state_dim + action_dim
    self.layer_param = critic_layer

    # Q1 architecture
    self.module_list_q1 = nn.ModuleList()
    for layer_def in self.layer_param:
      layer = create_nn_layer(layer_def)
      self.module_list_q1.append(layer)
    # Q2 architecture
    self.module_list_q2 = nn.ModuleList()
    for layer_def in self.layer_param:
      layer = create_nn_layer(layer_def)
      self.module_list_q2.append(layer)

    self.layer = [state_dim + action_dim, 256, 256, 1]

    ## Q1 architecture
    #self.l1 = nn.Linear(self.layer[0], self.layer[1])
    #self.l2 = nn.Linear(self.layer[1], self.layer[2])
    #self.l3 = nn.Linear(self.layer[2], self.layer[3])

    ## Q2 architecture
    #self.l5 = nn.Linear(self.layer[0], self.layer[1])
    #self.l6 = nn.Linear(self.layer[1], self.layer[2])
    #self.l7 = nn.Linear(self.layer[2], self.layer[3])


  def forward(self, state, action):
    x1 = torch.cat([state, action], 1)
    x2 = torch.cat([state, action], 1)
    for layer1, layer2 in zip(self.module_list_q1, self.module_list_q2):
      x1 = layer1(x1)
      x2 = layer1(x2)
    return x1, x2
    #x1 = F.relu(self.l1(sa))
    #x1 = F.relu(self.l2(x1))
    #x1 = self.l3(x1)
    #x2 = F.relu(self.l5(sa))
    #x2 = F.relu(self.l6(x2))
    #x2 = self.l7(x2)
    #return x1, x2

  def Q1(self, state, action):
    x = torch.cat([state, action], 1)
    for layer in self.module_list_q1:
      x = layer(x)
    return x
    #q1 = F.relu(self.l1(sa))
    #q1 = F.relu(self.l2(q1))
    #q1 = self.l3(q1)
    #return q1


class ReplayBuffer(object):

  def __init__(self, state_dim, action_dim, max_size=1e6):
    self.max_size = int(max_size)
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