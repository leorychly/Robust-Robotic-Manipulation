import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.td3.td3_utils import create_nn_layer
from src.agents.td3.model_utils import ReplayBuffer


class ANN(nn.Module):

  def __init__(self, state_dim, action_dim, model_layer, model_replay_buffer_size, lr=0.01):
    super(ANN, self).__init__()
    # Replay Buffer
    self.buffer = ReplayBuffer(input_dim=state_dim + action_dim,
                               output_dim=state_dim,
                               max_size=model_replay_buffer_size)

    # Create Model
    model_layer[0]["n_neurons"][0] = state_dim + action_dim
    model_layer[-2]["n_neurons"][1] = state_dim
    self.layer_param = model_layer
    self.module_list = nn.ModuleList()
    for layer_def in self.layer_param:
      layer = create_nn_layer(layer_def)
      self.module_list.append(layer)

    # Optimizer
    self.criterion = torch.nn.MSELoss()
    self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

  def forward(self, x):
    for layer in self.module_list:
      x = layer(x)
    x = x.clone()
    return x

  def add_to_buffer(self, in_val, out_val):
    self.buffer.add(in_val=in_val, out_val=out_val)

  def optimize_model(self, batch_size, iterations=1000):
    for _ in range(iterations):
      in_batch, out_batch = self.buffer.sample(batch_size=batch_size)
      network_output = self.forward(x=in_batch)
      batch_loss = self.criterion(input=network_output, target=out_batch)
      self.optimizer.zero_grad()
      batch_loss.backward()
      self.optimizer.step()

  def eval(self, in_batch=None, out_batch=None, batch_size=1000):
    if in_batch == None and out_batch == None:
      in_batch, out_batch = self.buffer.get_latest(n_latest=batch_size)

    total_loss = 0
    for in_val, out_val in zip(in_batch, out_batch):
      network_output = self.forward(x=in_val)
      total_loss += self.criterion(input=network_output, target=out_val)
    avrg_loss = total_loss / len(in_batch)
    return avrg_loss
