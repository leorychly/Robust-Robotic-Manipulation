import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from absl import logging

from src.agents.agent_commons import create_nn_layer
from src.agents.td3_mb.model_utils import ReplayBuffer


class ANN(torch.nn.Module):

  def __init__(self, input_dim, output_dim, model_param, lr, buffer_size, device):
    """
    Dense Neural Network for predicting the next state of the environemnt.

    :param state_dim: Int
      state_space + action_space
    :param action_dim: Int
      state_space
    :param model_param: Dict
      Network parameter
    :param lr: float
      Learning rate
    :param buffer_size: int
      Replay buffer size
    :param device: String
      'cpu' or 'gpu' depending on where to run the computations.
    """
    super(ANN, self).__init__()
    self.buffer = ReplayBuffer(input_dim=input_dim,
                               output_dim=output_dim,
                               max_size=buffer_size,
                               device=device)

    model_param[0]["n_neurons"][0] = input_dim
    model_param[-2]["n_neurons"][1] = output_dim
    self.layer_param = model_param
    self.module_list = nn.ModuleList()
    for layer_def in self.layer_param:
      layer = create_nn_layer(layer_def)
      self.module_list.append(layer)

    self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
    self.loss_func = torch.nn.MSELoss()

  def forward(self, x):
    for layer in self.module_list:
      x = layer(x)
    x = x.clone()
    return x

  def add_to_buffer(self, in_val, out_val):
    self.buffer.add(in_val=in_val, out_val=out_val)

  def optimize_model(self, batch_size, iterations=1):
    losses = []
    for _ in range(iterations):
      size = batch_size if self.buffer.ptr >= batch_size else self.buffer.ptr
      x, y = self.buffer.sample(batch_size=size)
      y_pred = self.forward(x)
      loss = self.loss_func(y_pred, y)
      losses.append(loss.cpu().detach().numpy())
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
    return losses

  def eval_model(self, sample_size=1000):
    with torch.no_grad():
      size = sample_size if self.buffer.ptr >= sample_size else self.buffer.ptr
      x, y = self.buffer.sample(batch_size=size)
      y = y.cpu().detach().numpy()
      y_pred = self.forward(x).cpu().detach().numpy()
      loss = np.mean(np.abs(y - y_pred))
      return loss

  def save(self, file_path):
    file_path.mkdir(parents=True, exist_ok=True)
    torch.save(self.state_dict(),
               (file_path / "td3_model").absolute().as_posix())
    torch.save(self.optimizer.state_dict(),
               (file_path / "td3_model_optimizer").absolute().as_posix())

  def load(self, file_path):
    logging.info(f"Loaded model from {file_path}")
    file_path.mkdir(parents=True, exist_ok=True)
    self.load_state_dict(torch.load((file_path / "td3_model").absolute().as_posix()))
    self.optimizer.load_state_dict(torch.load(
      (file_path / "td3_model_optimizer").absolute().as_posix()))


if __name__ == '__main__':
  torch.cuda.empty_cache()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
  y = x.pow(2) + 0.2 * torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)

  model_param = [
    {"type": "linear", "n_neurons": [0, 10]},
    {"type": "relu"},
    {"type": "linear", "n_neurons": [10, 0]},
    {"type": "tanh"}]

  net = ANN(input_dim=1, output_dim=1, model_param=model_param, lr=1e-2, buffer_size=1000, device=device)
  net = net.to(device)

  for x_val, y_val in zip(x, y):
    net.add_to_buffer(in_val=x_val, out_val=y_val)

  losses = net.optimize_model(steps=1000, is_debug=False)
  plt.plot(np.arange(len(losses)), losses, 'r-')
  plt.show()