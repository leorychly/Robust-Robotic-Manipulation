import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from src.agents.td3_mb.ann import ANN
from src.agents.agent_commons import create_nn_layer


class RNN(ANN):

  def __init__(self, input_dim, output_dim, model_param, lr, buffer_size, device):
    """
    Recurrent Neural Network for predicting the next state of the environemnt.

    :param input_dim: Int
      state_space + action_space
    :param output_dim: Int
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
    REC_LAYERS_SIZE = 128
    NUM_REC_LAYERS = 1
    super(RNN, self).__init__(input_dim, output_dim, model_param, lr, buffer_size, device)

    model_param[0]["n_neurons"][0] = input_dim
    ffn_output_dim = model_param[-2]["n_neurons"][1]

    self.layer_param = model_param
    self.module_list = nn.ModuleList()
    for layer_def in self.layer_param:
      layer = create_nn_layer(layer_def)
      self.module_list.append(layer)

    self.rnn = nn.RNN(input_size=ffn_output_dim,
                      hidden_size=REC_LAYERS_SIZE,
                      num_layers=NUM_REC_LAYERS,
                      batch_first=True)
    self.h_state = np.zeros(REC_LAYERS_SIZE)
    self.out = nn.Linear(REC_LAYERS_SIZE, 1)

  def forward(self, x):
    """
    Forward pass of the network.

    h_state:          (n_layers, batch, hidden_size)
    r_out:            (batch, time_step, hidden_size)

    :param x:
      Network input:  (batch, time_step, input_size)
    :return:
    """
    for layer in self.module_list:
      x = layer(x)
    x = x.clone()  # TODO: why clone() ? Why note detatch?

    r_out, self.h_state = self.rnn(x, self.h_state)

    outs = []  # save all predictions
    for time_step in range(r_out.size(1)):  # calculate output for each time step
      outs.append(self.out(r_out[:, time_step, :]))
    return torch.stack(outs, dim=1)


if __name__ == '__main__':
  TIME_STEP = 10  # rnn time step
  INPUT_SIZE = 1  # rnn input size
  LR = 0.02  # learning rate

  # show data
  steps = np.linspace(0, np.pi * 2, 100, dtype=np.float32)  # float32 for converting torch FloatTensor
  x_np = np.sin(steps)
  y_np = np.cos(steps)
