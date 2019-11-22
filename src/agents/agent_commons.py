import torch.nn as nn


def create_nn_layer(layer_def):
  """
  Create a PyTorch layer given a definition file.

  The data is defined with batch_size as the first dimension!

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
  else:
    raise NotImplementedError(f"The layer type '{layer_def['type']}' is not supported!")
