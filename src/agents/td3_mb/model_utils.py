import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReplayBuffer(object):

  def __init__(self, input_dim, output_dim, max_size, device):
    self.max_size = int(max_size)
    self.ptr = 0
    self.size = 0
    self.in_val = np.zeros((int(max_size), int(input_dim)))
    self.out_val = np.zeros((int(max_size), int(output_dim)))
    self.device = device  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def add(self, in_val, out_val):
    self.in_val[self.ptr] = in_val
    self.out_val[self.ptr] = out_val
    self.ptr = (self.ptr + 1) % self.max_size
    self.size = min(self.size + 1, self.max_size)

  def sample(self, batch_size):
    ind = np.random.randint(0, self.size, size=batch_size)
    in_val = torch.FloatTensor(self.in_val[ind]).to(self.device)
    out_val = torch.FloatTensor(self.out_val[ind]).to(self.device)
    return (in_val, out_val)

  def get_latest(self, n_latest):
    n = self.ptr if self.ptr >= n_latest else self.ptr
    in_data = self.in_val[self.ptr - n : self.ptr]
    out_data = self.out_val[self.ptr - n: self.ptr]
    in_batch = torch.FloatTensor(in_data).to(self.device)
    out_batch = torch.FloatTensor(out_data).to(self.device)
    return (in_batch, out_batch)