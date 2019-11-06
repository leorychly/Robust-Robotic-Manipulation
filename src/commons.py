import numpy as np

class BoundedSpaceContinuous:
  """Information about the action or observation space required for the OpenAI Gym environment """
  def __init__(self, dim, low, high, dtype=np.float32):
    self.shape = (dim,)
    self.low = low
    self.high = high
    self.n = dim
    self.dtype = dtype


class BoundedSpaceDiscrete:
  """Information about the action or observation space required for the OpenAI Gym environment """
  def __init__(self, dim, dtype=np.int):
    self.shape = (dim,)
    self.n = dim
    self.dtype = dtype
