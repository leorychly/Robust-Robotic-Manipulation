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


class Box:
  """Definition of action and observation spaces in OpenAI Gym environments."""
  def __init__(self,
               bounded_above,
               bounded_below,
               dtype,
               high,
               low,
               shape):
    self.bounded_above = bounded_above
    self.bounded_below = bounded_below
    self.dtype = dtype
    self.high = high
    self.low = low
    self.shape = shape