from abc import ABCMeta, abstractmethod

class BaseExecuter(metaclass=ABCMeta):
  """This agent implements the Sense-Plan-Act architecture."""

  def __init__(self, max_action=None):
    self.max_action=max_action

  def __call__(self, action):
    return action
