from abc import ABCMeta, abstractmethod

class BaseObserver(metaclass=ABCMeta):
  """This agent implements the Sense-Plan-Act architecture."""

  def __call__(self, action):
    return action
