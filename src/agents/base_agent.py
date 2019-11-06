from abc import ABCMeta, abstractmethod

from src.executer.base_executer import BaseExecuter
from src.observer.base_observer import BaseObserver

class BaseAgent(metaclass=ABCMeta):
  """This agent implements the Sense-Plan-Act architecture."""

  def __init__(self,
               action_space,
               observer=None,
               executer=None):
    self.action_space = action_space
    self.observer = observer or BaseObserver()
    self.executer = executer or BaseExecuter()

  def sense(self, state):
    obs  = self.observer(state)
    return obs

  def plan(self, state):
    obs = self.sense(state)
    action = self._plan(obs)
    control = self.act(action)
    return action

  def plan_random(self, obs=None):
    action = self.action_space.sample()
    return action

  def act(self, action):
    control = self.executer(action)
    return control

  def _plan(self, obs):
    action = self.plan_random()
    return action