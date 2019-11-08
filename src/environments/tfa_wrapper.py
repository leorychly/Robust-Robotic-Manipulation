import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from src.commons import BoundedSpaceContinuous


class TFAWrapper(py_environment.PyEnvironment):
  """Wrapper for TensorFlow Agents (https://github.com/tensorflow/agents)

  Arguments:
    py_environment -- Base class for environment from tf_agents
  """

  def __init__(self, env, observer, executor):
    super(TFAWrapper, self).__init__()
    self.env = env
    self.observer = observer
    self.executor = executor
    self._action_spec = array_spec.BoundedArraySpec(
        shape=self.env.action_space.shape,
        dtype=self.env.action_space.dtype,
        minimum=self.env.action_space.low,
        maximum=self.env.action_space.high,
        name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=self.env.observation_space.shape,
        dtype=self.env.observation_space.dtype,
        minimum=self.env.observation_space.low,
        maximum=self.env.observation_space.high,
        name='observation')
    self._state = np.zeros(shape=self.env.observation_space.shape,
                           dtype=self.env.observation_space.dtype)
    self._episode_ended = False

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _apply_observer(self, time_step):
    #time_step.observation = self.observer(time_step.observation)
    return time_step

  def _apply_executor(self, action):
    control = self.executor(action)
    return control

  def _reset(self):
    self._state = self.env.reset()
    self._episode_ended = False
    time_step = ts.restart(self._state)
    time_step = self._apply_observer(time_step)
    return time_step

  def _step(self, action):
    action = self._apply_executor(action)
    if self._episode_ended:
      return self.reset()
    state, action, reward, self._episode_ended = self.env.step(action)
    self._state = state
    if self._episode_ended:
      time_step = ts.termination(self._state, reward=reward)
    else:
      time_step = ts.transition(self._state, reward=reward, discount=0.9)
    time_step = self._apply_observer(time_step)
    return time_step