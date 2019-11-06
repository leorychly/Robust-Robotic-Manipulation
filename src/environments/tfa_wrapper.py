# Copyright (c) 2019 Leonard Rychly, Patrick Hart
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from src.commons import BoundedSpace


class TFADroneRace(py_environment.PyEnvironment):
  """Wrapper for TensorFlow Agents (https://github.com/tensorflow/agents)

  Arguments:
    py_environment -- Base class for environment from tf_agents
  """

  def __init__(self, env):
    self.env = env
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

  def _reset(self):
    self._state = self.env.reset()
    self._episode_ended = False
    return ts.restart(self._state)

  def _step(self, action):
    if self._episode_ended:
      return self.reset()
    state, action, reward, self._episode_ended = self.env.step(action)
    self._state = state
    if self._episode_ended:
      return ts.termination(self._state, reward=reward)
    else:
      return ts.transition(self._state, reward=reward, discount=0.9)
