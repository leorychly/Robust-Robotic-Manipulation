import unittest
import numpy as np

from src.commons import Box

from src.executer.base_executer import BaseExecuter
from src.executer.noise_executer import NoiseExecuter
from src.executer.shift_executer import ShiftExecuter


class ObserverTests(unittest.TestCase):
  """Tests for the all Observer."""

  def _create_action_space(self):
    action_space = Box(
      bounded_below=np.array([True, False, True, False]),
      bounded_above = np.array([False, False, True, True]),
      low = np.array([-1,     -np.inf, 0, -np.inf]),
      high = np.array([np.inf, np.inf, 1,  5]),
      dtype = np.float32,
      shape=(4,))
    mock_actions = [np.array([-0.9, 0, 0.8, -10]),
                np.array([-1.1, 10, 1.1, 4.99])]
    return action_space, mock_actions

  def test_base_executer(self):
    action_space, mock_actions = self._create_action_space()
    exe = BaseExecuter(action_space=action_space)
    assert np.allclose(exe(mock_actions[0]), mock_actions[0])
    assert np.allclose(exe(mock_actions[1]), np.array([-1, 10, 1, 4.99]))

  # TODO: add Normality Test (https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/)
  def test_noise_executer(self):
    action_space, mock_actions = self._create_action_space()
    exe = NoiseExecuter(action_space=action_space,
                        noise_scale=0.01)
    assert np.all(exe(mock_actions[0]) != mock_actions[0])

  def test_shift_executer(self):
    action_space, mock_actions = self._create_action_space()
    exe_pos = ShiftExecuter(action_space=action_space,
                            shift_scale=0.01)
    exe_neg = ShiftExecuter(action_space=action_space,
                            shift_scale=-0.01)
    assert np.allclose(exe_pos(mock_actions[0]), np.array([-0.89, 0.01, 0.81, -9.99]))
    assert np.allclose(exe_neg(mock_actions[1]), np.array([-1, 9.99, 1, 4.98]))


if __name__ == '__main__':
  unittest.main()
