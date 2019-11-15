import unittest
import numpy as np

from src.commons import Box

from src.observer.base_observer import BaseObserver
from src.observer.noise_observer import NoiseObserver
from src.observer.shift_observer import ShiftObserver


class ObserverTests(unittest.TestCase):
  """Tests for the all Observer."""

  #@unittest.skip("Utility function for other tests.")
  def _create_obs_space(self):
    obs_space = Box(
      bounded_below=np.array([True, False, True, False]),
      bounded_above = np.array([False, False, True, True]),
      low = np.array([-1,     -np.inf, 0, -np.inf]),
      high = np.array([np.inf, np.inf, 1,  5]),
      dtype = np.float32,
      shape=(4,))
    mock_obs = [np.array([-0.9, 0, 0.8, -10]),
                np.array([-1.1, 10, 1.1, 4.99])]
    return obs_space, mock_obs

  def test_base_observer(self):
    obs_space, mock_obs = self._create_obs_space()
    exe = BaseObserver(observation_space=obs_space)
    assert np.allclose(exe(mock_obs[0]), mock_obs[0])
    assert np.allclose(exe(mock_obs[1]), np.array([-1, 10, 1, 4.99]))

  # TODO: add Normality Test (https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/)
  def test_noise_observer(self):
    obs_space, mock_obs = self._create_obs_space()
    exe = NoiseObserver(observation_space=obs_space,
                        noise_scale=0.01)
    assert np.all(exe(mock_obs[0]) != mock_obs[0])

  def test_shift_observer(self):
    obs_space, mock_obs = self._create_obs_space()
    exe_pos = ShiftObserver(observation_space=obs_space,
                            shift_scale=0.01)
    exe_neg = ShiftObserver(observation_space=obs_space,
                            shift_scale=-0.01)
    assert np.allclose(exe_pos(mock_obs[0]), np.array([-0.89, 0.01, 0.81, -9.99]))
    assert np.allclose(exe_neg(mock_obs[1]), np.array([-1, 9.99, 1, 4.98]))


if __name__ == '__main__':
  unittest.main()
