import unittest
import numpy as np

from src.executer.base_executer import BaseExecuter
from src.executer.normal_noise_executer import NormalNoiseExecuter


class NormalNoiseExecuterTests(unittest.TestCase):
  """Tests for the Base and NormalNoise Executers."""

  def base_executer(self):
    exe = BaseExecuter(action_limits=np.array([[-1, 1], [-2, 2], [-3, 3]]))
    assert exe(np.array([-2, -2, 5])) == np.array([-1, -2, 3])

  # TODO: add Normality Test (https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/)
  def test_noise_distribution(self):
    exe = NormalNoiseExecuter(action_limits=np.array([[-1, 1], [-2, 2], [-3, 3]]))
    print(exe(
      np.array([0.5, -1, 2])
    ))


if __name__ == '__main__':
  unittest.main()
