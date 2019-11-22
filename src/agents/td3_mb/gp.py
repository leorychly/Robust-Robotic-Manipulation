import numpy as np
import matplotlib.pyplot as plt
import torch

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive, SVI, TraceMeanField_ELBO

from src.agents.td3_mb.model_utils import ReplayBuffer


class GaussianProcess(pyro.nn.PyroModule):

  def __init__(self, input_dim, output_dim, buffer_size=10000, lr=0.005):
    self.lr = lr
    self.input_dim = input_dim
    self.buffer = ReplayBuffer(input_dim=input_dim,
                               output_dim=output_dim,
                               max_size=buffer_size)
    pyro.clear_param_store()
    self.gpr1 = None
    self.gpr2 = None

  def save_to_buffer(self, x_in, x_out):
    self.buffer.add(x_in, x_out)

  def _init_gp(self, x_in, x_out):
    kernel1 = gp.kernels.RBF(input_dim=self.input_dim)
    kernel2 = gp.kernels.RBF(input_dim=self.input_dim)

    gpr1 = gp.models.GPRegression(
      x_in, None, kernel1, noise=torch.tensor(1e-3), mean_function=lambda x: x)
    gpr1.kernel.variance = pyro.nn.PyroSample(dist.Exponential(1))
    gpr1.kernel.lengthscale = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))

    gpr2 = gp.models.GPRegression(
      torch.zeros(len(x_in)), x_out, kernel2, noise=torch.tensor(1e-3))
    gpr2.kernel.variance = pyro.nn.PyroSample(dist.Exponential(1))
    gpr2.kernel.lengthscale = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))
    return gpr1, gpr2

  def forward(self, x_in):
    mean1, var1 = self.gpr1.model(x_in)
    x1 = pyro.sample("h", dist.Normal(mean1, var1.sqrt()))
    mean2, var2 = self.gpr2.model(x1)
    x2 = pyro.sample("h", dist.Normal(mean2, var2.sqrt()))
    return x2

  def fit_replay_buffer(self, n_samples):
    x_in_batch, x_out_batch = self.buffer.sample(batch_size=n_samples)
    # Setup GP models
    self.gpr1, self.gpr2 = self._init_gp(x_in=x_in_batch, x_out=x_out_batch)
    # Fit GPs
    hmc_kernel = NUTS(self, max_tree_depth=5)
    mcmc = MCMC(hmc_kernel, num_samples=n_samples)
    training_progress = mcmc.run()
    return training_progress


if __name__ == '__main__':
  N = 20
  X = torch.rand(N)
  y = (X >= 0.5).float() + torch.randn(N) * 0.05
  plt.plot(X.numpy(), y.numpy(), "kx")

  gp = GaussianProcess(input_dim=1, output_dim=1)

  for x_val, y_val in zip(X, y):
    gp.save_to_buffer(x_in=x_val, x_out=y_val)

  gp.fit_replay_buffer(n_samples=100)

  #first_half_loss = np.mean(np.array(losses[:int(len(losses) / 2)]))
  #second_half_loss = np.mean(np.array(losses[int(len(losses) / 2):]))
  #assert first_half_loss * 2 >= second_half_loss

  #mean, cov = gp.gp(X, full_cov=False)
  #print(f"Shape of mean {mean.shape}, cov {cov.shape}")

  #gp.forward()

  #plt.plot(np.arange(len(losses)), losses)
  plt.show()