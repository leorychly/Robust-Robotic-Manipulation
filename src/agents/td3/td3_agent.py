# This implementation is based on the code of Fujimoto, Scott and Hoof, Herke and Meger, David
# "Addressing Function Approximation Error in Actor-Critic Methods", ICML 2018
# https://github.com/sfujim/TD3/blob/master/TD3.py

import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from absl import logging
import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from src.agents.base_agent import BaseAgent
from src.agents.td3.td3_utils import Actor, Critic, ReplayBuffer


class TD3Agent(BaseAgent):
  def __init__(
    self,
    env,
    action_limits,
    observer=None,
    executer=None,
    buffer_size=10000,
    discount=0.99,
    tau=0.005,
    policy_noise=0.2,
    noise_clip=0.5,
    policy_freq=2):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    policy_noise = policy_noise * max_action
    noise_clip = noise_clip * max_action
    assert np.all(action_limits[:, 0] < action_limits[:, 1])

    super(TD3Agent, self).__init__(action_limits=action_limits,
                                   observer=observer,
                                   executer=executer)

    self.actor = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target = copy.deepcopy(self.actor)
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

    self.critic = Critic(state_dim, action_dim).to(device)
    self.critic_target = copy.deepcopy(self.critic)
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

    self.replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=buffer_size)

    self.action_dim = action_dim
    self.max_action = max_action
    self.discount = discount
    self.tau = tau
    self.policy_noise = policy_noise
    self.noise_clip = noise_clip
    self.policy_freq = policy_freq

    self.total_iter = 0
    self.evaluations = []

  # TODO: is selec_action == BaseAgent.plan? How to train?

  def plan(self, state):
    state = torch.FloatTensor(state.reshape(1, -1)).to(device)
    action = self.actor(state).cpu().data.numpy().flatten()
    control_action = self.executer(action)
    return control_action

  def eval_policy_on_env(self, eval_gym_env, eval_episodes=10):
    """Uses the observer and executer."""
    eval_gym_env.seed(int(time.time()))
    avg_reward = 0.
    for _ in range(eval_episodes):
      state, done = eval_gym_env.reset(), False
      obs = self.observer(state)
      while not done:
        action = self.plan(np.array(obs))
        state, reward, done, _ = eval_gym_env.step(action)
        obs = self.observer(state)
        avg_reward += reward

    avg_reward /= eval_episodes
    return avg_reward

  def train(self,
            env,
            seed,
            train_steps,
            initial_steps,
            model_save_path,
            results_path,
            expl_noise=0.1,
            batch_size=100,
            eval_steps=100,
            eval_freq=1000):
    # Create directory for results
    results_path.mkdir(parents=True, exist_ok=True)

    # Evaluate untrained policy
    self.evaluations.append(self.eval_policy_on_env(eval_gym_env=env,
                                            eval_episodes=eval_steps))

    state, done = env.reset(), False
    obs = self.observer(state)
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(train_steps)):

      episode_timesteps += 1

      # Select action randomly
      if t < initial_steps:
        action = env.action_space.sample()
      # Select next action according to policy
      else:
        action = (self.plan(np.array(obs))
                  + np.random.normal(0, self.max_action * expl_noise, size=self.action_dim)
                 ).clip(-self.max_action, self.max_action)

      # Perform action
      next_state, reward, done, _ = env.step(action)
      next_obs = self.observer(next_state)

      done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

      # Store data in replay buffer
      self.replay_buffer.add(obs, action, next_obs, reward, done_bool)

      obs = next_obs
      episode_reward += reward

      # Train agent after collecting data
      if t >= initial_steps:
        self._optimize(batch_size)

      if done:
        # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
        #if (t + 1) % logg_freq == 0:
        #  print(f"Total T: {t + 1} "
        #        f"Episode Num: {episode_num + 1} "
        #        f"Episode T: {episode_timesteps} "
        #        f"Reward: {episode_reward:.3f}")
        # Reset environment
        state, done = env.reset(), False
        obs = self.observer(state)
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

      # Evaluate and safe episode
      if t % eval_freq == 0:
        avg_reward = self.eval_policy_on_env(env, eval_episodes=eval_steps)

        logging.info(f"After Iteration {t}, "
                     f"Evaluation over {eval_steps} episodes, "
                     f"Average Reward: {avg_reward:.3f}")

        self.evaluations.append(avg_reward)
        np.save((results_path/"results.npy").absolute().as_posix(), self.evaluations)
        self.save(model_save_path)
        self._plot_results(results_path)

  def _optimize(self, batch_size):
    self.total_iter += 1

    # Sample replay buffer
    state, action, next_state, reward, not_done = self.replay_buffer.sample(batch_size)

    with torch.no_grad():
      # Select action according to policy and add clipped noise
      noise = (torch.randn_like(action) * self.policy_noise).clamp(
        -self.noise_clip, self.noise_clip)

      next_action = (self.actor_target(next_state) + noise).clamp(
        -self.max_action, self.max_action)

      # Compute the target Q value
      target_Q1, target_Q2 = self.critic_target(next_state, next_action)
      target_Q = torch.min(target_Q1, target_Q2)
      target_Q = reward + not_done * self.discount * target_Q

    # Get current Q estimates
    current_Q1, current_Q2 = self.critic(state, action)

    # Compute critic loss
    critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

    # Optimize the critic
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    # Delayed policy updates
    if self.total_iter % self.policy_freq == 0:

      # Compute actor losse
      actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

      # Optimize the actor
      self.actor_optimizer.zero_grad()
      actor_loss.backward()
      self.actor_optimizer.step()

      # Update the frozen target models
      for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

      for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

  def save(self, file_path):
    file_path.mkdir(parents=True, exist_ok=True)
    torch.save(self.critic.state_dict(),
               (file_path / "td3_critic").absolute().as_posix())
    torch.save(self.critic_optimizer.state_dict(),
               (file_path / "td3_critic_optimizer").absolute().as_posix())
    torch.save(self.actor.state_dict(),
               (file_path / "td3_actor").absolute().as_posix())
    torch.save(self.actor_optimizer.state_dict(),
               (file_path / "td3_actor_optimizer").absolute().as_posix())

  def load(self, file_path):
    file_path.mkdir(parents=True, exist_ok=True)
    self.critic.load_state_dict(torch.load(
      (file_path / "td3_critic").absolute().as_posix()))
    self.critic_optimizer.load_state_dict(torch.load(
      (file_path / "td3_critic_optimizer").absolute().as_posix()))
    self.actor.load_state_dict(torch.load(
      (file_path / "td3_actor").absolute().as_posix()))
    self.actor_optimizer.load_state_dict(torch.load(
      (file_path / "td3_actor_optimizer").absolute().as_posix()))

  def _plot_results(self, file_path):
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(self.evaluations)), self.evaluations)
    ax.set(xlabel="Iterations [1k]", ylabel="Average Reward",
           title="Average Reward of Evaluation during Training")
    ax.grid()
    fig.savefig((file_path / "test.png").absolute().as_posix())
    plt.close()

  def get_param(self):
    return {
      "action_dim": self.action_dim,
      "max_action": self.max_action,
      "discount": self.discount,
      "tau": self.tau,
      "policy_noise": self.policy_noise,
      "noise_clip": self.noise_clip,
      "policy_freq": self.policy_freq,
      "buffer_size": self.replay_buffer.max_size,
      "actor_layers": self.actor.layer,
      "critic_layers": self.critic.layer
    }