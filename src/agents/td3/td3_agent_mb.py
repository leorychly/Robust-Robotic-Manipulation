import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from absl import logging
from gym import wrappers
import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nRunning computation on: '{device}'\n")

from src.agents.base_agent import BaseAgent
from src.agents.td3.td3_utils import Actor, Critic, ReplayBuffer
from src.agents.td3.ann import ANN


class TD3AgentMB(BaseAgent):
  def __init__(
    self,
    env,
    actor_layer,
    critic_layer,
    actor_lr,
    critic_lr,
    observer,
    executer,
    buffer_size,
    discount,
    tau,
    policy_noise,
    noise_clip,
    policy_freq,
    model_layer,
    model_lr,
    model_replay_buffer_size,
    **unused_kwargs):

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    policy_noise = policy_noise * max_action
    noise_clip = noise_clip * max_action

    super(TD3AgentMB, self).__init__(observation_space=env.observation_space,
                                   action_space=env.action_space,
                                   observer=observer,
                                   executer=executer)

    self.replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=buffer_size)

    self.actor = Actor(state_dim, action_dim, max_action, actor_layer)  # .to(device)
    self.actor = self.actor.to(device)
    self.actor_target = copy.deepcopy(self.actor)
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

    self.critic = Critic(state_dim, action_dim, critic_layer)  # .to(device)
    self.critic = self.critic.to(device)
    self.critic_target = copy.deepcopy(self.critic)
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    self.env_model = ANN(state_dim=state_dim,
                         action_dim=action_dim,
                         model_layer=model_layer,
                         model_replay_buffer_size=model_replay_buffer_size,
                         lr=model_lr)
    self.env_model = self.env_model.to(device)

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
    state = torch.FloatTensor(state.reshape(1, -1))  # .to(device)
    state = state.to(device)
    action = self.actor(state).cpu().data.numpy().flatten()
    control_action = self.executer(action)
    return control_action

  def eval_policy_on_env(self, eval_gym_env, eval_episodes=10, seed=None):
    """Uses the observer and executer."""
    if not seed:
      eval_gym_env.seed(seed)
    else:
      eval_gym_env.seed(int(time.time()))
    avg_reward = 0.
    for i in range(eval_episodes):
      state, done = eval_gym_env.reset(), False
      obs = self.observer(state)
      step = 0
      #while not done and step < max_steps:
      while not done:
        action = self.plan(np.array(obs))
        state, reward, done, _ = eval_gym_env.step(action)
        obs = self.observer(state)
        avg_reward += reward
        step += 1

    avg_reward /= eval_episodes
    return avg_reward

  def train(self,
            env,
            train_steps,
            initial_steps,
            model_save_path,
            results_path,
            expl_noise,
            batch_size,
            eval_steps,
            model_batch_size,
            eval_freq=1000):
    # Create directory for results
    results_path.mkdir(parents=True, exist_ok=True)

    # Evaluate untrained policy
    self.evaluations.append(self.eval_policy_on_env(
      eval_gym_env=env,
      eval_episodes=eval_steps))

    state, done = env.reset(), False
    obs = self.observer(state)
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    t0 = time.time()
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
      self.env_model.add_to_buffer(in_val=np.concatenate((obs, action)), out_val=next_obs)

      obs = next_obs
      episode_reward += reward

      # Train agent after collecting data
      if t >= initial_steps:
        self._optimize(batch_size)
        self.env_model.optimize_model(batch_size=model_batch_size)

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
        avrg_loss = self.env_model.eval()

        logging.info(f"After Iteration {t}, "
                     f"Evaluation over {eval_steps} episodes, "
                     f"Avrg Reward: {avg_reward:.3f}, "
                     f"Avrg Model loss: {avrg_loss}, "
                     f"({time.time() - t0:.2f} sec)")
        t0 = time.time()

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
    fig.savefig((file_path / "training_reward.png").absolute().as_posix())
    plt.close()

  def _save_as_gif(self, env, save_path, episodes=1000):
    # TODO: does not work yet. ffmpeg bug...
    env = wrappers.Monitor(env, save_path)
    avg_reward = 0.
    for i in range(episodes):
      state, done = env.reset(), False
      obs = self.observer(state)
      while not done:
        action = self.plan(np.array(obs))
        state, reward, done, _ = env.step(action)
        obs = self.observer(state)
        avg_reward += reward
    avg_reward /= episodes
    return avg_reward

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
      "actor_layer_param": self.actor.layer_param,  # TODO add layer_param froma ctior
      "critic_layer_param": self.critic.layer_param
    }