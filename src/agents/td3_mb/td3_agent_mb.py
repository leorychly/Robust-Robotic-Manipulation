import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from absl import logging
from gym import wrappers
import torch
import torch.nn.functional as F

from src.agents.base_agent import BaseAgent
from src.agents.td3.td3_utils import Actor, Critic, ReplayBuffer
from src.agents.td3_mb.ann import ANN


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
    device,
    use_model=False,
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

    self.actor = Actor(state_dim, action_dim, max_action, actor_layer)
    self.actor = self.actor.to(device)
    self.actor_target = copy.deepcopy(self.actor)
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

    self.critic = Critic(state_dim, action_dim, critic_layer)
    self.critic = self.critic.to(device)
    self.critic_target = copy.deepcopy(self.critic)
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    self.env_model = ANN(input_dim=state_dim + action_dim,
                         output_dim=state_dim,
                         model_param=model_layer,
                         lr=model_lr,
                         buffer_size=model_replay_buffer_size,
                         device=device)
    self.env_model = self.env_model.to(device)

    self.action_dim = action_dim
    self.max_action = max_action
    self.discount = discount
    self.tau = tau
    self.policy_noise = policy_noise
    self.noise_clip = noise_clip
    self.policy_freq = policy_freq
    self.use_model = use_model
    self.device = device

    self.total_iter = 0
    self.evaluations_agent = []
    self.evaluations_model = []
    self.model_pred_errors = []

  # TODO: is selec_action == BaseAgent.plan? How to train?

  def plan(self, obs, prev_obs=None, prev_action=None):
    obs = torch.FloatTensor(obs.reshape(1, -1))
    obs = obs.to(self.device)
    prev_obs = torch.FloatTensor(prev_obs)
    prev_obs = prev_obs.to(self.device)
    prev_action = torch.FloatTensor(prev_action)
    prev_action = prev_action.to(self.device)
    pred_error = -1
    if self.use_model:
      obs, pred_error = self._correct_observation(obs=obs,
                                                  prev_obs=prev_obs,
                                                  prev_action=prev_action)
    action = self.actor(obs).cpu().data.numpy().flatten()
    control_action = self.executer(action)
    return control_action, pred_error

  def _correct_observation(self, obs, prev_obs, prev_action):
    THRESH_PCT = 0.2
    obs_pred = self.env_model(torch.cat((prev_obs, prev_action)))
    obs_pred = obs_pred.reshape(1, -1)

    obs_delta = torch.abs(obs - obs_pred)
    pred_error = torch.mean((obs - obs_pred)**2).detach().cpu().numpy()
    mask_lst = obs_delta.detach().cpu().numpy() > obs_pred.detach().cpu().numpy() * THRESH_PCT
    mask = torch.BoolTensor(mask_lst)
    mask = mask.to(self.device)
    obs.masked_scatter_(mask, obs_pred[mask])
    return obs.to(self.device), pred_error

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
      prev_action = eval_gym_env.action_space.sample()
      prev_obs = obs
      step = 0
      #while not done and step < max_steps:
      while not done:
        action, pred_error = self.plan(obs=np.array(obs),
                                       prev_obs=prev_obs,
                                       prev_action=prev_action)
        state, reward, done, _ = eval_gym_env.step(action)
        prev_action = action
        prev_obs = obs
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
    self.evaluations_agent.append(self.eval_policy_on_env(
      eval_gym_env=env,
      eval_episodes=eval_steps))

    state, done = env.reset(), False
    obs = self.observer(state)
    prev_obs = obs
    prev_action = env.action_space.sample()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    model_errors = []
    t0 = time.time()
    for t in range(int(train_steps)):

      episode_timesteps += 1

      # Select action randomly
      if t < initial_steps:
        action = env.action_space.sample()
        model_errors.append(0.)
      # Select next action according to policy
      else:
        action, pred_error = self.plan(obs=np.array(obs), prev_obs=prev_obs, prev_action=prev_action)
        model_errors.append(pred_error.flatten())
        action += np.random.normal(0, self.max_action * expl_noise, size=self.action_dim)
        action = action.clip(-self.max_action, self.max_action)

      # Perform action
      next_state, reward, done, _ = env.step(action)
      next_obs = self.observer(next_state)

      done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

      # Store data in replay buffer
      self.replay_buffer.add(obs, action, next_obs, reward, done_bool)
      self.env_model.add_to_buffer(in_val=np.concatenate((obs, action)), out_val=next_obs)
      prev_obs = obs
      obs = next_obs
      prev_action = action
      episode_reward += reward

      # Train agent after collecting data
      self.env_model.optimize_model(batch_size=model_batch_size)
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
        avrg_loss = self.env_model.eval_model()

        logging.info(f"After Iteration {t}, "
                     f"Avrg Reward ({eval_steps} ep): {avg_reward:.3f}, "
                     f"Avrg (abs) Model Prediction Error : {avrg_loss:.4f}, "
                     f"({time.time() - t0:.2f} sec)")
        t0 = time.time()

        self.evaluations_agent.append(avg_reward)
        self.evaluations_model.append(avrg_loss)
        self.model_pred_errors.append(np.array([np.mean(model_errors), np.std(model_errors)]).flatten())
        model_errors = []

        np.save((results_path/"agent_training.npy").absolute().as_posix(), self.evaluations_agent)
        np.save((results_path/"model_training.npy").absolute().as_posix(), self.evaluations_model)
        np.save((results_path/"model_usage_error.npy").absolute().as_posix(), np.array(self.model_pred_errors))

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
    self.env_model.save(file_path=file_path)

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
    self.env_model.save(file_path=file_path)

  def _plot_results(self, file_path):
    fig, ax = plt.subplots(1, 3, figsize=(16, 8))
    ax[0].plot(np.arange(len(self.evaluations_agent)), self.evaluations_agent)
    ax[0].set(xlabel="Iterations [1k]", ylabel="Average Reward",
           title="Average Reward of Evaluation during Training")
    ax[0].grid()

    ax[1].plot(np.arange(len(self.evaluations_model)), self.evaluations_model)
    ax[1].set(xlabel="Iterations [1k]", ylabel="L1: Mean Absolute Training Error",
              title="Average Loss of Training the Env. Model")
    ax[1].grid()

    error = np.array(self.model_pred_errors)[:, 0]
    std = np.array(self.model_pred_errors)[:, 1]
    ax[2].plot(np.arange(len(error)), error)
    #ax[2].plot(np.arange(len(std)), error+std, "g--", alpha=0.2)
    #ax[2].plot(np.arange(len(std)), error-std, "g--", alpha=0.2)
    ax[2].fill_between(np.arange(len(std)), error - std, error + std, color="gray", alpha=0.2)
    #ax[2].errorbar(x, y + 3, yerr=yerr, label='both limits (default)')
    ax[2].set(xlabel="Iterations [1k]", ylabel="Average (over 1k Iter.) Absolute Prediction Error",
              title="Prediction Error when using the Model.")
    ax[2].grid()

    fig.savefig((file_path / "training_progress.png").absolute().as_posix())
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
      "actor_layer_param": self.actor.layer_param,  # TODO add layer_param froma ctior
      "critic_layer_param": self.critic.layer_param
    }