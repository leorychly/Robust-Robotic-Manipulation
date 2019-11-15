import time
from absl import logging

from src.utils import read_json
from src.agents.td3.td3_agent import TD3Agent

def eval_td3_agent(env,
                   model_path,
                   param_path,
                   eval_ep=1000):
  """
  Test the TD3 Agent on randomly initialized environments and render the result.

  :param env: gym.Env
    OpenAI Gym environment to test in.
  :param model_path: String
    Path to the saved TD3 model.
  :param param_path: String
    Path to the parameters of the saved TD3 model.
  :param eval_ep: Int
    Number of episoed to evaluate the agent.
  """
  print("\n========== START EVALUATION ==========\n")
  param = read_json(str(param_path.absolute() / "experiment_param.json"))
  config = param["experiment_config"]

  #agent = TD3Agent(env=env,
  #                 actor_layer=param["agent_param"]["actor_layer_param"],
  #                 critic_layer=param["agent_param"]["critic_layer_param"],
  #                 actor_lr=param["experiment_config"]["actor_lr"],
  #                 critic_lr=param["experiment_config"]["critic_lr"],
  #                 observer=eval(param["experiment_config"]["observer"])(
  #                   observation_space=env.observation_space),
  #                 executer=eval(param["experiment_config"]["executer"])(
  #                   action_space=env.action_space),
  #                 buffer_size=param["experiment_config"][
  #                   "buffer_size"])

  observer = config["observer"]
  executer = config["executer"]

  agent = TD3Agent(env=env,
                   actor_layer=config["actor_layer"],
                   critic_layer=config["critic_layer"],
                   actor_lr=config["actor_lr"],
                   critic_lr=config["critic_lr"],
                   observer=eval(observer)(observation_space=env.observation_space),
                   executer=eval(executer)(action_space=env.action_space),
                   buffer_size=config["buffer_size"],
                   discount=config["discount"],
                   tau=config["tau"],
                   policy_noise=config["policy_noise"],
                   noise_clip=config["noise_clip"],
                   policy_freq=config["policy_freq"])

  agent.load(model_path)
  logging.info(f"The model was loaded from '{model_path}'")

  logging.info(f"Starting {eval_ep} evaluation episodes...")
  avg_reward = agent.eval_policy_on_env(eval_gym_env=env, eval_episodes=eval_ep)
  logging.info(f"Average Reward during {eval_ep} evaluation episodes: {avg_reward}.")

  run_agent("InvertedDoublePendulumMuJoCoEnv-v0", agent, steps=1000)


def run_agent(env_name, agent, steps):
  """
  Run an agent on an environment for _ steps.

  :param env_name: String
    Name of the environment.
  :param agent: BaseAgent
    A trained agent.
  :param steps: Int
    Number of steps to run the agent.
  """
  env = gym.make(env_name)
  env.render()
  obs = env.reset()
  action = agent.run(state=obs)
  obs, rewards, done, _ = env.step(action=action)
  for i in range(steps):
    action = agent.run(state=obs)
    obs, rewards, done, _ = env.step(action)
    time.sleep(0.01)