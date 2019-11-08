# Learning and Robust Control for Robotic Tasks

---

### Evaluating the robustness of learned control policies for robotic tasks:

The agent is designed according to the standard robotics paradigm Sense-Plan-Act. In this project the Sense and Act
modules are utilized for assessing the planner's robustness against noise and domain shift in observations as well as
during the execution of actions.

---

### Run the experiments:

Requirements:
* Python 3.6+
* PyTorch 1.0+
* [PyBullet Gymperium](https://github.com/benelot/pybullet-gym), an open-source implementation of the OpenAI Gym MuJoCo

Installation:
* Create a virtual python environment and install  all requirements with: `bash install.sh`
(The requirements can also be installed separately with `pip3 install -r requirements.txt`)
* To install PyBullet run `bash install_pybulletgym.sh`. This will activate the venv, clone pybulletgym and install it.

Experiments for a particular environment can be run using:

```
python ./experiments/run.py

-e    --environment   to choose one of the pybulletgym environments. Default is "InvertedDoublePendulumMuJoCoEnv-v0"
-a    --agent         to choose which agent to run.
-t    --train         if set to True, the agent is also trained before evaluation.
-exe  --executor      select an execution model. By default the BaseExecutor is used which executes the action given from the agent without modification.
-obs  --observer      select an observer model. By default the baseObserver is used which passes the environment state as is to the agent.
-l    --logging       select logging level. "info" for  basic output; "debug" for debugging purposes.
-s    --seed          set the random seed that will ne used for numpy and PyTorch.
```

---

### Project Overview
#### Agents
* __TD3__: (successor of DDPG) a state of the art model-free reinforcement learning algorithm for continuous control problems.
 The TD3 focuses on reducing the overestimation bias seen from the DDPG and similar algorithms by:
    * Using a pair of critic networks
    * Delayed updates of the actor
    * Action noise regularisation
    
  As a result the TD3 training should be more stable and less reliant on finding the correct hyper parameters for the current task,
  because it does not continuously over estimates the Q values of the critic (value) network. 
  Otherwise, these estimation errors build up over time and can lead to the agent falling into a local optima 
  or experience catastrophic forgetting. 