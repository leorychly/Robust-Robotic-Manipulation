# Robust Learning and Control for Robotic Tasks

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
* Create a virtual python environment and install  all requirements with `bash install.sh`. This will also clone and install PyBullet in ./environments/pybulletgym.
(The requirements can alternatively be installed separately with `pip3 install -r requirements.txt`)

Experiments for a particular environment can be run using the following command and a config file (`./experiments/configs/exp01.json`) where the experiment paramters are defined.

```
bash run_experiment_train.sh

-t    --train         If set to True, the agent is also trained before evaluation. (E.g. True)
-l    --logging       Select logging level. "info" for  basic output; "debug" for debugging purposes. (Eg. 'info')
-c    --config        Experiment config file. (E.g '/Robust-Robotic-Manipulation/experiments/configs/exp01.json')
-d    --directory     The experiment output directory. (E.g.: './experiment_results')
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
