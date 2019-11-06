# Learning and Robust Control for Robotic Tasks

---

### Evaluating the robustness of learned control policies for robotic tasks:

The agent is designed according to the standard robotics paradigm Sense-Plan-Act. In this project the Sense and Act
modules are utilized for assessing the planner's robustness against noise and domain shift in observations as well as
during the execution of actions.

---

### Run the experiments:

Requirements:
- Python 3.6+
- [PyBullet Gymperium](https://github.com/benelot/pybullet-gym), an open-source implementation of the OpenAI Gym MuJoCo

To install all requirements a virtual python environment is recommended which can be installed using the command:
```bash install.sh```.

Install PyBullet:

```
cd ./src/environments/
git clone https://github.com/benelot/pybullet-gym.git
cd pybullet-gym
pip install -e .
```

To run an experiment use the following command inside the virtual environment:
```python train.py```

---

### Project Overview
#### Agents
