source ./venv/bin/activate
mkdir ./environments/
cd ./environments/
git clone https://github.com/benelot/pybullet-gym.git
cd pybullet-gym
pip install -e .