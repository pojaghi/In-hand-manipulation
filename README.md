### Code for Mujoco Simulation in OpenAI Gym

This repository contains the code used in the Ojaghi et. al. submission to Sciences Advance Feb 2024.

#### Installation

To install the environment, please execute `setup.py`.
- Execute the code using:

  ```bash
  python setup.py install

#### MuJoCo Model

Before simulating Mujoco in OpenAI Gym, you'll need a MuJoCo XML model file in its native MJCF format. Details about the modeling process can be found in the `xml/ball` folder.

#### Building Gym Environment

Once the model is created, you can build the Gym environment to simulate the Mujoco model in the OpenAI Gym framework:

1. Put the model XML file in `gym/gym/envs/mujoco/assets`.
2. Create a new environment Python file (e.g., `yourenvname.py`) in `gym/gym/envs/mujoco`.  You can refer to the existing files provided by the Gym framework.

#### Additional Setup

- Put the `xml/ball` folder in `gym/gym/envs/mujoco/assets`.
- Install the required packages using:
pip install -r requirements.txt


#### Requirements

- Gym
- Mujoco

Please note that this project relies on the following dependencies:

- Gym version 0.17.3
- Stable Baselines version 2.9.0
- TensorFlow version 1.14.0
- Numpy version 1.21.0
- Mujocopy version 2.0.2.13


You can install TensorFlow using the following command:

pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.14.0-py3-none-any.whl

Ensure that all the requirements are met before running the code. If you encounter any issues, feel free to open an issue or reach out for support.

