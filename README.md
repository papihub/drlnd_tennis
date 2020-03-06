# drlnd_tennis
Deep Reinforcement Learning - Multi Agent Tennis

![MaDDPG agents Playing Tennis](https://video.udacity-data.com/topher/2018/May/5af7955a_tennis/tennis.png)

This repository contains a Deep Reinforcement learning agent based on DDPG algorithm. Two instances of this agent control rackets to bounce a ball over a net. The goal of the agents is to keep the ball in play for as many time steps as possible. The agent receives a reward of +0.1 if it hits the ball over the net. If the agent lets a ball hit the ground or hit the ball out of bounds, it receives a -0.01.

## Dependencies
I developed code in this repository on a windows 10 64bit OS. So, I havent tested if this code works on any other OS.

**Miniconda**: Install miniconda3 from [miniconda download page](https://docs.conda.io/en/latest/miniconda.html)

**Python**: Follow the instructions in [DRLND Github repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your python environment. These instructions will guide you to install PyTorch, ML-Agents toolkit and a couple of other python packages required for this project.

**Unity Environment**: Download the unity environment from [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip). This is for windows 10 64bit OS. Please refer to the course material if you want the environment for a different OS.

## Instructions to train the agent:
Install the dependencies above.

open a jupyter notebook.

Run the notebook continuous_control.ipynb to train and test the agent. The notebook has instructions to load a saved model and to save a trained model.

Refer to the Report.md for the approach used by the agent, training and evaluation.
