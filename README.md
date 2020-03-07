# drlnd_tennis
Deep Reinforcement Learning - Multi Agent Tennis

![MaDDPG agents Playing Tennis](https://video.udacity-data.com/topher/2018/May/5af7955a_tennis/tennis.png)

This repository contains a Deep Reinforcement learning agent based on DDPG algorithm. Two instances of this agent control rackets to bounce a ball over a net. The goal of the agents is to keep the ball in play for as many time steps as possible. The agent receives a reward of +0.1 if it hits the ball over the net. If the agent lets a ball hit the ground or hit the ball out of bounds, it receives a -0.01.

## Environment
In this environment the observation/state space has 24 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. The action space in continuous.

The task is episodic. This means there is a distinct "done" state. In order to solve the environment, our agent must get an average of +0.5 reward over 100 consecutive episodes after taking the maximum over both agents.

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores. This yields a single score for each episode.

We need avearage of this episode reward to be > 0.5 over 100 consecutive episodes to successfully solve this environment.

In tennis.ipnb we train a Maddpg agent to learn by interacting with this environment and solve it. The agent has no knowledge of the environment, the rules of the game or the reward structure. All it knows is are the observation and action space shape (size/type).

## Dependencies
I developed code in this repository on a windows 10 64bit OS. So, I havent tested if this code works on any other OS.

**Miniconda**: Install miniconda3 from [miniconda download page](https://docs.conda.io/en/latest/miniconda.html)

**Python**: Follow the instructions in [DRLND Github repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your python environment. These instructions will guide you to install PyTorch, ML-Agents toolkit and a couple of other python packages required for this project.

**Unity Environment**: Download the unity environment from [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip). This is for windows 10 64bit OS. Please refer to the course material if you want the environment for a different OS.

## Instructions to train the agent:
Install the dependencies above.

open a jupyter notebook.

Run the notebook Tennis.ipynb to train and test the agent. The notebook has instructions to load a saved model and to save a trained model.

Refer to the Report.md for the approach used by the agent, training and evaluation.
