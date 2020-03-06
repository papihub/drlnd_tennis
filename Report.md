# drlnd_tennis
Deep Reinforcement Learning - Multi Agent Tennis

![MaDDPG agents Playing Tennis](https://video.udacity-data.com/topher/2018/May/5af7955a_tennis/tennis.png)

This repository contains a Deep Reinforcement learning agent based on DDPG algorithm. Two instances of this agent control rackets to bounce a ball over a net. The goal of the agents is to keep the ball in play for as many time steps as possible. The agent receives a reward of +0.1 if it hits the ball over the net. If the agent lets a ball hit the ground or hit the ball out of bounds, it receives a -0.01.

## Environment
In this environment the observation/state space has 24 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. The action space in continuous.

The task is episodic. This means there is a distinct "done" state. In order to solve the environment, our agent must get an average of +0.5 reward over 100 consecutive episodes after taking the maximum over both agents.

## Success criteria:
Our agent must get an average of +0.5 reward over 100 consecutive episodes.
After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores. This yields a single score for each episode.
We need avearage of this episode reward to be > 0.5 over 100 consecutive episodes.

## Approach : Reinforced Learning with Multi agent, Actor- Critic, Deep Deterministic Policy Gradient (DDPG) approach
This agent trains using DDPG algorithm (https://arxiv.org/abs/1509.02971) using four deep neural networks. Two of them correspond to an "Actor" and two to a "Critic". We implement multi agent reinforcement learning concepts discussed in this paper (https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf).

Actor of each agent takes current state of both agents as input and provides action as an output. There is a target actor network thats only used during training the actor. The target actor network is a time delayed copy of the actor network. Since we dont have a complete view of our environment, we use our actors current understanding of the environment as a target. Since our actor's understanding of the environment changes with every step, our target is non stationary. Neural networks learn better in a stable environment. So we try to keep the target for the actor relatively stable by making it a timedelayed copy of the actor.

Critic takes the actions from both the agents along with the current state of both agents as inputs. It provides a single value as an output. Critic's output is an evaluation of the actor's output. It helps the actor decide appropriate action while learning. Critic also uses a critic-target network while training. Simiar to actor-target, critic-target is a time delayed copy of critic and helps stablize critic's learning.

Since the critic has access to all agent's observations and the actions taken by all agents, the environment can be considered  staionary. This will give us the convergence criteria for training the DDPG agents.

Each agent maintain a replay buffer to store previously seen states, actions taken in those states, resulting rewards, next states returned by environment and whether its a done state or not. Both agents have access to states encountered either of the agents and actions taken by the agents.

Initially the neural networks are initialized with random weights.

Each agent interacts with the environment by taking actions and observing the reward and state changes.

When the agents takes an action in a current state, the environment returns the next state along with a reward and a *done* indicator if we reached the terminal state.

When either of the agents reache a terminal state, the eposide terminates.

The agents maintain a tuple of ( state, action, reward, next state, done ) for all steps in an episode in a replay buffer.

The agents sample episodes from their replay buffers to trains their neural networks models.

The agents learn after a pre-determined number of steps - a hyper parameter - 1 in this case.

Each agent uses its 'actor' deep neural netowrks to come up with an action given the current state. It uses a 'critic' DNN to evaluate the actor's output. The critic has access to observations from both agents as well as actions take by both agents.

For stability of the neural networks, the agents maintain a target network and a local network for both actor and critic networks. Actor local network is used to take action and is refeined in each learning step. The target network is only updated after a fixed number of learning steps. Target networks are not trained. They meerly get a scaled copy of the local networks.

In each learning step, the agents compute the difference between expected and predicted values and use a learning rate along with a discount factor to learn from the difference(loss) between expected and predicted values.

In training mode, during the initial episodes we encourage our agents to explore. As we collect more experiences, we want to reduce exploration and increase exploitation. This is achieved by adding noise to the actions predicted by the actor. The amount of noise is high initially and is gradually reduced to a minimum. The starting noise, the decay factor for noise and the low bound for noise are all hyper parameters.

### Network Architecture
Each agent uses 2 different deep neural networks to learn from the environment interactions.

Actor network:
1 input layer, 2 hidden layer and 1 output layer.
All layers are fully connected.
Input layers has (state_size)*2 inputs
The two hidden layers have 256 and 128 units respectively.
Output layer has (action_size) outputs

State_size = 24 and action_size = 2 for this environment.

Input and hidden layers go thru a relu activation function.
Output layer goes thru a tanh activation function to ensure all output values are between -1 and 1

We use a mean squared loss function to compute the loss values.

We use Adam optimizer to backpropogate the loss and update weights.

Critic network:
4 layers all fully connected.
layer 1 takes states as input
layer 2 takes actions as input and the output from layer 1
layer 3 takes the output from layer 2
layer 4 is the output layer 

All these layers are fully connected.
layer 1, & 3 go thru a relu activation function
layer 4 is the output layers and has no activation function.

The critic also uses a mean squared loss function and an Adam optimizer to backpropogate the loss and udpate weights.


### Hyper parameters and their values
|Hyper parameter|Value|Comment|
|---------------|:---:|-------|
|Replay buffer size|1000,000|buffer_size|
|Discount Factor|0.99|gamma|
|How often do we learn?|1|update_every|
|No of experiences we use for learning|128|sample size|
|Factor for target network update|1e-3|tau. Same tau for both actor and critic|
|Learning Rate|1e-3|lr_(actor|critic). Use the same learning rate for actor and critic.|
|No of epochs during each training step|1| We sample once from the buffers and learn from each of those sample sets|

## Training results and evaluation

The agent was trained for upto 2000 episodes. During each training episode, in each step both agents take actions. The results are captured and stored in their respective replay buffers. After we accumulate enough experiences to sample from, each agent perfrom a learning step using 128 samples from the replay buffers. At the end of each episode we get the max score between the two agents. The below graph shows these averaged scores until the agent reached its goal of +0.5 over 100 consecutive episodes.
![Scores during Training](https://github.com/papihub/drlnd_tennis/blob/master/tennis_training_2000.png)

Here is an evaluation run:
![Evaluation of trained agent](https://github.com/papihub/drlnd_tennis/blob/master/tennis_200_steps_min.gif)

Actor network weights used in the above evaluation run are in [checkpoint_*.pth](https://github.com/papihub/drlnd_continuous_control/blob/master/) files.

## Future actions / steps
I want to try some of the other training methods like Proximal policy optimizing, TD3 and TRPO in a multi agent setting.
I also want to try various exploration vs exploitation hyper parameters and network structures. I noticed even small changes in these hyper parameters cause the agent to crash or not learn at all. I want to try to understand how these hyper parameters affect agent performance.
