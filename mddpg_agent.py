import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def layer_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size*2, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*layer_init(self.fc1))
        self.fc2.weight.data.uniform_(*layer_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=256, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size*2, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+(action_size*2), fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*layer_init(self.fcs1))
        self.fc2.weight.data.uniform_(*layer_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.eps = 6
        self.eps_decay = 750
        self.eps_end = 0
        self.t_step = 0
        self.batch_size = 128        # minibatch size
        self.buffer_size = int(1e6)  # replay buffer size
        self.gamma = 0.99            # discount factor
        self.lr_actor = 1e-3         # learning rate of the actor 
        self.lr_critic = 1e-3        # learning rate of the critic
        self.tau = 1e-3              # for soft update of target parameters
        self.weight_decay = 0        # L2 weight decay
        self.update_every = 1        # time steps between network updates
        self.n_updates = 1           # number of times training

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay)
                
        # Noise process
        self.noise = OUNoise((1, action_size), random_seed)

        # Replay memory
        self.memory_n = ReplayBuffer(self.buffer_size, random_seed)
        self.memory_p = ReplayBuffer(self.buffer_size, random_seed)
    
    def step(self, state, action, reward, next_state, done, agent_number, rsum):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.t_step += 1
        # Save experience / reward
        if(rsum > 0):
            self.memory_p.add(state, action, reward, next_state, done)
        else:
            self.memory_n.add(state, action, reward, next_state, done)
        self.memory_n.add(state, action, reward, next_state, done)
    
        # Learn, if enough samples are available in memory and at interval settings
        if len(self.memory_n) + len(self.memory_p) > self.batch_size:
            if self.t_step % self.update_every == 0:
                for _ in range(self.n_updates):
                    #p_sz = len(self.memory_p)
                    #n_sz = int(self.batch_size/2)
                    #if(p_sz < n_sz):
                    #    n_sz = self.batch_size - p_sz
                    #elif(p_sz > n_sz):
                    #    p_sz = n_sz
                    n_sz = self.batch_size
                    
                    n_exp = self.memory_n.sample(n_sz)
                    #p_exp = self.memory_p.sample(p_sz)
                    #self.learn(n_exp, p_exp, GAMMA, agent_number)
                    self.learn(n_exp, self.gamma, agent_number)

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)
        #actions = np.zeros((1, self.action_size))
        self.actor_local.eval()
        #print("states.shape: ", states.shape)
        with torch.no_grad():
            #for agent_num, state in enumerate(states):
            #    action = self.actor_local(states).cpu().data.numpy()
            #    actions[agent_num,:] = action
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        #print("Before noise: actions: ", actions)
        if add_noise:
            noise = self.eps * self.noise.sample()
            actions += noise
            #print("noise: ", noise)
        #print("After noise: ", actions)
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    #def learn(self, n_exp, p_exp, gamma, agent_number):
    def learn(self, n_exp, gamma, agent_number):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states_n, actions_n, rewards_n, next_states_n, dones_n = n_exp
        #states_p, actions_p, rewards_p, next_states_p, dones_p = p_exp
        #states, actions, rewards, next_states, dones = n_exp
        
        #states = torch.from_numpy(np.vstack(states_n + states_p)).float().to(device)
        #actions = torch.from_numpy(np.vstack(actions_n + actions_p)).float().to(device)
        #rewards = torch.from_numpy(np.vstack(rewards_n + rewards_p)).float().to(device)
        #next_states = torch.from_numpy(np.vstack(next_states_n + next_states_p)).float().to(device)
        #dones = torch.from_numpy(np.vstack(dones_n + dones_p).astype(np.uint8)).float().to(device)
        
        states = torch.from_numpy(np.vstack(states_n)).float().to(device)
        actions = torch.from_numpy(np.vstack(actions_n)).float().to(device)
        rewards = torch.from_numpy(np.vstack(rewards_n)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_states_n)).float().to(device)
        dones = torch.from_numpy(np.vstack(dones_n).astype(np.uint8)).float().to(device)

        #print("Types: states: {}, actions: {}, rewards: {}, next_states: {}, dones: {}".
        #     format(type(states), type(actions), type(rewards), type(next_states), type(dones.shape)))
        #print("shapes: states: {}, actions: {}, rewards: {}, next_states: {}, dones: {}".
        #     format(states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape))
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        
        actions_next = torch.cat((actions[:,:agent_number*2],actions_next,actions[:,(agent_number+1)*2:]), dim=1)
            
        Q_targets_next = self.critic_target(next_states, actions_next)
        #print("agent_number:{} next_states.shape:{} actions_next.shape:{}, q_targets_next.shape:{}".
        #     format(agent_number, next_states.shape, actions_next.shape, Q_targets_next.shape))
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        #print("rewards.shape:{} Q_targets.shape:{}".format(rewards.shape, Q_targets.shape))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        
        actions_pred = torch.cat((actions[:,:agent_number*2], actions_pred, actions[:,(agent_number+1)*2:]), dim=1)

        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)                     

        # Update epsilon noise value
        self.eps = self.eps - (1/self.eps_decay)
        if self.eps < self.eps_end:
            self.eps=self.eps_end
                  
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=0.13, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
    
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self, sz):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=sz)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for i in experiences:
            states.append(i.state)
            actions.append(i.action)
            rewards.append(i.reward)
            next_states.append(i.next_state)
            dones.append(i.done)
            
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
