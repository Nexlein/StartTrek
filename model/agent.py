##
## EPITECH PROJECT, 2026
## StartTrek
## File description:
## agent
##

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from model.memory import ReplayMemory
from model.model import DQN


class DQNAgent:
    """
    Deep Q-Network Agent for reinforcement learning.

    This agent uses a policy network to select actions and a target network
    to stabilize the Q-value targets during training. It employs experience
    replay and an epsilon-greedy exploration strategy.
    """

    def __init__(self, state_dim=8, action_dim=4, lr=0.0001):
        """
        Initialize the DQNAgent with state and action dimensions.

        Args:
            state_dim (int): The dimensionality of the state space.
            action_dim (int): The dimensionality of the action space.
            lr (float): The learning rate for the optimizer.
        """
        self.action_dim = action_dim

        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayMemory(100000)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.gamma = 0.99

    def select_action(self, state):
        """
        Select an action using an epsilon-greedy policy.

        Args:
            state (array-like): The current state observation.

        Returns:
            int: The chosen action index.
        """
        # Epsilon-Greedy : Random or Best action
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)  # 100% random action (exploration)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()  # Return the index of the action with the highest Q-value (exploitation)

    def learn(self, batch_size=64):
        """
        Update the policy network using a batch of sampled experiences from memory.

        Uses Double DQN methodology for updating the network.

        Args:
            batch_size (int): The number of transitions to sample and train on.
        """
        # If we don't have enough memories, we can't learn yet
        if len(self.memory) < batch_size:
            return

        # Take a random batch of experiences from the memory
        transitions = self.memory.sample(batch_size)

        # Convert batch of experiences into tensors for PyTorch
        batch = list(zip(*transitions))
        states = torch.FloatTensor(np.array(batch[0]))
        actions = torch.LongTensor(np.array(batch[1])).unsqueeze(1)
        rewards = torch.FloatTensor(np.array(batch[2])).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(batch[3]))
        dones = torch.FloatTensor(np.array(batch[4])).unsqueeze(1)

        # What our brain THOUGHT the action was worth currently
        current_q_values = self.policy_net(states).gather(1, actions)

        # What the action is actually worth (reward + future reward) (Bellman equation)
        with torch.no_grad():
            # Double DQN: Use policy_net to select best next action, target_net to evaluate it
            best_next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            max_next_q_values = self.target_net(next_states).gather(
                1, best_next_actions
            )

            # Immediate reward + (estimated future reward * gamma)
            expected_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        # Calcul the loss
        loss = nn.functional.smooth_l1_loss(current_q_values, expected_q_values)

        # Update the brain (Backpropagation)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self, tau=0.005):
        """
        Soft updates the target network weights from the policy network.

        Args:
            tau (float): The interpolation parameter for soft updating.
        """
        for target_param, local_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )
