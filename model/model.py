##
## EPITECH PROJECT, 2026
## StartTrek
## File description:
## model
##

import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Standard fully connected Deep Q-Network.

    A simple multilayer perceptron used to estimate Q-values for each action
    given a state observation.
    """

    def __init__(self, input_dim=8, output_dim=4):
        """
        Initialize the DQN neural network layers.

        Args:
            input_dim (int): Dimensionality of the input state.
            output_dim (int): Number of possible actions (output dimension).
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): A batch of state observations.

        Returns:
            torch.Tensor: Q-values for each action.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
