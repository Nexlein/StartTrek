##
## EPITECH PROJECT, 2026
## StartTrek
## File description:
## memory
##

import random
from collections import deque


class ReplayMemory:
    """
    A cyclic buffer of bounded size that holds the transitions observed recently.
    """

    def __init__(self, capacity=10000):
        """
        Initialize the replay memory.

        Args:
            capacity (int): The maximum number of transitions the memory can store.
        """
        # deque : when the memory is full, it automatically discards the oldest memories.
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Save a transition in the replay memory.

        Args:
            state: The previous state observation.
            action: The action taken.
            reward: The reward received.
            next_state: The resulting state observation.
            done (bool): Whether the episode has terminated.
        """
        # Save the experience as a tuple in the memory
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions from memory.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            list: A list of sampled transitions.
        """
        # Randomly sample a batch of experiences from the memory
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Get the current size of internal memory.

        Returns:
            int: The number of transitions currently stored.
        """
        # Return the current number of experiences stored in memory
        return len(self.memory)

    def set_capacity(self, capacity):
        """
        Update the maximum capacity of the replay memory buffer.

        Args:
            capacity (int): The new maximum capacity.
        """
        # Recreate the deque with a new maximum length
        existing_items = list(self.memory)
        self.memory = deque(existing_items, maxlen=capacity)
