##
## EPITECH PROJECT, 2025
## StartTrek
## File description:
## test_agent.py
##

import torch
from model.agent import DQNAgent


def test_agent_greedy_action_range():
    """
    Verifies that the agent selects a valid action between 0 and 3
    when exploration is disabled.

    Prevents environment crashes by ensuring the neural network's
    output always maps to a valid action index.
    """
    agent = DQNAgent(state_dim=8, action_dim=4)
    agent.epsilon = 0.0

    obs = torch.randn(8).numpy()

    action = agent.select_action(obs)

    assert 0 <= action <= 3


def test_replay_buffer_push_and_len():
    """
    Ensures that adding transitions to the memory buffer correctly
    updates the reported length.

    Guarantees that the agent is successfully collecting and retaining
    the experience data required for learning.
    """
    agent = DQNAgent(state_dim=8, action_dim=4)
    agent.memory.set_capacity(50)

    for i in range(10):
        agent.memory.push([float(i)] * 8, 0, 1.0, [0.0] * 8, False)

    assert len(agent.memory) == 10


def test_agent_learn_step():
    """
    Confirms that the agent can successfully execute a training update
    using stored experiences without crashing.

    Validates the entire backpropagation pipeline, ensuring that loss
    calculation and weight updates are structurally sound.
    """
    agent = DQNAgent(state_dim=8, action_dim=4)

    for _ in range(40):
        agent.memory.push([0.0] * 8, 0, 1.0, [0.0] * 8, False)

    agent.learn(batch_size=32)
