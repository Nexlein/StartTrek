##
## EPITECH PROJECT, 2026
## StartTrek
## File description:
## train
##

import os
import torch
import random
import argparse
import numpy as np
import gymnasium as gym

from artifacts import Artifacts
from model.agent import DQNAgent
from utils import load_hyperparameters, load_settings, make_video_env


def seed_everything(seed: int):
    """
    Set all random seeds for reproducibility.

    Args:
        seed (int): The seed value to use for random number generators
                    (random, numpy, and torch).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(seed_value: int, artifact: Artifacts):
    """
    Train the DQNAgent on the LunarLander environment.

    Loads hyperparameters, initializes the environment and agent, and runs
    the training loop over multiple episodes. It handles epsilon decay,
    memory updates, model training steps, and saves the model checkpoints.

    Args:
        seed_value (int): The random seed to ensure reproducible training runs.
    """
    seed_everything(seed_value)

    config = load_hyperparameters()
    settings = load_settings()

    env_id = settings["environment"]["env_id"]
    max_episodes = settings["training"]["max_episodes"]

    env = gym.make(env_id, render_mode="rgb_array")
    env = make_video_env(
        env_id=env_id,
        base_folder=artifact.videos_folder,
        mode="train",
        seed=seed_value
    )

    agent = DQNAgent(state_dim=8, action_dim=4)
    agent.optimizer.param_groups[0]["lr"] = config["learning_rate"]
    agent.gamma = config["gamma"]
    buffer_size = config.get("buffer_size", agent.memory.memory.maxlen)
    agent.memory.set_capacity(buffer_size)

    agent.epsilon = config["exploration_initial_eps"]
    agent.epsilon_min = config["exploration_final_eps"]
    epsilon_decay = 0.995

    print(f"Starting training on {env_id} with seed {seed_value}")
    print(f"Artifact folder: {artifact.videos_folder}")

    for episode in range(max_episodes):
        state, _ = env.reset(seed=seed_value + episode)
        episode_reward = 0.0
        done = False
        step = 0

        while not done:
            action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.memory.push(state, action, reward, next_state, terminated)

            if len(agent.memory) > config["batch_size"]:
                agent.learn(config["batch_size"])
                agent.update_target_network()

            state = next_state
            episode_reward += float(reward)
            step += 1

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= epsilon_decay

        print(
            f"Episode {episode}: Score = {episode_reward:.2f}, Steps = {step}, Epsilon = {agent.epsilon:.2f}"
        )
        artifact.log_step([episode, episode_reward, step, agent.epsilon])

        if episode > 0:
            if episode + 1 == max_episodes:
                artifact.save_final_model(agent.policy_net.state_dict(), seed_value, episode)
            elif episode % 50 == 0:
                artifact.save_checkpoint_model(agent.policy_net.state_dict(), seed_value, episode)

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=1, help="Seed value for reproducibility"
    )
    parser.add_argument(
        "--artifact", type=str, default=None, help="Path to an existing artifact folder"
    )
    args = parser.parse_args()

    artifact = Artifacts(
        load_path=args.artifact,
    )

    train(seed_value=args.seed, artifact=artifact)
