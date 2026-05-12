##
## EPITECH PROJECT, 2026
## StartTrek
## File description:
## train
##

import os
import gymnasium as gym
import torch
import argparse
import random
import numpy as np

from model.agent import DQNAgent
from utils import load_hyperparameters, load_settings


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


def train(cli_seed=None, cli_wind=None):
    """
    Train the DQNAgent on the LunarLander environment.

    Loads hyperparameters, initializes the environment and agent, and runs
    the training loop over multiple episodes. It handles epsilon decay,
    memory updates, model training steps, and saves the model checkpoints.

    Args:
        cli_seed (int, optional): Overrides the seed from settings.yml if provided. Defaults to None.
        cli_wind (float, optional): Overrides the wind configuration.
                                    None uses config, -1.0 uses config power but forces True,
                                    any other >= 0 float uses that as wind power. Defaults to None.
    """
    settings = load_settings()

    seed_value = (
        cli_seed if cli_seed is not None else settings["environment"].get("seed", 1)
    )
    seed_everything(seed_value)

    config = load_hyperparameters()

    env_id = settings["environment"]["env_id"]

    if cli_wind is None:
        enable_wind = settings["environment"].get("enable_wind", False)
        wind_power = settings["environment"].get("wind_power", 15.0)
    elif cli_wind == -1.0:
        enable_wind = True
        wind_power = settings["environment"].get("wind_power", 15.0)
    else:
        enable_wind = True
        wind_power = cli_wind

    model_folder = settings["paths"]["model_folder"]
    max_episodes = settings["training"]["max_episodes"]

    env = gym.make(
        env_id, render_mode="rgb_array", enable_wind=enable_wind, wind_power=wind_power
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

    log_file = f"train_results_seed_{seed_value}.csv"
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("Episode,Reward,Length,Epsilon\n")

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
        with open(log_file, "a") as f:
            f.write(f"{episode},{episode_reward},{step},{agent.epsilon}\n")

        if episode > 0 and episode % 50 == 0:
            os.makedirs(model_folder, exist_ok=True)
            torch.save(
                agent.policy_net.state_dict(),
                f"{model_folder}/model_seed_{seed_value}_ep_{episode}.pth",
            )

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed value for reproducibility (overrides settings.yml)",
    )
    parser.add_argument(
        "--wind",
        nargs="?",
        type=float,
        const=-1.0,
        default=None,
        help="Enable wind in the environment. Optionally provide wind power (e.g., --wind 15.0)",
    )
    args = parser.parse_args()

    train(cli_seed=args.seed, cli_wind=args.wind)
