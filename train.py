##
## EPITECH PROJECT, 2026
## StartTrek
## File description:
## train
##

import os
import yaml
import torch
import random
import argparse
import numpy as np
from artifacts import Artifacts
from model.agent import DQNAgent
from gymnasium.envs.box2d.lunar_lander import LunarLander
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


def train(artifact: Artifacts, cli_seed=None, cli_random_wind=None):
    """
    Train the DQNAgent on the LunarLander environment.

    Loads hyperparameters, initializes the environment and agent, and runs
    the training loop over multiple episodes. It handles epsilon decay,
    memory updates, model training steps, and saves the model checkpoints.

    Args:
        artifacts (Artifacts): An instance of the Artifacts class.
        cli_seed (int, optional): Overrides the seed from settings.yml if provided. Defaults to None.
        cli_random_wind (bool, optional): Overrides random_wind from config if provided. Defaults to None.
    """
    settings = load_settings()

    seed_value = (
        cli_seed if cli_seed is not None else settings["environment"].get("seed", 1)
    )
    seed_everything(seed_value)

    random_wind_value = (
        cli_random_wind
        if cli_random_wind is not None
        else settings["environment"].get("random_wind", False)
    )

    settings["environment"]["seed"] = seed_value
    settings["environment"]["random_wind"] = random_wind_value
    with open(
        os.path.join(artifact.configs_folder, "settings.yml"), "w", encoding="utf-8"
    ) as f:
        yaml.dump(settings, f, default_flow_style=False)

    config = load_hyperparameters()
    env_id = settings["environment"]["env_id"]
    max_episodes = settings["training"]["max_episodes"]

    env = make_video_env(
        env_id=env_id,
        base_folder=artifact.videos_folder,
        mode="train",
        seed=seed_value,
        video_freq=0,
    )

    agent = DQNAgent(state_dim=8, action_dim=4, lr=config["learning_rate"])
    agent.gamma = config["gamma"]
    buffer_size = config.get("buffer_size", agent.memory.memory.maxlen)
    agent.memory.set_capacity(buffer_size)

    agent.epsilon = config["exploration_initial_eps"]
    agent.epsilon_min = config["exploration_final_eps"]
    epsilon_decay = config.get("exploration_decay", 0.9995)

    print(
        f"[TRAIN] Starting training on {env_id} | Seed: {seed_value} | Random Wind: {random_wind_value}"
    )
    print(
        f"[TRAIN] Artifact tracking enabled. Videos saved to: {artifact.videos_folder}"
    )

    best_reward = -float("inf")

    for episode in range(max_episodes):
        episode_loss = []
        loss_val = None

        if random_wind_value:
            # Randomly toggle wind for this episode
            is_windy = random.choice([True, False])

            unwrapped_env = env.unwrapped
            if isinstance(unwrapped_env, LunarLander):
                unwrapped_env.enable_wind = is_windy
                if is_windy:
                    unwrapped_env.wind_power = random.uniform(5.0, 20.0)

        state, _ = env.reset(seed=seed_value + episode)
        episode_reward = 0.0
        done = False
        step = 0
        termination_reason = "ongoing"

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if done:
                if terminated:
                    if float(reward) <= -100:
                        termination_reason = "crash"
                    else:
                        termination_reason = "sleep"
                elif truncated:
                    termination_reason = "out-of-view"

            agent.memory.push(state, action, reward, next_state, terminated)

            if len(agent.memory) > config["batch_size"]:
                loss_val = agent.learn(config["batch_size"])
                agent.update_target_network()

            if loss_val is not None:
                    episode_loss.append(loss_val)

            state = next_state
            episode_reward += float(reward)
            step += 1

            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= epsilon_decay

        print(
            f"[TRAIN] Episode {episode + 1:4d}/{max_episodes} | Score: {episode_reward:7.2f} | Steps: {step:4d} | Epsilon: {agent.epsilon:.3f} | Cause: {termination_reason}"
        )

        avg_loss = float(np.mean(episode_loss)) if episode_loss else float("nan")
        artifact.log_step(
            [episode, episode_reward, step, agent.epsilon, termination_reason, avg_loss]
        )

        if episode > 0:
            if episode + 1 == max_episodes:
                artifact.save_final_model(
                    agent.policy_net.state_dict(), seed_value, episode
                )
            elif episode % 50 == 0:
                artifact.save_checkpoint_model(
                    agent.policy_net.state_dict(), seed_value, episode
                )

        if episode_reward > best_reward:
            best_reward = episode_reward
            os.makedirs(artifact.models_folder, exist_ok=True)
            artifact.save_best_model(agent.policy_net.state_dict(), seed_value)
            print(f"[TRAIN] New best model saved! Reward: {best_reward:.2f}")

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
        "--random-wind",
        action="store_true",
        default=None,
        help="Randomize wind presence and power across episodes for robust training (overrides settings.yml)",
    )
    parser.add_argument(
        "--artifact", type=str, default=None, help="Path to an existing artifact folder"
    )
    args = parser.parse_args()

    artifact_obj = Artifacts(
        load_path=args.artifact,
        configs=["configs/hyperparameters.yml", "configs/settings.yml"],
    )

    train(artifact=artifact_obj, cli_seed=args.seed, cli_random_wind=args.random_wind)
