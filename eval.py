##
## EPITECH PROJECT, 2026
## StartTrek
## File description:
## eval
##

import os
import argparse
import gymnasium as gym
import torch
from model.agent import DQNAgent
from utils import get_name_from_path, get_model_path, load_settings


def make_env(
    model_name: str,
    env_id: str,
    video_folder: str,
    enable_wind: bool = False,
    wind_power: float = 15.0,
):
    """
    Create a Gymnasium environment with video recording for evaluation.

    Args:
        model_name (str): The name of the model being evaluated, used for the video folder.
        env_id (str): The Gymnasium environment ID.
        video_folder (str): Directory where the videos will be saved.
        enable_wind (bool, optional): Whether to enable wind dynamics. Defaults to False.
        wind_power (float, optional): The strength of the wind. Defaults to 15.0.

    Returns:
        gym.Env: The wrapped Gymnasium environment for recording evaluation videos.
    """
    env = gym.make(
        id=env_id,
        render_mode="rgb_array",
        enable_wind=enable_wind,
        wind_power=wind_power,
    )
    env = gym.wrappers.RecordVideo(
        env=env,
        video_folder=os.path.join(video_folder, model_name),
        name_prefix="eval",
        episode_trigger=lambda episode_id: True,
    )
    return env


def main():
    """
    Main function to evaluate a trained DQN model.

    Parses command-line arguments to find the model path, loads the trained
    weights into a DQNAgent, and runs evaluation episodes while recording video.

    Returns:
        int: Exit status code (0 for success, 84 for error).
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN model.")
    parser.add_argument(
        "model_path", nargs="?", default=None, help="Path to the trained model (.pth)"
    )
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

    settings = load_settings()
    env_id = settings["environment"]["env_id"]

    if args.wind is None:
        enable_wind = settings["environment"].get("enable_wind", False)
        wind_power = settings["environment"].get("wind_power", 15.0)
    elif args.wind == -1.0:
        enable_wind = True
        wind_power = settings["environment"].get("wind_power", 15.0)
    else:
        enable_wind = True
        wind_power = args.wind

    seed_value = (
        args.seed if args.seed is not None else settings["environment"].get("seed", 1)
    )

    video_folder = settings["paths"]["video_folder"]
    n_episodes = settings["evaluation"]["n_episodes"]

    model_path = get_model_path(args.model_path)
    model_name = get_name_from_path(model_path)
    env = make_env(model_name, env_id, video_folder, enable_wind, wind_power)

    os.makedirs(video_folder, exist_ok=True)

    if os.path.exists(model_path):
        print(f"Model found at {model_path} -> loading...")
        agent = DQNAgent(state_dim=8, action_dim=4)
        agent.policy_net.load_state_dict(torch.load(model_path))
        agent.policy_net.eval()
        agent.epsilon = 0.0
    else:
        print(f"Model not found at {model_path} -> exit...")
        return 84

    print("Evaluating...")
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_value + ep)
        done = False

        print(f"Episode {ep + 1} / {n_episodes}")
        while not done:
            action = agent.select_action(obs)
            obs, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                done = True

    env.close()
    return 0


if __name__ == "__main__":
    main()
