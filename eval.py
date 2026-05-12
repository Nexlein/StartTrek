##
## EPITECH PROJECT, 2026
## StartTrek
## File description:
## eval
##

import os
import sys
import torch
import argparse
import gymnasium as gym
from artifacts import Artifacts
from utils import load_settings
from model.agent import DQNAgent

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


def eval_model(artifact: Artifacts, cli_seed=None, cli_wind=None):
    """
    Evaluate a trained DQN model.

    Loads the trained weights into a DQNAgent, and runs evaluation episodes while recording video.

    Args:
        artifact (Artifacts): The artifact manager containing model paths.
        cli_seed (int, optional): Overrides the seed from settings.yml if provided. Defaults to None.
        cli_wind (float, optional): Overrides the wind configuration.
                                    None uses config, -1.0 uses config power but forces True,
                                    any other >= 0 float uses that as wind power. Defaults to None.

    Returns:
        int: Exit status code (0 for success, 84 for error).
    """
    model_path = artifact.final_model_path
    if model_path is None or not os.path.exists(model_path):
        print("No final model found in artifact -> exit...", file=sys.stderr)
        return 84

    settings = load_settings()
    env_id = settings["environment"]["env_id"]
    n_episodes = settings["evaluation"]["n_episodes"]

    if cli_wind is None:
        enable_wind = settings["environment"].get("enable_wind", False)
        wind_power = settings["environment"].get("wind_power", 15.0)
    elif cli_wind == -1.0:
        enable_wind = True
        wind_power = settings["environment"].get("wind_power", 15.0)
    else:
        enable_wind = True
        wind_power = cli_wind

    seed_value = (
        cli_seed if cli_seed is not None else settings["environment"].get("seed", 1)
    )

    env = make_env(
        artifact.final_model_name,
        env_id,
        artifact.videos_folder,
        enable_wind,
        wind_power
    )

    print(f"Model found : {artifact.final_model_name}")
    print(f"Loading from: {model_path}")

    agent = DQNAgent(state_dim=8, action_dim=4)
    agent.policy_net.load_state_dict(torch.load(model_path))
    agent.policy_net.eval()
    agent.epsilon = 0.0

    print(f"Evaluating for {n_episodes} episodes (Seed: {seed_value}, Wind: {enable_wind})...")
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_value + ep)
        done = False
        episode_reward = 0.0

        print(f"Episode {ep + 1} / {n_episodes}")
        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                done = True

        print(f"Finished Episode {ep + 1} with Reward: {episode_reward:.2f}")

    env.close()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN model.")
    parser.add_argument(
        "--artifact",
        type=str,
        required=True,
        help="Path to the artifact folder containing the model",
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

    artifact_obj = Artifacts(load_path=args.artifact)

    sys.exit(
        eval_model(
            artifact=artifact_obj,
            cli_seed=args.seed,
            cli_wind=args.wind
        )
    )
