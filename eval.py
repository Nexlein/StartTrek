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

SEED = 1

def make_env(model_name: str, env_id: str, video_folder: str):
    """
    Create a Gymnasium environment with video recording for evaluation.

    Args:
        model_name (str): The name of the model being evaluated, used for the video folder.

    Returns:
        gym.Env: The wrapped Gymnasium environment for recording evaluation videos.
    """
    env = gym.make(id=env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env=env,
        video_folder=os.path.join(video_folder, model_name),
        name_prefix="eval",
        episode_trigger=lambda episode_id: True,
    )
    return env


def evaluate(artifact: Artifacts):
    """
    Main function to evaluate a trained DQN model.

    Parses command-line arguments to find the model path, loads the trained
    weights into a DQNAgent, and runs evaluation episodes while recording video.

    Returns:
        int: Exit status code (0 for success, 84 for error).
    """
    model_path = artifact.final_model_path
    if model_path is None:
        print("No final model found in artifact -> exit...", file=sys.stderr)
        return 84

    settings = load_settings()
    env_id = settings["environment"]["env_id"]
    n_episodes = settings["evaluation"]["n_episodes"]

    env = make_env(artifact.final_model_name, env_id, artifact.videos_folder)

    print(f"Model found : {artifact.final_model_name}")
    print(f"Loading from: {model_path}")
    agent = DQNAgent(state_dim=8, action_dim=4)
    agent.policy_net.load_state_dict(torch.load(model_path))
    agent.policy_net.eval()
    agent.epsilon = 0.0

    print("Evaluating...")
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=SEED + ep)
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

    sys.exit(evaluate(artifact))
