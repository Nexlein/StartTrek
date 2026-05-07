##
## EPITECH PROJECT, 2026
## StartTrek
## File description:
## eval
##

import os
import sys
import gymnasium as gym
import torch
from model.agent import DQNAgent
from utils import get_name_from_path, get_model_path, load_settings

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


def main():
    """
    Main function to evaluate a trained DQN model.

    Parses command-line arguments to find the model path, loads the trained
    weights into a DQNAgent, and runs evaluation episodes while recording video.

    Returns:
        int: Exit status code (0 for success, 84 for error).
    """
    argv = sys.argv
    if len(argv) > 2:
        print("USAGE:\n\tpython3 ./eval.py [modelPath | None]")
        return 84

    settings = load_settings()
    env_id = settings["environment"]["env_id"]
    video_folder = settings["paths"]["video_folder"]
    n_episodes = settings["evaluation"]["n_episodes"]

    model_path = get_model_path(argv)
    model_name = get_name_from_path(model_path)
    env = make_env(model_name, env_id, video_folder)

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
    main()
