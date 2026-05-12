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
from artifacts import Artifacts
from model.agent import DQNAgent
from utils import load_settings, make_video_env

SEED = 1

def evaluate(seed_value: int, artifact: Artifacts):
    """
    Main function to evaluate a trained DQN model.

    Parses command-line arguments to find the model path, loads the trained
    weights into a DQNAgent, and runs evaluation episodes while recording video.

    Returns:
        int: Exit status code (0 for success, 84 for error).
    """
    model_files = artifact.get_all_final_models()
    if not model_files:
        print("No models found to evaluate.", file=sys.stderr)
        return 84

    settings = load_settings()
    env_id = settings["environment"]["env_id"]
    n_episodes = settings["evaluation"]["n_episodes"]

    print(f"Starting evaluating on {env_id} with seed {seed_value}")
    print(f"Artifact folder: {artifact.videos_folder}")
    for model_file in model_files:
        model_name = model_file.replace(".pth", "")
        model_path = os.path.join(artifact.models_folder, model_file)

        print(f"Model name: {model_name}")

        env = make_video_env(
            env_id=env_id,
            base_folder=artifact.videos_folder,
            mode="eval",
            model_name=model_name,
            seed=seed_value
        )

        agent = DQNAgent(state_dim=8, action_dim=4)
        agent.policy_net.load_state_dict(torch.load(model_path))
        agent.policy_net.eval()
        agent.epsilon = 0.0

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed_value + ep)
            done = False

            print(f"Episode {ep + 1} / {n_episodes}")
            while not done:
                action = agent.select_action(obs)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

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

    sys.exit(evaluate(args.seed, artifact))
