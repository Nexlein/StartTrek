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
    model_files = artifact.get_all_final_models()
    if not model_files:
        print("No models found to evaluate.", file=sys.stderr)
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

    print(f"Starting evaluating on {env_id} with seed {seed_value}")
    print(f"Artifact videos folder: {artifact.videos_folder}")
    for model_file in model_files:
        model_name = model_file.replace(".pth", "")
        model_path = os.path.join(artifact.models_folder, model_file)

        env = make_video_env(
            env_id=env_id,
            base_folder=artifact.videos_folder,
            mode="eval",
            model_name=model_name,
        )

        print(f"Model found : {artifact.final_model_name}")
        print(f"Loading from: {model_path}")

        agent = DQNAgent(state_dim=8, action_dim=4)
        agent.policy_net.load_state_dict(torch.load(model_path))
        agent.policy_net.eval()
        agent.epsilon = 0.0

        print(f"Evaluating for {n_episodes} episodes (Wind: {enable_wind})...")
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
