##
## EPITECH PROJECT, 2026
## StartTrek
## File description:
## eval
##

import os
import sys
import yaml
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
    model_name_prop = artifact.model_name
    if not artifact.name_given or model_name_prop is None:
        model_files = artifact.get_models()
        if not model_files:
            print("[EVAL] Error: No models found to evaluate.", file=sys.stderr)
            return 84
    else:
        model_files = [model_name_prop]

    settings = load_settings()
    env_id = settings["environment"]["env_id"]
    n_episodes = settings["evaluation"]["n_episodes"]

    if cli_wind is None:
        enable_wind = settings["evaluation"].get("enable_wind", False)
        wind_power = settings["evaluation"].get("wind_power", 15.0)
    elif cli_wind == -1.0:
        enable_wind = True
        wind_power = settings["evaluation"].get("wind_power", 15.0)
    else:
        enable_wind = True
        wind_power = cli_wind

    seed_value = (
        cli_seed if cli_seed is not None else settings["environment"].get("seed", 1)
    )

    if not artifact.name_given and cli_seed is not None:
        filtered_models = [m for m in model_files if f"seed_{cli_seed}" in m]
        best_models = [m for m in filtered_models if m.startswith("best_model")]
        if best_models:
            model_files = [best_models[-1]]
        elif filtered_models:
            model_files = [filtered_models[-1]]

    # Reflect evaluation overrides in a specific eval settings file within the artifact
    settings["environment"]["seed"] = seed_value
    settings["evaluation"]["enable_wind"] = enable_wind
    settings["evaluation"]["wind_power"] = wind_power
    with open(
        os.path.join(artifact.configs_folder, "eval_settings.yml"),
        "w",
        encoding="utf-8",
    ) as f:
        yaml.dump(settings, f, default_flow_style=False)

    print(
        f"[EVAL] Starting evaluation on {env_id} | Seed: {seed_value} | Wind: {enable_wind} (Power: {wind_power})"
    )
    print(f"[EVAL] Videos will be saved to: {artifact.videos_folder}")
    env = None
    for model_file in model_files:
        model_name = model_file.replace(".pth", "")
        model_path = os.path.join(artifact.models_folder, model_file)

        env = make_video_env(
            env_id=env_id,
            base_folder=artifact.videos_folder,
            mode="eval",
            model_name=model_name,
            enable_wind=enable_wind,
            wind_power=wind_power,
        )

        print(f"[EVAL] Targeting Model: {model_name}")
        print(f"[EVAL] Loading weights from: {model_path}")

        agent = DQNAgent(state_dim=8, action_dim=4)
        agent.policy_net.load_state_dict(
            torch.load(model_path, map_location=agent.device, weights_only=True)
        )
        agent.policy_net.eval()
        agent.epsilon = 0.0

        eval_log_path = os.path.join(
            artifact.logs_folder, f"eval_scores_seed_{seed_value}.csv"
        )
        with open(eval_log_path, "w") as f:
            f.write("Episode,Reward\n")

        print(f"[EVAL] Running {n_episodes} episodes...")
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed_value + ep)
            done = False
            episode_reward = 0.0

            while not done:
                action = agent.select_action(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += float(reward)

                if terminated or truncated:
                    done = True

            with open(eval_log_path, "a") as f:
                f.write(f"{ep + 1},{episode_reward:.2f}\n")
            print(
                f"[EVAL] Episode {ep + 1}/{n_episodes} | Reward: {episode_reward:7.2f}"
            )

        if env is not None:
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

    sys.exit(eval_model(artifact=artifact_obj, cli_seed=args.seed, cli_wind=args.wind))
