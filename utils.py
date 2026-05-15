##
## EPITECH PROJECT, 2026
## StartTrek
## File description:
## utils
##

import os
import yaml
import gymnasium as gym
from typing import Dict, Any


def load_hyperparameters(
    config_path: str = "configs/hyperparameters.yml",
) -> Dict[str, Any]:
    """Load hyperparameters from a YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_settings(config_path: str = "configs/settings.yml") -> Dict[str, Any]:
    """Load general project settings from a YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_video_env(
    env_id: str,
    base_folder: str,
    mode: str,
    model_name: str | None = None,
    seed: int | None = None,
    enable_wind: bool = False,
    wind_power: float = 15.0,
    video_freq: int = 1,
):
    path_parts = [base_folder, mode]
    if model_name:
        path_parts.append(model_name)
    if seed is not None:
        path_parts.append(f"seed_{seed}")

    video_path = os.path.join(*path_parts)

    render_mode = "rgb_array" if video_freq > 0 else None
    env = gym.make(
        env_id, render_mode=render_mode, enable_wind=enable_wind, wind_power=wind_power
    )

    if video_freq > 0:
        env = gym.wrappers.RecordVideo(
            env=env,
            video_folder=video_path,
            name_prefix=f"{mode}{'_' if mode and model_name else ''}{model_name if model_name else ''}",
            episode_trigger=lambda x: x % video_freq == 0,
        )
    return env
