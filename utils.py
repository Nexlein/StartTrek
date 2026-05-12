##
## EPITECH PROJECT, 2026
## StartTrek
## File description:
## utils
##

from typing import Dict, Any, Optional
import yaml


def get_name_from_path(path: str) -> str:
    """Extract the model name from a given file path by removing the extension."""
    filename = path.split("/")[-1]
    parts = filename.split(".")
    if len(parts) == 1:
        return parts[0]
    return "".join(parts[:-1])


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


def get_model_path(cli_model_path: Optional[str] = None) -> str:
    """Determine the model path based on command-line arguments and settings."""
    settings = load_settings()
    model_folder = settings["paths"]["model_folder"]
    model_ext = ".pth"
    default_name = "dqn-model"

    if cli_model_path and cli_model_path.endswith(model_ext):
        return cli_model_path

    return model_folder + default_name + model_ext
