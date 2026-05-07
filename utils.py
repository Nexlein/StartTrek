##
## EPITECH PROJECT, 2026
## StartTrek
## File description:
## utils
##

from typing import List, Dict, Any
import yaml

MODEL_FOLDER = "results/models/"
DEFAULT_NAME = "dqn-model"
MODEL_EXT = ".pth"


def get_name_from_path(path: str) -> str:
    """
    Extract the model name from a given file path by removing the extension.

    Args:
        path (str): The file path.

    Returns:
        str: The extracted name of the model.
    """
    filename = path.split("/")[-1]

    parts = filename.split(".")

    if len(parts) == 1:
        return parts[0]
    return "".join(parts[:-1])


def get_model_path(argv: List[str]) -> str:
    """
    Determine the model path based on command-line arguments.

    Args:
        argv (List[str]): The list of command-line arguments.

    Returns:
        str: The determined model path. Uses a default path if not provided in arguments.
    """
    if len(argv) == 2 and argv[1].endswith(MODEL_EXT):
        return argv[1]

    return MODEL_FOLDER + DEFAULT_NAME + MODEL_EXT


def load_hyperparameters() -> Dict[str, Any]:
    """
    Load hyperparameters from a YAML configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the parsed hyperparameters.
    """
    hyperparameters_path = "config/hyperparameters.yml"
    with open(hyperparameters_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
