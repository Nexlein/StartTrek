##
## EPITECH PROJECT, 2025
## StartTrek
## File description:
## test_config.py
##

from utils import load_hyperparameters, load_settings


def test_config_loading():
    """
    Verifies that the YAML configuration files are existing, accessible,
    and correctly parsed into Python dictionaries.

    Prevents the entire pipeline from failing at launch due to missing files
    or syntax errors in the configuration.
    """
    hp = load_hyperparameters("configs/hyperparameters.yml")
    settings = load_settings("configs/settings.yml")

    assert isinstance(hp, dict)
    assert isinstance(settings, dict)
    assert "environment" in settings


def test_hyperparameters_required_keys():
    """
    Ensures that all mandatory hyperparameter keys (like learning rate or gamma)
    are defined in the YAML file.

    Avoids "KeyError" crashes mid-training when the agent tries to access
    a missing optimization parameter.
    """
    hp = load_hyperparameters("configs/hyperparameters.yml")
    required = [
        "learning_rate",
        "gamma",
        "batch_size",
        "exploration_initial_eps",
        "exploration_final_eps",
    ]

    for key in required:
        assert key in hp


def test_settings_required_keys():
    """
    Confirms that the global project settings, such as environment IDs
    and training limits, are properly specified.

    Guarantees that the training and evaluation scripts have all the necessary
    metadata to initialize the simulation correctly.
    """
    settings = load_settings("configs/settings.yml")

    assert "environment" in settings
    assert "env_id" in settings["environment"]
    assert "training" in settings
    assert "max_episodes" in settings["training"]


def test_config_value_types():
    """
    Validates that hyperparameter values fall within logical
    and mathematical ranges (e.g., probability between 0 and 1).

    Acts as a "sanity check" to catch typos (like a negative learning rate)
    that would lead to divergent training or silent bugs.
    """
    hp = load_hyperparameters("configs/hyperparameters.yml")

    assert 0 < hp["learning_rate"] < 1
    assert 0 < hp["gamma"] <= 1
    assert hp["batch_size"] > 0
