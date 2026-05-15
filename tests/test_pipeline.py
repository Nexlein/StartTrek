##
## EPITECH PROJECT, 2025
## StartTrek
## File description:
## test_pipeline.py
##

import os
import yaml
import torch
import artifacts
from train import train
from eval import eval_model
from artifacts import Artifacts
from model.agent import DQNAgent


def test_train_short_run(tmp_path):
    """
    Verifies that a short training session correctly produces
    and saves a model file using a local config.

    Validates the entire training pipeline, from environment
    interaction to weight serialization, in a realistic file-based setup.
    """
    original_base = artifacts.ARTIFACTS_FOLDER

    try:
        artifacts.ARTIFACTS_FOLDER = str(tmp_path) + "/"

        tmp_config_dir = tmp_path / "configs"
        tmp_config_dir.mkdir()

        settings_file = tmp_config_dir / "settings.yml"
        settings_data = {
            "environment": {"env_id": "LunarLander-v3"},
            "training": {"max_episodes": 3},
            "evaluation": {"n_episodes": 1},
        }
        with open(settings_file, "w") as f:
            yaml.dump(settings_data, f)

        hp_file = tmp_config_dir / "hyperparameters.yml"
        hp_data = {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "batch_size": 32,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.1,
            "exploration_fraction": 0.5,
            "target_update_frequency": 10,
        }
        with open(hp_file, "w") as f:
            yaml.dump(hp_data, f)

        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            art = Artifacts()
            train(artifact=art, cli_seed=0)
            assert len(art.get_models()) > 0
        finally:
            os.chdir(original_cwd)

    finally:
        artifacts.ARTIFACTS_FOLDER = original_base


def test_eval_runs_with_saved_model(tmp_path):
    """
    Verifies that eval_model can load a saved model and execute a
    full episode using a temporary configuration file.
    """
    original_base = artifacts.ARTIFACTS_FOLDER
    original_cwd = os.getcwd()

    try:
        artifacts.ARTIFACTS_FOLDER = str(tmp_path) + "/"

        tmp_config_dir = tmp_path / "configs"
        tmp_config_dir.mkdir()
        cfg_file = tmp_config_dir / "settings.yml"

        test_data = {
            "environment": {"env_id": "LunarLander-v3", "seed": 42},
            "evaluation": {"n_episodes": 1, "enable_wind": False},
        }

        with open(cfg_file, "w") as f:
            yaml.dump(test_data, f)

        os.chdir(tmp_path)

        try:
            art = Artifacts()

            agent = DQNAgent(state_dim=8, action_dim=4)
            model_name = "best_model_test.pth"
            model_path = os.path.join(art.models_folder, model_name)
            torch.save(agent.policy_net.state_dict(), model_path)

            folder_name = os.path.basename(
                os.path.dirname(art.logs_folder.rstrip("/\\"))
            )
            art_with_model = Artifacts(load_path=folder_name)

            exit_code = eval_model(artifact=art_with_model, cli_seed=42)

            assert exit_code == 0
            assert os.path.exists(art_with_model.videos_folder)

        finally:
            os.chdir(original_cwd)

    finally:
        artifacts.ARTIFACTS_FOLDER = original_base
