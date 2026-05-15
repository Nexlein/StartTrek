##
## EPITECH PROJECT, 2025
## StartTrek
## File description:
## test_artifacts.oy
##

import os
import csv
import artifacts
from artifacts import Artifacts
from model.agent import DQNAgent


def test_artifact_architecture(tmp_path):
    """
    Verifies that the mandatory directory structure (models, logs, configs)
    is automatically created upon initialization.

    Ensures the project environment is correctly set up to prevent
    "File Not Found" errors during training or evaluation.
    """
    original_base = artifacts.ARTIFACTS_FOLDER
    artifacts.ARTIFACTS_FOLDER = str(tmp_path) + "/"
    art = Artifacts(configs=[])

    assert os.path.exists(art.models_folder)
    assert os.path.exists(art.logs_folder)
    assert os.path.exists(art.configs_folder)

    artifacts.ARTIFACTS_FOLDER = original_base


def test_artifact_log_step(tmp_path):
    """
    Validates that training metrics are correctly formatted and appended
    to the logs.csv file.

    Guarantees that experiment data is safely persisted, which is critical
    for generating performance graphs and the final report.
    """
    original_base = artifacts.ARTIFACTS_FOLDER
    artifacts.ARTIFACTS_FOLDER = str(tmp_path) + "/"

    art = Artifacts(log_header="Episode,Reward")
    art.log_step([1, 42.5])
    log_path = os.path.join(art.logs_folder, "logs.csv")

    with open(log_path, "r") as f:
        rows = list(csv.reader(f))

    assert rows[1] == ["1", "42.5"]

    artifacts.ARTIFACTS_FOLDER = original_base


def test_artifact_save_and_load_best_model(tmp_path):
    """
    Confirms that the agent's best model weights are successfully saved
    to the disk and the path is correctly resolved.

    Prevents the loss of progress by ensuring the "Best Model" is actually
    retrievable for future deployment or testing.
    """
    original_base = artifacts.ARTIFACTS_FOLDER
    artifacts.ARTIFACTS_FOLDER = str(tmp_path) + "/"

    art = Artifacts()
    agent = DQNAgent(state_dim=8, action_dim=4)
    art.save_best_model(agent.policy_net.state_dict(), seed=0)

    assert art.model_path is not None
    assert os.path.exists(art.model_path)

    artifacts.ARTIFACTS_FOLDER = original_base
