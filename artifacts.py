##
## EPITECH PROJECT, 2025
## StartTrek
## File description:
## artifacts.py
##

import os
import time
import torch
import shutil
from typing import List

ARTIFACTS_FOLDER = "artifacts/"

class Artifacts:
    _date: str
    _name: str
    _folder: str

    _configs_folder: str
    _models_folder: str
    _videos_folder: str
    _logs_folder: str

    final_model_name: str | None

    _logs_name: str = "logs.csv"
    _report_name: str = "report.md"

    def __init__(
        self,
        configs: List[str] = None,
        load_path: str = None,
        log_header: str = "Episode,Reward,Length,Epsilon"
    ):
        """
        Initialise or load an artifact.

        Parameters
        ----------
        configs : List[str], optional
            List of paths to config files to copy into the artifact folder.
            Only used when creating a new artifact (i.e. load_path is None).
        load_path : str, optional
            Path to an existing artifact folder to load.
            When provided, the artifact is loaded instead of created.
        log_header : str
            Header of logs.csv: column names separated by commas.
        """
        self.final_model_name = None

        if load_path is not None:
            self._load(load_path)
        else:
            self._create(configs or [], log_header)

    def _create(self, configs: List[str], log_header: str):
        """Create a brand-new artifact with timestamped folder."""
        gmt = time.gmtime()
        self._date = "-".join([str(gmt.tm_year), str(gmt.tm_mon), str(gmt.tm_mday)])
        current_time = ":".join([str(gmt.tm_hour), str(gmt.tm_min), str(gmt.tm_sec)])
        self._name = "_".join([self._date, current_time]) + "/"

        self._folder = ARTIFACTS_FOLDER + self._name
        self._make_subdirs()

        for file in configs:
            if os.path.exists(file):
                shutil.copyfile(file, self._configs_folder + os.path.basename(file))

        self._init_log(log_header)

    def _load(self, load_path: str):
        """Load an existing artifact from load_path."""
        stripped = load_path.rstrip("/")
        self._name = stripped.split("/")[-1] + "/"
        self._date = self._name.split("_")[0]

        self._folder = ARTIFACTS_FOLDER + self._name
        self._make_subdirs()

        if os.path.exists(self._models_folder):
            for file in os.listdir(self._models_folder):
                if file.startswith("final_model"):
                    self.final_model_name = file
                    print(f"Final model found: {self.final_model_name}")
                    break

    @property
    def videos_folder(self) -> str:
        """Absolute path to the videos sub-folder of this artifact."""
        return self._videos_folder

    @property
    def models_folder(self) -> str:
        """Absolute path to the models sub-folder of this artifact."""
        return self._models_folder

    @property
    def logs_folder(self) -> str:
        """Absolute path to the logs sub-folder of this artifact."""
        return self._logs_folder

    @property
    def configs_folder(self) -> str:
        """Absolute path to the configs sub-folder of this artifact."""
        return self._configs_folder

    @property
    def final_model_path(self) -> str | None:
        """
        Full path to the final saved model, or None if not yet saved / not found.
        """
        if self.final_model_name is None:
            return None
        return self._models_folder + self.final_model_name

    def _make_subdirs(self):
        """Create the standard sub-directories."""
        self._configs_folder = self._folder + "configs/"
        self._models_folder  = self._folder + "models/"
        self._videos_folder  = self._folder + "videos/"
        self._logs_folder    = self._folder + "logs/"

        os.makedirs(self._configs_folder, exist_ok=True)
        os.makedirs(self._models_folder,  exist_ok=True)
        os.makedirs(self._videos_folder,  exist_ok=True)
        os.makedirs(self._logs_folder,    exist_ok=True)

    def _init_log(self, log_header: str):
        """Write the CSV header if the log file does not exist yet."""
        log_path = self._logs_folder + self._logs_name
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write(f"{log_header}\n")

    def log_step(self, values: List):
        line = ",".join(str(v) for v in values)
        with open(self._logs_folder + self._logs_name, "a") as f:
            f.write(line + "\n")

    def save_checkpoint_model(self, model: object, seed: int, episode: int):
        fileName = f"checkpoint_model_seed_{seed}_ep_{episode}.pth"
        torch.save(model, self._models_folder + fileName)

    def save_final_model(self, model: object, seed: int, episode: int):
        fileName = f"final_model_seed_{seed}_ep_{episode}.pth"
        torch.save(model, self._models_folder + fileName)
        self.final_model_name = fileName

    def get_all_final_models(self) -> List[str]:
        if not os.path.exists(self._models_folder):
            return []
        return [f for f in os.listdir(self._models_folder) if f.startswith("final_model")]

    def generate_report(self):
        replace_variables = {
            "[date]":     self._date,
            "[run_name]": self._name,
        }

        template_path = ARTIFACTS_FOLDER + "Report.md.template"
        report_path   = self._folder + self._report_name

        with open(template_path, "r") as template, \
             open(report_path,   "w") as report:
            for line in template:
                for variable, value in replace_variables.items():
                    if variable in line:
                        line = line.replace(variable, value)
                report.write(line)
