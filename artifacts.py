##
## EPITECH PROJECT, 2025
## StartTrek
## File description:
## artifacts.py
##

import os
import re
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

    _model_name: str | None
    _has_name_given: bool

    _logs_name: str = "logs.csv"
    _report_name: str = "report.md"

    def __init__(
        self,
        configs: List[str] | None = None,
        load_path: str | None = None,
        log_header: str = "Episode,Reward,Length,Epsilon",
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
        self._model_name = None
        self._has_name_given = False

        if load_path is not None:
            self._load(load_path)
        else:
            self._create(configs or [], log_header)

    def _create(self, configs: List[str], log_header: str):
        """Create a brand-new artifact with timestamped folder."""
        self._create_artifact_name()

        self._folder = ARTIFACTS_FOLDER + self._name
        self._make_subdirs()

        for file in configs:
            if os.path.exists(file):
                shutil.copyfile(file, self._configs_folder + os.path.basename(file))

        self._init_log(log_header)

    def _load(self, load_path: str):
        """Load an existing artifact from load_path."""
        stripped = load_path.rstrip("/")
        contents = stripped.split("/")
        for content in contents:
            name = re.search(r'(\d+-\d+-\d+_\d+:\d+:\d+)', content)
            if name:
                self._name = name.string + "/"
            if content.endswith(".pth"):
                self._model_name = content
                self._has_name_given = True

        if self._name is None:
            self._create_artifact_name()
        else:
            self._date = self._name.split("_")[0]

        self._folder = ARTIFACTS_FOLDER + self._name
        self._make_subdirs()

        if self._model_name is None and os.path.exists(self._models_folder):
            for file in os.listdir(self._models_folder):
                if file.startswith("best_model"):
                    self._model_name = file
                    break
                if (self._model_name is None or self._model_name.startswith("checkpoint_model")) \
                and file.startswith("final_model"):
                    self._model_name = file
                if self._model_name is None and file.startswith("checkpoint_model"):
                    self._model_name = file
        print(f"Model found: {self._model_name}")

    @property
    def videos_folder(self) -> str:
        """Absolute path to the videos sub-folder of this artifact."""
        return self._videos_folder

    @property
    def models_folder(self) -> str:
        """Absolute path to the models sub-folder of this artifact."""
        return self._models_folder

    @property
    def model_path(self) -> str | None:
        """
        Full path to the saved model, or None if not yet saved / not found.
        """
        if self._model_name is None:
            return None
        return self._models_folder + self._model_name

    @property
    def model_name(self) -> str:
        """Name of the saved model, or None if not yet saved / not found."""
        if self._model_name is None:
            return None
        return self._model_name

    @property
    def name_given(self) -> bool:
        """True if the model's name was given when loading the artifact."""
        return self._has_name_given

    @property
    def logs_folder(self) -> str:
        """Absolute path to the logs sub-folder of this artifact."""
        return self._logs_folder

    @property
    def configs_folder(self) -> str:
        """Absolute path to the configs sub-folder of this artifact."""
        return self._configs_folder

    def _create_artifact_name(self):
        gmt = time.gmtime()
        self._date = "-".join([str(gmt.tm_year), str(gmt.tm_mon), str(gmt.tm_mday)])
        current_time = ":".join([str(gmt.tm_hour), str(gmt.tm_min), str(gmt.tm_sec)])
        self._name = "_".join([self._date, current_time]) + "/"
        return self._name

    def _make_subdirs(self):
        """Create the standard sub-directories."""
        self._configs_folder = self._folder + "configs/"
        self._models_folder = self._folder + "models/"
        self._videos_folder = self._folder + "videos/"
        self._logs_folder = self._folder + "logs/"

        os.makedirs(self._configs_folder, exist_ok=True)
        os.makedirs(self._models_folder, exist_ok=True)
        os.makedirs(self._videos_folder, exist_ok=True)
        os.makedirs(self._logs_folder, exist_ok=True)

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

    def save_best_model(self, model: object, seed: int, episode: int):
        fileName = f"best_model_seed_{seed}_ep_{episode}.pth"
        torch.save(model, self._models_folder + fileName)
        self._model_name = fileName

    def get_models(self) -> List[str]:
        if not os.path.exists(self._models_folder):
            return []
        models = []
        for f in os.listdir(self._models_folder):
            if f.startswith("best_model") or f.startswith("final_model"):
                models.append(f)
        return models

    def generate_report(self):
        replace_variables = {
            "[date]": self._date,
            "[run_name]": self._name,
        }

        template_path = ARTIFACTS_FOLDER + "Report.md.template"
        report_path = self._folder + self._report_name

        with open(template_path, "r") as template, open(report_path, "w") as report:
            for line in template:
                for variable, value in replace_variables.items():
                    if variable in line:
                        line = line.replace(variable, value)
                report.write(line)
