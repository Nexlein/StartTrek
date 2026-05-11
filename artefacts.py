##
## EPITECH PROJECT, 2025
## StartTrek
## File description:
## artefacts.py
##

import os
import time
import torch
import shutil
from typing import List
from dataclasses import dataclass

ARTEFACTS_FOLDER = "artefacts/"

@dataclass
class Artefacts:
    _name: str
    _folder: str

    _logs_name: str = "logs.csv"
    _report_name: str = "report.md"

    def __init__(
            self,
            configs: List[str] = None, 
            log_header: str = "Episode,Reward,Length,Epsilon"
        ):
        configs = configs or []

        gmt = time.gmtime()
        date = "-".join([str(gmt.tm_year), str(gmt.tm_mon), str(gmt.tm_mday)])
        current_time = ":".join([str(gmt.tm_hour), str(gmt.tm_min), str(gmt.tm_sec)])
        self._name = "_".join([date, current_time]) + "/"

        self._folder = ARTEFACTS_FOLDER + self._name
        os.makedirs(self._folder, exist_ok=True)

        for folder in (["configs", "models", "logs", "eval/videos"]):
            os.makedirs(self._folder + folder, exist_ok=True)

        for file in configs:
            if os.path.exists(file):
                shutil.copyfile(f"configs/{file}", f"{self._folder}configs/{file}")
        
        if not os.path.exists(f"{self._folder}logs/{self._logs_name}"):
            with open(f"{self._folder}logs/{self._logs_name}", "w") as f:
                f.write(f"{log_header}\n")

    def log_step(self, values: List):
        convert = []
        for v in values:
            convert.append(str(v))

        line = ",".join(convert)
        with open(f"{self._folder}logs/{self._logs_name}", "a") as f:
            f.write(line + "\n")
    
    def save_checkpoint_model(self, model: object, seed: int, episode: int):
        fileName = f"checkpoint_model_seed_{seed}_ep_{episode}.pth"
        torch.save(
            model,
            f"{self._folder}models/{fileName}"
        )

    # ce serait cool de pouvoir remplir "Results" aussi
    def generate_report(self):
        with open(f"{ARTEFACTS_FOLDER}ReportTemplate.md", "r") as template, \
         open(f"{self._folder}{self._report_name}", "w") as report:
            for line in template:
                if line.find("2026-XX-XX") != -1:
                    line = line.replace("2026-XX-XX", self._name.split("_")[0])
                if line.find("XX:XX:XX") != -1:
                    line = line.replace("XX:XX:XX", self._name.split("_")[1])
                report.write(line)
