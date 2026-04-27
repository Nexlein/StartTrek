##
## EPITECH PROJECT, 2026
## StartTrek
## File description:
## logger
##

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback


class EpisodeCsvLoggerCallback(BaseCallback):
    def __init__(self, csv_path: str):
        super().__init__()
        self.csv_path = csv_path
        self._episode_count = 0
        self._csv_file: Any = None
        self._csv_writer: Any = None

    @staticmethod
    def _infer_status(episode_reward: float, truncated: bool) -> str:
        if truncated:
            return "Timeout"
        if episode_reward >= 200.0:
            return "Landed"
        return "Crashed"

    def _on_training_start(self) -> None:
        file_exists = Path(self.csv_path).exists()
        self._csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)

        if not file_exists:
            self._csv_writer.writerow(
                [
                    "Time",
                    "Episode",
                    "TotalTimesteps",
                    "Status",
                    "EpisodeReward",
                    "EpisodeLength",
                    "EpisodeDurationSec",
                ]
            )
            self._csv_file.flush()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, done in enumerate(dones):
            if not done:
                continue

            info = infos[i] if i < len(infos) else {}
            episode_info = info.get("episode")

            if not episode_info:
                continue

            episode_reward = float(episode_info.get("r", 0.0))
            episode_length = int(episode_info.get("l", 0))
            episode_duration = float(episode_info.get("t", 0.0))

            truncated = bool(info.get("TimeLimit.truncated", False))
            status = self._infer_status(episode_reward, truncated)
            self._episode_count += 1

            if self._csv_writer is not None:
                self._csv_writer.writerow(
                    [
                        datetime.now().isoformat(timespec="seconds"),
                        self._episode_count,
                        self.num_timesteps,
                        status,
                        episode_reward,
                        episode_length,
                        episode_duration,
                    ]
                )
                self._csv_file.flush()

        return True

    def _on_training_end(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
