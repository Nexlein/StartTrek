##
## EPITECH PROJECT, 2026
## StartTrek
## File description:
## train
##

import os
from pathlib import Path
import sys
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from utils import MODEL_FOLDER, get_model_path, load_hyperparameters
from logger import EpisodeCsvLoggerCallback

ENV_ID = "LunarLander-v3"
SEED = 42

POLICY = "MlpPolicy"
TOTAL_STEPS = 1_000_000
EVAL_FREQ = 10_000
CHECKPOINT_FREQ = 50_000
N_EVAL_EPISODES = 5
BEST_MODEL_NAME = "best_model.zip"
TRAIN_LOG_FILE = "trainLogs.csv"

MODEL_SAVE_PATH = Path(MODEL_FOLDER)
BEST_MODEL_PATH = MODEL_SAVE_PATH / BEST_MODEL_NAME

DQN_KWARGS = load_hyperparameters()


def make_env():
    env = gym.make(id=ENV_ID)
    env = Monitor(env)
    return env


def main():
    argv = sys.argv
    if len(argv) > 2:
        print("USAGE: python3 ./train.py [modelPath | None]")
        return 84

    env = make_vec_env(make_env, n_envs=1, seed=SEED)
    eval_env = make_vec_env(make_env, n_envs=1, seed=SEED + 1)

    os.makedirs(MODEL_FOLDER, exist_ok=True)

    model_path = get_model_path(argv)
    if os.path.exists(model_path):
        print(f"Model find at {model_path} -> loading...")
        model = DQN.load(model_path, env=env)
    else:
        print(f"Model not find at {model_path} -> creation...")
        model = DQN(policy=POLICY, env=env, **DQN_KWARGS)

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=MODEL_FOLDER,
        log_path=MODEL_FOLDER,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=MODEL_FOLDER,
        name_prefix="checkpoint",
    )
    episode_log_callback = EpisodeCsvLoggerCallback(TRAIN_LOG_FILE)

    print("Training...")
    model.learn(
        total_timesteps=TOTAL_STEPS,
        reset_num_timesteps=False,
        callback=[eval_callback, checkpoint_callback, episode_log_callback],
    )
    print(f"Training finished at {TOTAL_STEPS} timesteps")

    print("Saving model...")
    model.save(model_path)

    if BEST_MODEL_PATH.exists():
        Path(model_path).write_bytes(BEST_MODEL_PATH.read_bytes())

    env.close()
    eval_env.close()
    return 0


if __name__ == "__main__":
    main()
