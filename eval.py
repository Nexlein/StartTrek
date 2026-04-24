import os
import sys
import gymnasium as gym
from stable_baselines3 import DQN
from utils import get_name_from_path, get_model_path

ENV_ID          = "LunarLander-v3"
SEED            = 42
N_EPISODE       = 10

VIDEO_FOLDER    = "results/videos/"


def make_env(model_name: str):
    env = gym.make(id=ENV_ID, render_mode="rgb_array")

    env = gym.wrappers.RecordVideo(
        env=env,
        video_folder=VIDEO_FOLDER + model_name,
        name_prefix="eval",
        episode_trigger=lambda episode_id: True
    )

    return env


def main():
    argv = sys.argv
    if len(argv) > 2:
        print("USAGE:\n\tpython3 ./train.py [modelPath | None]")
        return 84

    model_path = get_model_path(argv)
    model_name = get_name_from_path(model_path)
    env = make_env(model_name)

    os.makedirs(VIDEO_FOLDER, exist_ok=True)

    if os.path.exists(model_path):
        print(f"Model find at {model_path} -> loading...")
        model = DQN.load(model_path, env=env)
    else:
        print(f"Model not find at {model_path} -> exit...")
        print("USAGE:\n\tpython3 ./train.py [modelPath | None]")
        return 84

    print("Evaluating...")
    for ep in range(N_EPISODE):
        obs, _ = env.reset(seed=SEED + ep)
        done = False

        print(f"Episode {ep} / {N_EPISODE}")
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                done = True

    env.close()
    return 0


if __name__ == "__main__":
    main()
