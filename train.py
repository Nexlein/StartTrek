import os
import gymnasium as gym
from stable_baselines3 import DQN

ENV_ID          = "LunarLander-v3"
VIDEO_FOLDER    = "results/videos/"
MAX_VIDEOS      = 10

MODEL_NAME      = "model-v1"
MODEL_FOLDER    = "results/models/"
MODEL_PATH      = MODEL_FOLDER + MODEL_NAME + ".zip"

POLICY          = "MlpPolicy"
TOTAL_STEPS     = 50_000


def count_videos(folder: str):
    dir_list = os.listdir(folder)
    count = 0
    for file in dir_list:
        if file.endswith(".mp4"):
            count += 1
    return count


def make_env():
    env = gym.make(id=ENV_ID, render_mode="rgb_array")

    env = gym.wrappers.RecordVideo(
        env=env,
        video_folder=VIDEO_FOLDER,
        name_prefix=MODEL_NAME,
        episode_trigger=lambda episode_id: episode_id < MAX_VIDEOS
    )

    return env


def main():
    env = make_env()

    os.makedirs(VIDEO_FOLDER, exist_ok=True)
    os.makedirs(MODEL_FOLDER, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        print(f"Model: {MODEL_NAME} find at {MODEL_PATH} -> loading...")
        model = DQN.load(MODEL_PATH, env=env)
    else:
        print(f"Model: {MODEL_NAME} not find at {MODEL_PATH} -> creation...")
        model = DQN(policy=POLICY, env=env)

    while True:
        nb_videos = count_videos(VIDEO_FOLDER)
        if nb_videos >= MAX_VIDEOS:
            print(f"Maximum videos reached ({MAX_VIDEOS} videos).")
            break

        print("Training...")
        model.learn(total_timesteps=TOTAL_STEPS, reset_num_timesteps=False)

    print(f"Saving model {MODEL_NAME}...")
    model.save(MODEL_PATH)

    env.close()
    return 0


if __name__ == "__main__":
    main()
