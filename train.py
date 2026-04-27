import os
import sys
import gymnasium as gym
from stable_baselines3 import DQN
from utils import MODEL_FOLDER, get_model_path

ENV_ID = "LunarLander-v3"
SEED = 42

POLICY = "MlpPolicy"
TOTAL_STEPS = 200_000


def make_env():
    env = gym.make(id=ENV_ID)
    return env


def main():
    argv = sys.argv
    if len(argv) > 2:
        print("USAGE:\n\tpython3 ./train.py [modelPath | None]")

    env = make_env()

    os.makedirs(MODEL_FOLDER, exist_ok=True)

    model_path = get_model_path(argv)
    if os.path.exists(model_path):
        print(f"Model find at {model_path} -> loading...")
        model = DQN.load(model_path, env=env)
    else:
        print(f"Model not find at {model_path} -> creation...")
        model = DQN(policy=POLICY, seed=SEED, env=env)

    print("Training...")
    model.learn(total_timesteps=TOTAL_STEPS, reset_num_timesteps=False)

    print("Saving model...")
    model.save(model_path)

    env.close()
    return 0


if __name__ == "__main__":
    main()
