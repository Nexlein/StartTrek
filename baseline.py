##
## EPITECH PROJECT, 2026
## StartTrek
## File description:
## baseline
##

import os
import gymnasium as gym

ENV_ID = "LunarLander-v3"
VIDEO_FOLDER = "results/videos/"


def make_video_env(name: str):
    """
    Create a Gymnasium environment with video recording enabled.

    Args:
        name (str): The name prefix for the recorded video files and folder.

    Returns:
        gym.Env: The wrapped Gymnasium environment capable of recording videos.
    """
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env=env,
        video_folder=os.path.join(VIDEO_FOLDER, name),
        name_prefix=name,
        episode_trigger=lambda x: True,
    )
    return env


def random_policy():
    """
    Run a baseline agent that selects actions completely at random.

    This policy executes 3 episodes in the environment, sampling actions
    uniformly from the action space, and saves the recordings.
    """
    print("--- Running Random Policy Baseline ---")
    env = make_video_env("baseline_random")

    for ep in range(3):
        env.reset(seed=1 + ep)
        done = False
        while not done:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    env.close()


def heuristic_policy():
    """
    Run a baseline agent that uses a hardcoded heuristic to select actions.

    The heuristic attempts to balance the lander by firing side engines
    if the angle is too steep, or the main engine if it is falling too fast.
    Executes 3 episodes and saves the recordings.
    """
    print("--- Running Heuristic Policy Baseline ---")
    env = make_video_env("baseline_heuristic")

    for ep in range(3):
        obs, _ = env.reset(seed=1 + ep)
        done = False
        while not done:
            # State : [x, y, v_x, v_y, angle, v_angle, left_contact, right_contact]
            angle = obs[4]
            v_y = obs[3]

            # Heuristique très basique codée en dur
            if angle < -0.05:
                action = 3  # Start the right engine to correct the angle
            elif angle > 0.05:
                action = 1  # Start the left engine to correct the angle
            elif v_y < -0.5:
                action = 2  # Start the main engine if falling too fast
            else:
                action = 0  # Do nothing

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    env.close()


if __name__ == "__main__":
    random_policy()
    heuristic_policy()
