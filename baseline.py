##
## EPITECH PROJECT, 2026
## StartTrek
## File description:
## baseline
##

from artifacts import Artifacts
from utils import load_settings, make_video_env


def random_policy(artifact: Artifacts, seed: int = 0):
    """
    Run a baseline agent that selects actions completely at random.

    This policy executes 3 episodes in the environment, sampling actions
    uniformly from the action space, and saves the recordings.
    """
    settings = load_settings()
    print("--- Running Random Policy Baseline ---")

    env = make_video_env(
        env_id=settings["environment"]["env_id"],
        base_folder=artifact.videos_folder,
        mode="baseline",
        model_name="random",
    )

    for ep in range(3):
        env.reset(seed=1 + ep)
        done = False
        termination_reason = "ongoing"
        while not done:
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if done:
                if terminated:
                    if float(reward) <= -100:
                        termination_reason = "crash"
                    else:
                        termination_reason = "sleep"
                elif truncated:
                    termination_reason = "out-of-view"
        print(f"Random Policy Ep {ep + 1} terminated by: {termination_reason}")
    env.close()


def heuristic_policy(artifact: Artifacts):
    """
    Run a baseline agent that uses a hardcoded heuristic to select actions.

    The heuristic attempts to balance the lander by firing side engines
    if the angle is too steep, or the main engine if it is falling too fast.
    Executes 3 episodes and saves the recordings.
    """
    settings = load_settings()
    print("--- Running Heuristic Policy Baseline ---")

    env = make_video_env(
        env_id=settings["environment"]["env_id"],
        base_folder=artifact.videos_folder,
        mode="baseline",
        model_name="heuristic",
    )

    for ep in range(3):
        obs, _ = env.reset(seed=1 + ep)
        done = False
        termination_reason = "ongoing"
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

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if done:
                if terminated:
                    if float(reward) <= -100:
                        termination_reason = "crash"
                    else:
                        termination_reason = "sleep"
                elif truncated:
                    termination_reason = "out-of-view"
        print(f"Heuristic Policy Ep {ep + 1} terminated by: {termination_reason}")
    env.close()


if __name__ == "__main__":
    artifact = Artifacts(
        configs=["configs/settings.yml", "configs/hyperparameters.yml"]
    )
    random_policy(artifact)
    heuristic_policy(artifact)

    artifact.generate_report()
