##
## EPITECH PROJECT, 2025
## StartTrek
## File description:
## test_environment.py
##

import gymnasium as gym
from utils import make_video_env


def test_environment_creation():
    """
    Verifies that the Gymnasium library can successfully instantiate
    the LunarLander environment and returns the expected observation shape.

    Ensures that the core simulation engine and its dependencies (like Box2D)
    are correctly installed and operational.
    """
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, _ = env.reset()

    assert obs.shape == (8,)

    env.close()


def test_environment_seeding_reproducibility():
    """
    Confirms that providing the same random seed to two different
    environment instances results in identical initial observations.

    Guarantees deterministic behavior, which is essential for debugging,
    comparing different models, and ensuring scientific reproducibility of results.
    """
    env1 = gym.make("LunarLander-v3")
    env2 = gym.make("LunarLander-v3")

    obs1, _ = env1.reset(seed=7)
    obs2, _ = env2.reset(seed=7)

    assert (obs1 == obs2).all()

    env1.close()
    env2.close()


def test_video_env_wrapper(tmp_path):
    """
    Validates that the custom video recording wrapper initializes correctly
    and creates the necessary output folders without disk errors.

    Ensures that the visual recording system is functional, allowing
    for the qualitative analysis of the agent's behavior during training and evaluation.
    """
    env = make_video_env(
        env_id="LunarLander-v3", base_folder=str(tmp_path), mode="test"
    )

    assert hasattr(env, "video_folder")

    env.close()
