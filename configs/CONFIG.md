# Configuration Reference

This folder contains the YAML configuration files used to centralize runtime options, training hyperparameters, and filesystem paths for the StartTrek project. Editing these files lets you change behavior without modifying the Python code.

## **settings.yml**

- **environment.env_id:** The Gymnasium/Environment ID to use. Default: `LunarLander-v3`.
- **environment.enable_wind:** Whether to enable wind dynamics in the environment (e.g., for LunarLander). Default: `false`.
- **environment.wind_power:** The strength of the wind if enabled. Default: `15.0`.
- **environment.seed:** Seed for random number generators (numpy, torch, gym) for reproducibility. Default: `1`.
- **training.max_episodes:** Maximum number of training episodes (integer).
- **evaluation.n_episodes:** Number of evaluation episodes to run when testing a trained agent.
- **paths.video_folder / paths.model_folder:** Output folders for recorded videos and saved models. Provide relative paths.

## **hyperparameters.yml**

- **learning_rate:** Optimizer learning rate. Example: `0.0001`.
- **policy_kwargs.net_arch:** Neural network hidden layer sizes (list of integers). Example: `[256, 256]`.
- **learning_starts:** Number of environment steps before training begins (replay buffer warm-up).
- **batch_size:** Mini-batch size sampled from the replay buffer for each training update.
- **train_freq / gradient_steps:** Frequency (in environment steps) to train and the number of gradient steps per training call.
- **target_update_interval:** Steps between target network updates.
- **exploration_initial_eps / exploration_final_eps / exploration_fraction:** Epsilon-greedy schedule: start, end values and fraction of total training for the decay.
- **gamma:** Discount factor for the Bellman equation (e.g., `0.99`).

Notes

- Keep configuration values consistent with your environment and compute budget. Large networks and batch sizes need more memory and compute.
- Add comments in the YAML files if you need to document experiments or non-default settings.
- These files are loaded at runtime by the training/evaluation scripts; changing them will affect subsequent runs.
