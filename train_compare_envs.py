"""Train the same PPO agent on different environments."""
from grid_exp_oe import ExperimentConfig, PPOHparams, run_training_experiment

EXPERIMENT_PATH = "experiments/simple_ppo_envs_comparison_t"
LOGS_PATH = "logs/simple_ppo_envs_comparison_t"


def algo_hparams_fn():
    return PPOHparams(
        total_steps=5_000,
        num_envs=16,
        horizon=128,
        epochs=5,
        batch_size=256,
        annealing_steps=1_000_000,
        final_learning_rate=0.0,
    )


ENVS_IDS = [
    "MiniGrid-LavaCrossingS9N1-v0",
    "MiniGrid-LavaCrossingS9N2-v0",

    "MiniGrid-SimpleCrossingS9N1-v0",
    "MiniGrid-SimpleCrossingS9N2-v0",

    "MiniGrid-DistShift1-v0",

    "MiniGrid-DoorKey-5x5-v0",
    "MiniGrid-DoorKey-8x8-v0",

    "MiniGrid-KeyCorridorS3R1-v0",
    "MiniGrid-KeyCorridorS3R2-v0",

    "MiniGrid-LavaGapS5-v0",
    "MiniGrid-LavaGapS6-v0",


    "MiniGrid-MultiRoom-N2-S4-v0",
    "MiniGrid-MultiRoom-N4-S5-v0",

    "MiniGrid-ObstructedMaze-1Dlhb-v0",

    "MiniGrid-RedBlueDoors-6x6-v0",

    "MiniGrid-Unlock-v0",

    "MiniGrid-BlockedUnlockPickup-v0",

]


def config_from_env_id(env_id: str) -> ExperimentConfig:
    return ExperimentConfig(
        env_id=env_id,
        algo=algo_hparams_fn(),
        name=env_id.replace("MiniGrid-", "").lower(),
    )


if __name__ == "__main__":
    num_runs = len(ENVS_IDS)
    for i, env_id in enumerate(ENVS_IDS):
        config = config_from_env_id(env_id)
        print(f"Training env {env_id}, num {i}/{num_runs}, experiment name {config.name}")
        run_training_experiment(config, logdir=LOGS_PATH, expdir=EXPERIMENT_PATH)
