# Exploration-heavy Reinforcement Learning for Grid Environments

Just implementing and playing with RL algorithms, with a focus on exploration, novelty seeking and open endedness for grid environment with sparse rewards ([Minigrid](https://github.com/Farama-Foundation/Minigrid) and potentially [Minihack](https://github.com/samvelyan/minihack) and [Crafter](https://github.com/danijar/crafter)).

While the codebase is planned to be focused on exploration, reward free and novelty seeking algorithms, only a Proximal Policy Optimization (PPO) baseline implementation (a RNN version is also available).


## Installation

Simply install the dependencies from the `pyproject.toml`, [uv](https://docs.astral.sh/uv/) has been used and it is recommended:

```bash
uv pip install -r pyproject.toml
```


Alternatively, a .venv can be created and all the exact dependencies versions autoamtically installed with simply:

```bash
uv sync
```


## Features

Training in vectorized MiniGrid environments environmnets, only `image` and `direction` observations are used, `mission` string observation is currently ignored, so training of environments that require language may not be solvable.

A basic experiment tracking is built-in, models checkpoints are stored, training stats are logged to a csv and Tensorboard, config files with algorithms and models hyperparameters, as well metadata (time and commit hash) are stored.

An evaluation tool is also included, it supports the evaluation of random policies on any environment, the evaluation of existing training experiments (with automatic loading of the best checkpoint or manual selection) and optionally rending of the environment for humans.


## Usage

### Training

The main entry point for training is the `train.py` script. The algorithm and its hyperparameters can be selected via the command line interface.

Example usage:

```bash
python train.py --name experiment_name --env_id MiniGrid-Empty-5x5-v0 --algo ppo --total_steps 10000 --horizon 128 --num_envs 8

```

To see available arguments:

```bash
python train.py --help
```

The used algorithm can be selected with the `--algo` parameters, hyperparameters arguments change depending on the selected algorithm. To see arguments for a specific algorithm (in this case the `rnn_ppo`):


```bash
python train.py --algo rnn_ppo --help
```
The [simple_parsing](https://github.com/lebrice/SimpleParsing/tree/master) library is used to automatically generate an argument parser from dataclasses.

If a model configuration is not provided, a default model is associated with the selected algorithm, default hyper parameters are used. To provide a custom model configuration, use the `--model_config` argument with the path to a JSON file containing the model hyperparameters. An example of configuration for a `conv_actor_critic` model based policy is:


```json
{
    "type": "conv_actor_critic",
    "hparams": {
        "policy_units": 64,
        "dense_layers": [32, 32]
    }
}
```

Each available model has an associated hyperparameters dataclass, this JSON `"haparams"` fields are deserialized into the corresponding model hyperparameters dataclass, if no some field is not included the default values are taken.

Available models:

  * `conv_actor_critic`, config dataclass: `grid_exp_oe.models.conv_actor_critic.ConvActorCriticHParams`.
  * `lstm_conv_actor_critic`, config dataclass: `grid_exp_oe.models.lstm_conv_actor_critic.LSTMConvActorCriticHParams`, only compatible with `rnn_ppo`.


### Evaluation

The `evaluate.py` script is used to evaluate the performance of trained models. It supports both single and vectorized environments.

Example usage:

```bash
python evaluate.py --experiment path_to/experiment_folder --total_steps 10000 --num_envs 8
```

If no `--ckpt` is passed, the best performing checkpoint is automatically loaded and if no `--env_id` is passed, the same environment used for training in the experiment is selected.

The `--human_render` flag is set, then only one environment is used and the environment is rendered for humans.

You may not pass an `--experiment` and set the `--random` flag to evaluate a random policy (an `--env_id` is required too).


## Future Work

- Implement additional algorithms, such as:
   * RND
   * Intrinsic motivation
   * NSR-ES
   * MAP Elites
   * Active Inference inspired agents
- Add more models.
- Add `mission` observation in MiniGrid, try other 2D grid environments.
- Optimize the performance, by optimizing implementations, using better vectorization (e.g by using [PufferLib](https://github.com/PufferAI/PufferLib)), or GPU based environments.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


