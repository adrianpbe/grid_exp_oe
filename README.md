# Exploration-heavy Reinforcement Learning for Grid Environments

Just implementing and playing with RL algorithms, with a focus on exploration, novelty seeking and open endedness, for grid environment with sparse rewards ([Minigrid](https://github.com/Farama-Foundation/Minigrid) and potentially [Minihack](https://github.com/samvelyan/minihack) and [Crafter](https://github.com/danijar/crafter)).


## Installation

Simply install the dependencies from the `pyproject.toml`, [uv](https://docs.astral.sh/uv/) has been used and it is recommended:

```bash
uv pip install -r pyproject.toml
```


Alternatively, a `.venv` can be created and all exact versions of the dependencies automatically installed with simply:

```bash
uv sync
```


## Features

Training in vectorized MiniGrid environments, only `image` and `direction` observations are used, `mission` string observation is currently ignored, so environments that require language may not be solvable.

A basic experiment tracking is built-in, models checkpoints are stored, training stats are logged to a CSV file and to Tensorboard, config files with algorithms and models hyperparameters, as well metadata (time and commit hash) are stored.

An evaluation tool is also included, it supports the evaluation of random policies on any environment, the evaluation of existing training experiments (with automatic loading of the best checkpoint or manual selection) and optionally rendering of the environment for humans.

### Algorithms

Implemented algorithms are listed here, to see how to train them see the next section.

* [PPO](https://arxiv.org/abs/1707.06347): Stable policy gradient method, useful as a baseline for comparison. Algorithm id: `ppo`.
  * There is a recurrent version with algorithm id `rnn_ppo`.
* [RND](https://arxiv.org/abs/1810.12894): It's a PPO implementation that includes the RND exploration strategy: it uses an intrinsic reward to encourage exploration based on the distillation of a random network that outputs a random embedding of observations. Algorithm id: `rnd_ppo`.


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

If a model configuration is not provided, a default model with default hyperparameters is automatically selected based on the selected algorithm. To provide a custom model configuration, use the `--model_config` argument with the path to a JSON file containing the model hyperparameters. An example of configuration for a `conv_actor_critic` model based policy is:


```json
{
    "type": "conv_actor_critic",
    "hparams": {
        "policy_units": 64,
        "dense_layers": [32, 32]
    }
}
```

Each available model has an associated hyperparameters dataclass, this JSON `"hparams"` field is parsed into the corresponding model hyperparameters dataclass, if some field is not included it default value is set.

**Available models:**

  * `conv_actor_critic`, config dataclass: `grid_exp_oe.models.conv_actor_critic.ConvActorCriticHParams`.
  * `lstm_conv_actor_critic`, config dataclass: `grid_exp_oe.models.lstm_conv_actor_critic.LSTMConvActorCriticHParams`, only compatible with `rnn_ppo`.

#### Training performance

Training performance is almost always higher using CPU, so GPU is ignored by default. It can be activated with the flag `--use_gpu`.

### Evaluation

The `evaluate.py` script is used to evaluate the performance of trained models. It supports both single and vectorized environments.

Example usage:

```bash
python evaluate.py --experiment path_to/experiment_folder --total_steps 10000 --num_envs 8
```

If no `--ckpt` is passed, the best performing checkpoint is automatically loaded and if no `--env_id` is passed, the same environment used for training in the experiment is selected.

If the `--human_render` flag is set, then only one environment is used and the environment is rendered for humans.

You can ignore the `--experiment` argument and set the `--random` flag to evaluate a random policy if the environment set by the `--env_id` argument.


## Future Work

- Implement additional algorithms, such as:
   * RND
   * Intrinsic motivation
   * NSR-ES
   * MAP Elites
   * Active Inference inspired agents
- Add more models.
- Add `mission` observation in MiniGrid, try other 2D grid environments.
- Optimize the performance, by optimizing implementations, using better vectorization (e.g by using [PufferLib](https://github.com/PufferAI/PufferLib)) or GPU based environments.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

