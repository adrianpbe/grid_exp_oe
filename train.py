from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
import subprocess

from simple_parsing import ArgumentParser, subgroups

from grid_exp_oe.base import AlgorithmHParams
from grid_exp_oe.env import create_vectorized_env
from grid_exp_oe.models import ModelHparams, get_model_builder
from grid_exp_oe.ppo import PPOHparams, RNNPPOHparams, train


def _get_commit() -> str:
    """Get the commit hash of the repository, requires having git"""
    repo_path = os.path.dirname(os.path.realpath(__file__))
    process = subprocess.run(["git", "-C", repo_path, "rev-parse", "HEAD"], capture_output=True)
    commit_hash = process.stdout.decode("utf-8").strip()
    return commit_hash


AVAILABLE_ALGORITHMS_HPARAMS = {
    "ppo": PPOHparams,
    "rnn_ppo": RNNPPOHparams,

}


DEFAULT_MODEL_BUILDER = {
    "ppo": "conv_actor_critic",
    "rnn_ppo": "lstm_conv_actor_critic",
}


@dataclass
class ModelFileConfig:
    type: str
    hparams: ModelHparams


@dataclass
class ExperimentConfig:
    env_id: str
    algo: AlgorithmHParams = subgroups(AVAILABLE_ALGORITHMS_HPARAMS, default="ppo")
    name: str | None = None
    env_seed: int | None = None
    model_config: str | None = None


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="experiment")
    parser.add_argument("--logdir", type=str, default="logs", 
                        help="path to Tensorboard logs folder (a new folder for the current experiment is created within)")
    parser.add_argument("--expdir", type=str, default="experiments", 
                        help="path to experiments folder (a new folder for the current experiment is created within)")
    args = parser.parse_args()


    experiment: ExperimentConfig = args.experiment

    algo_hparams = experiment.algo
    algo_id = algo_hparams.algo_id()
    print(f"training algorithm {algo_id}")
    envs = create_vectorized_env(experiment.env_id, algo_hparams.num_envs)

    config_file = experiment.model_config

    if config_file is None:
        model_id = DEFAULT_MODEL_BUILDER[algo_id]
        model_config_data = None
        print(f"Using default model {model_id} for {algo_id}")
    else:
        with open(config_file, "r") as f:
            model_config_data = json.load(f)
        model_id = model_config_data["type"]
        model_config_data = model_config_data["hparams"]

    model_builder = get_model_builder(model_id, model_config_data)

    exp_name = experiment.name
    exp_time_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    logdir = os.path.join(
        args.logdir,
        exp_name + "_" + exp_time_str if exp_name is not None else exp_time_str 
    )

    expdir = os.path.join(
        args.expdir,
        exp_name + "_" + exp_time_str if exp_name is not None else exp_time_str 
    )
    ckptdir = os.path.join(expdir, "checkpoints")

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.expdir, exist_ok=True)

    os.makedirs(logdir)
    os.makedirs(expdir)
    os.makedirs(ckptdir)

    all_configs = {
        "experiment": {
            "name": experiment.name,
            "env_id": experiment.env_id,
            "env_seed": experiment.env_seed,
        },
        "algorithm": {
            "algorithm_id": algo_id,
            "hparams": asdict(algo_hparams),
        },
        "model": {
            "model_id": model_id,
            "hparams": asdict(model_builder.hparams)
        },
        "metadata": {
            "time": exp_time_str
        }
    }
    try:
        commit_hash = _get_commit()
    except FileExistsError as e:
        print("Commit hash could not be found, git is required")
        pass
    else:
        all_configs["metadata"]["commit"] = commit_hash

    store_cfg = os.path.join(expdir, "config.json")
    with open(store_cfg, "w") as f:
        json.dump(all_configs, f, indent=4)

    policy, stats = train(
        model_builder, algo_hparams, envs,
        experimentdir=expdir,
        ckptdir=ckptdir,
        logdir=logdir,
        env_seed=experiment.env_seed
    )
