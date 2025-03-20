from simple_parsing import ArgumentParser
from grid_exp_oe import ExperimentConfig, run_training_experiment


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="experiment")
    parser.add_argument("--logdir", type=str, default="logs", 
                        help="path to Tensorboard logs folder (a new folder for the current experiment is created within)")
    parser.add_argument("--expdir", type=str, default="experiments", 
                        help="path to experiments folder (a new folder for the current experiment is created within)")
    args = parser.parse_args()

    experiment: ExperimentConfig = args.experiment
    expdir = args.expdir
    logdir = args.logdir

    run_training_experiment(experiment, logdir=logdir, expdir=expdir)
